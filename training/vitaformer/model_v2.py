"""
model_v2.py — OrganoidSeg-v2

Architecture:
  BiomedCLIP ViT-B/16 encoder (multi-scale features from layers 4, 8, 12)
  + Boundary-Aware Module (BAM): gradient-based attention
  + Morphology Prior Module (MPM): pellucidity / entropy / gradient features
  + Hierarchical Decoder: token → spatial, FPN-style upsampling
  + Dual heads: segmentation (4-class) + morphology output (interpretability)

Loss:
  ClassTversky: α=0.7 (penalize FP) for Dead; standard Dice for others
  CrossEntropy with class weights
  Optional morphology auxiliary head (no ground truth needed, self-supervised)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# BiomedCLIP encoder: ViT-B/16 with intermediate hooks
# ─────────────────────────────────────────────────────────

class BiomedCLIPEncoder(nn.Module):
    """
    BiomedCLIP visual encoder using timm vit_base_patch16_224.

    BiomedCLIP stores visual weights under "visual.trunk.*" in the .bin file.
    We load only those keys into a plain timm ViT-B/16, skipping the text encoder.
    Hooks are placed at blocks 3, 7, 11 (0-indexed = layers 4, 8, 12).
    """

    def __init__(self, weight_path: str, freeze_blocks: int = 8, use_imagenet: bool = False):
        super().__init__()
        import timm

        # use_imagenet=True → standard ImageNet ViT-B/16 (ablation: w/o BiomedCLIP)
        if use_imagenet:
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
            imagenet_path = os.environ.get(
                'IMAGENET_VIT_B16_WEIGHTS',
                'pretrained/torch_hub/checkpoints/vit_b_16-c867db91.pth')
            from imagenet_weights import load_torchvision_vit_into_timm
            load_torchvision_vit_into_timm(self.vit, imagenet_path)
            print(f"[BiomedCLIPEncoder] Using ImageNet pretrained ViT-B/16: {imagenet_path}")
        else:
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
            # Load BiomedCLIP visual weights
            if weight_path and weight_path != 'random':
                self._load_biomedclip_weights(weight_path)

        # Freeze early blocks
        for p in self.vit.patch_embed.parameters():
            p.requires_grad_(False)
        for blk in list(self.vit.blocks)[:freeze_blocks]:
            for p in blk.parameters():
                p.requires_grad_(False)

        # Hooks to capture intermediate token features
        self._features: dict = {}
        self.vit.blocks[3].register_forward_hook(self._make_hook('l4'))
        self.vit.blocks[7].register_forward_hook(self._make_hook('l8'))
        self.vit.blocks[11].register_forward_hook(self._make_hook('l12'))

    def _load_biomedclip_weights(self, weight_path: str):
        import torch as _torch
        state = _torch.load(weight_path, map_location='cpu', weights_only=True)

        # BiomedCLIP stores visual weights under "visual.trunk.*"
        prefix = 'visual.trunk.'
        vit_state = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix)
        }
        if not vit_state:
            # Fallback: try "visual.*" (standard open_clip format)
            prefix = 'visual.'
            vit_state = {
                k[len(prefix):]: v
                for k, v in state.items()
                if k.startswith(prefix) and not k.startswith('visual.proj')
            }

        missing, unexpected = self.vit.load_state_dict(vit_state, strict=False)
        loaded = len(vit_state) - len(unexpected)
        print(f"[BiomedCLIPEncoder] Loaded {loaded}/{len(vit_state)} keys "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self._features[name] = out
        return hook

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, 3, 224, 224) normalized
        Returns: {'l4', 'l8', 'l12'} each (B, 768, 14, 14)
        """
        self._features.clear()
        self.vit.forward_features(x)  # populates hooks

        out = {}
        for key in ('l4', 'l8', 'l12'):
            tok = self._features[key]      # (B, 197, 768) incl. CLS
            tok = tok[:, 1:, :]            # drop CLS → (B, 196, 768)
            B, N, C = tok.shape
            H = W = int(math.isqrt(N))    # 14
            out[key] = tok.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return out


# ─────────────────────────────────────────────────────────
# Morphology Prior Module (MPM)
# ─────────────────────────────────────────────────────────

class MorphologyPriorModule(nn.Module):
    """
    Compute image-derived morphological features for interpretability:
      - Pellucidity: contrast between 3x3 center and 7x7 surround
      - Entropy: local Shannon entropy in 9x9 window
      - Gradient: Sobel magnitude

    These are NOT trainable; computed analytically from the grayscale input.
    They are injected into the decoder as prior information.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0))

        # Center / surround kernels for pellucidity
        center = torch.zeros(1, 1, 7, 7)
        center[0, 0, 3, 3] = 1.0
        surround = torch.ones(1, 1, 7, 7) / 49.0
        self.register_buffer('center_k', center)
        self.register_buffer('surround_k', surround)

    @staticmethod
    def _to_gray(x: torch.Tensor) -> torch.Tensor:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def _pellucidity(self, gray: torch.Tensor) -> torch.Tensor:
        """Center minus local mean: bright center on darker surround → high pellucidity."""
        center_val = F.conv2d(gray, self.center_k, padding=3)
        mean_val = F.conv2d(gray, self.surround_k, padding=3)
        return torch.abs(center_val - mean_val)

    def _local_entropy(self, gray: torch.Tensor) -> torch.Tensor:
        """Approximate local entropy via variance proxy (avoids expensive histogram)."""
        mu = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        sq_diff = F.avg_pool2d((gray - mu) ** 2, kernel_size=9, stride=1, padding=4)
        return sq_diff.clamp(min=1e-6).log()

    def _gradient(self, gray: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) normalized RGB
        Returns: (B, 3, H, W) — [pellucidity, entropy, gradient], each in [0,1]
        """
        gray = self._to_gray(x)
        pell = self._pellucidity(gray)
        entr = self._local_entropy(gray)
        grad = self._gradient(gray)

        def _norm(t):
            t_min = t.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
            t_max = t.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
            return (t - t_min) / (t_max - t_min + 1e-6)

        return torch.cat([_norm(pell), _norm(entr), _norm(grad)], dim=1)


# ─────────────────────────────────────────────────────────
# Boundary-Aware Module (BAM)
# ─────────────────────────────────────────────────────────

class BoundaryAwareModule(nn.Module):
    """
    Compute boundary attention from input image gradient.
    Multiply decoder features by (1 + boundary_weight * attn),
    so boundaries get amplified.
    """

    def __init__(self, feat_ch: int, boundary_weight: float = 0.5):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.refine = nn.Sequential(
            nn.Conv2d(feat_ch + 1, feat_ch, 1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.GELU(),
        )
        # Sobel x and y
        sx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sy = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.register_buffer('sx', sx.unsqueeze(0))
        self.register_buffer('sy', sy.unsqueeze(0))

    def _boundary_map(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gray_r = F.interpolate(gray, size=(h, w), mode='bilinear', align_corners=False)
        gx = F.conv2d(gray_r, self.sx, padding=1)
        gy = F.conv2d(gray_r, self.sy, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        mag = (mag - mag.flatten(2).min(dim=2)[0].view(-1, 1, 1, 1)) / \
              (mag.flatten(2).max(dim=2)[0].view(-1, 1, 1, 1) - mag.flatten(2).min(dim=2)[0].view(-1, 1, 1, 1) + 1e-6)
        return mag

    def forward(self, feat: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        boundary = self._boundary_map(x_orig, H, W)  # (B, 1, H, W)
        feat_cat = torch.cat([feat, boundary], dim=1)
        return self.refine(feat_cat)


# ─────────────────────────────────────────────────────────
# Decoder block
# ─────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            # Resize skip to match upsampled x if resolutions differ
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────
# Hierarchical Decoder
# ─────────────────────────────────────────────────────────

class HierarchicalDecoder(nn.Module):
    """
    Input: ViT features l4/l8/l12 each (B, 768, 14, 14) for 224x224 input
    Upsample path: 14→28→56→112→224 with skip connections

    MPM features (B, 3, H, W) injected at 56x56 stage.
    BAM applied at 112x112 stage.
    """

    def __init__(self, vit_ch: int = 768, morph_ch: int = 3, use_mpm: bool = True):
        super().__init__()
        self.use_mpm = use_mpm

        # Project each ViT layer to 256-ch
        self.proj_l12 = nn.Sequential(
            nn.Conv2d(vit_ch, 256, 1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.proj_l8 = nn.Sequential(
            nn.Conv2d(vit_ch, 256, 1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.proj_l4 = nn.Sequential(
            nn.Conv2d(vit_ch, 128, 1, bias=False), nn.BatchNorm2d(128), nn.GELU())

        # Decoder: 14→28 (l12 + l8 skip)
        self.dec1 = DecoderBlock(256, 256, 256)
        # 28→56 (+ l4 projected to 128 skip)
        self.dec2 = DecoderBlock(256, 128, 128)
        # 56→112: MPM skip (16ch) when use_mpm=True, else no skip (0ch)
        if use_mpm:
            self.morph_proj = nn.Sequential(
                nn.Conv2d(morph_ch, 16, 1, bias=False), nn.BatchNorm2d(16), nn.GELU())
            self.dec3 = DecoderBlock(128, 16, 64)
        else:
            self.dec3 = DecoderBlock(128, 0, 64)
        # 112→224 (final upsampling)
        self.dec4 = DecoderBlock(64, 0, 32)

    def forward(self, feats: dict, morph: torch.Tensor) -> torch.Tensor:
        l4 = self.proj_l4(feats['l4'])    # (B, 128, 14, 14)
        l8 = self.proj_l8(feats['l8'])    # (B, 256, 14, 14)
        l12 = self.proj_l12(feats['l12'])  # (B, 256, 14, 14)

        # 14 → 28: use l8 as skip
        x = self.dec1(l12, l8)            # (B, 256, 28, 28)
        # 28 → 56: use l4 as skip
        x = self.dec2(x, l4)             # (B, 128, 56, 56)
        # 56 → 112: inject MPM only when use_mpm=True
        if self.use_mpm:
            morph_56 = F.interpolate(morph, size=(56, 56), mode='bilinear', align_corners=False)
            morph_56 = self.morph_proj(morph_56)  # (B, 16, 56, 56)
            x = self.dec3(x, morph_56)    # (B, 64, 112, 112)
        else:
            x = self.dec3(x)              # (B, 64, 112, 112), no MPM skip
        # 112 → 224
        x = self.dec4(x)                 # (B, 32, 224, 224)
        return x


# ─────────────────────────────────────────────────────────
# OrganoidSeg-v2: full model
# ─────────────────────────────────────────────────────────

class OrganoidSegV2(nn.Module):
    """
    Full OrganoidSeg-v2 model.

    weight_path: path to open_clip_pytorch_model.bin (BiomedCLIP weights)
    num_classes: 4 (BG=0, alive=1, dead=2, bubble=3)
    freeze_blocks: freeze first N ViT blocks (default 8, fine-tune last 4)
    """

    # Normalization constants from BiomedCLIP config
    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    def __init__(self, weight_path: str, num_classes: int = 4, freeze_blocks: int = 8,
                 use_mpm: bool = True, use_bam: bool = True,
                 use_imagenet: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.use_mpm = use_mpm
        self.use_bam = use_bam

        self.encoder = BiomedCLIPEncoder(weight_path, freeze_blocks=freeze_blocks,
                                         use_imagenet=use_imagenet)
        self.mpm = MorphologyPriorModule()
        self.decoder = HierarchicalDecoder(vit_ch=768, morph_ch=3, use_mpm=use_mpm)
        self.bam = BoundaryAwareModule(feat_ch=32) if use_bam else nn.Identity()

        self.seg_head = nn.Conv2d(32, num_classes, 1)
        self.morph_head = nn.Conv2d(32, 3, 1)  # interpretability output

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W) — already normalized (ImageNet stats from PatchOrganoidDataset).
            If H/W != 224, resized to 224 for the encoder.

        Returns:
            seg_logits:  (B, 4, H, W) — segmentation logits at original resolution
            morph_pred:  (B, 3, H, W) — morphology features (pellucidity, entropy, grad)
        """
        B, C, H, W = x.shape

        # Compute morphology on original (or crop) resolution before resize
        morph = self.mpm(x)  # (B, 3, H, W)

        # Resize to 224x224 for BiomedCLIP encoder (no extra normalization;
        # ImageNet stats ≈ BiomedCLIP stats, fine-tuning handles the small gap)
        x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Encode
        feats = self.encoder(x_224)  # {l4, l8, l12} each (B, 768, 14, 14)

        # Resize morph to 224x224 for decoder consistency
        morph_224 = F.interpolate(morph, size=(224, 224), mode='bilinear', align_corners=False)

        # Decode: output is (B, 32, 224, 224)
        dec_feat = self.decoder(feats, morph_224)

        # BAM: boundary attention on original x (already 224 internally)
        if self.use_bam:
            dec_feat = self.bam(dec_feat, x_224)
        else:
            dec_feat = self.bam(dec_feat)

        # Heads
        seg_224 = self.seg_head(dec_feat)    # (B, 4, 224, 224)
        morph_224_out = self.morph_head(dec_feat)  # (B, 3, 224, 224)

        # Upsample to original resolution if needed
        if H != 224 or W != 224:
            seg_out = F.interpolate(seg_224, size=(H, W), mode='bilinear', align_corners=False)
            morph_out = F.interpolate(morph_224_out, size=(H, W), mode='bilinear', align_corners=False)
        else:
            seg_out = seg_224
            morph_out = morph_224_out

        return seg_out, morph_out


# ─────────────────────────────────────────────────────────
# Loss: ClassTversky + CrossEntropy
# ─────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Per-class Tversky loss. For Dead (class 2): α=0.7 penalizes FP heavily
    to fix the precision=25% problem. For others: α=0.5 (symmetric Dice).
    """

    # α per class: [BG, alive, dead, bubble]
    ALPHA = [0.5, 0.5, 0.7, 0.5]
    BETA  = [0.5, 0.5, 0.3, 0.5]

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C, H, W)
        targets: (B, H, W) long
        """
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()

        loss = 0.0
        for c in range(num_classes):
            p = probs[:, c]
            t = targets_oh[:, c]
            alpha = self.ALPHA[c]
            beta = self.BETA[c]

            TP = (p * t).sum()
            FP = (p * (1 - t)).sum()
            FN = ((1 - p) * t).sum()

            tversky = (TP + self.smooth) / (TP + alpha * FP + beta * FN + self.smooth)
            loss += (1 - tversky)

        return loss / num_classes


class CombinedLossV2(nn.Module):
    """
    CE + TverskyLoss, weighted by class frequency.
    Optionally adds morphology consistency loss.
    """

    def __init__(self, class_weights=None, ce_w=0.4, tversky_w=0.6, morph_w=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32) if class_weights else None,
            ignore_index=255,
        )
        self.tversky = TverskyLoss()
        self.ce_w = ce_w
        self.tversky_w = tversky_w
        self.morph_w = morph_w

    def forward(
        self,
        seg_logits: torch.Tensor,
        targets: torch.Tensor,
        morph_pred: torch.Tensor = None,
        morph_gt: torch.Tensor = None,
    ) -> dict:
        mask = targets != 255
        targets_clean = targets.clone()
        targets_clean[~mask] = 0

        loss_ce = self.ce(seg_logits, targets_clean)
        loss_tv = self.tversky(seg_logits, targets_clean)

        total = self.ce_w * loss_ce + self.tversky_w * loss_tv

        if morph_pred is not None and morph_gt is not None and self.morph_w > 0:
            loss_morph = F.mse_loss(morph_pred, morph_gt)
            total = total + self.morph_w * loss_morph
        else:
            loss_morph = torch.tensor(0.0)

        return {
            'total': total,
            'ce': loss_ce.item(),
            'tversky': loss_tv.item(),
            'morph': loss_morph.item() if isinstance(loss_morph, torch.Tensor) else 0.0,
        }


# ─────────────────────────────────────────────────────────
# Helper: build model from config
# ─────────────────────────────────────────────────────────

def build_organoid_seg_v2(
    weight_path: str,
    num_classes: int = 4,
    freeze_blocks: int = 8,
    class_weights=None,
) -> tuple:
    """Returns (model, criterion)."""
    model = OrganoidSegV2(weight_path, num_classes=num_classes, freeze_blocks=freeze_blocks)
    criterion = CombinedLossV2(class_weights=class_weights)
    return model, criterion


# ─────────────────────────────────────────────────────────
# Metrics (same interface as model_3d.py)
# ─────────────────────────────────────────────────────────

def compute_metrics_v2(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int = 4):
    """
    pred_mask: (B, H, W) long
    gt_mask:   (B, H, W) long
    Returns dict with per-class IoU, Dice, Precision, Recall, F1, and mIoU/mDice.
    """
    iou_per_class  = []
    dice_per_class = []
    prec_per_class = []
    rec_per_class  = []
    f1_per_class   = []

    valid = gt_mask != 255

    for c in range(num_classes):
        pred_c = (pred_mask == c) & valid
        gt_c   = (gt_mask == c)   & valid
        inter  = (pred_c & gt_c).sum().item()
        union  = (pred_c | gt_c).sum().item()
        tp     = inter
        fp     = pred_c.sum().item() - tp
        fn     = gt_c.sum().item() - tp

        iou  = inter / (union + 1e-6) if union > 0 else float('nan')
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f1   = 2 * prec * rec / (prec + rec + 1e-6)

        iou_per_class.append(iou)
        dice_per_class.append(dice)
        prec_per_class.append(prec)
        rec_per_class.append(rec)
        f1_per_class.append(f1)

    import numpy as np
    valid_iou  = [v for v in iou_per_class  if not math.isnan(v)]
    valid_dice = [v for v in dice_per_class if not math.isnan(v)]
    miou  = float(np.mean(valid_iou))  if valid_iou  else 0.0
    mdice = float(np.mean(valid_dice)) if valid_dice else 0.0

    class_names = ['background', 'alive', 'dead', 'bubble']
    return {
        'mIoU':  miou,
        'mDice': mdice,
        'per_class': {
            class_names[c]: {
                'iou':       iou_per_class[c],
                'dice':      dice_per_class[c],
                'precision': prec_per_class[c],
                'recall':    rec_per_class[c],
                'f1':        f1_per_class[c],
            }
            for c in range(num_classes)
        }
    }


def compute_instance_metrics_fast(
    all_preds: list,
    all_gts:   list,
    cls: int = 2,
    min_px: int = 20,
    iou_thresh: float = 0.5,
) -> dict:
    """
    Fast crop-level instance metrics for class `cls` (default = Dead = 2).

    Processes each (pred_crop, gt_crop) pair independently; treats every
    connected component within a crop as a separate instance. Results are
    approximate (instances spanning crop boundaries are split) but run in
    O(n_crops) and give a useful training-time signal.

    Returns: tp, fp, fn, precision, recall, f1
    """
    from scipy import ndimage as _ndi
    total_tp = total_fp = total_fn = 0

    for pred, gt in zip(all_preds, all_gts):
        pred_np = (pred.numpy() == cls)
        gt_np   = (gt.numpy()   == cls)

        gt_lab,   n_gt   = _ndi.label(gt_np)
        pred_lab, n_pred = _ndi.label(pred_np)

        # Filter tiny components
        valid_gt   = [g for g in range(1, n_gt   + 1) if (gt_lab   == g).sum() >= min_px]
        valid_pred = [p for p in range(1, n_pred + 1) if (pred_lab == p).sum() >= min_px]

        matched_pred = set()
        tp = 0
        for g in valid_gt:
            g_mask = gt_lab == g
            for p in valid_pred:
                if p in matched_pred:
                    continue
                p_mask = pred_lab == p
                inter = (g_mask & p_mask).sum()
                union = (g_mask | p_mask).sum()
                if union > 0 and inter / union >= iou_thresh:
                    tp += 1
                    matched_pred.add(p)
                    break

        fn = len(valid_gt)   - tp
        fp = len(valid_pred) - len(matched_pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / (total_tp + total_fp + 1e-6)
    rec  = total_tp / (total_tp + total_fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    return {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': prec, 'recall': rec, 'f1': f1,
    }


if __name__ == '__main__':
    print("Testing model structure (random init, no BiomedCLIP weights)...")

    # Use 'random' to skip weight loading for smoke test
    model = OrganoidSegV2(weight_path='random', num_classes=4, freeze_blocks=8).eval()

    dummy = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        seg, morph = model(dummy)
    print(f"  Input:         {tuple(dummy.shape)}")
    print(f"  seg_logits:    {tuple(seg.shape)}   (expected: 2,4,256,256)")
    print(f"  morph_pred:    {tuple(morph.shape)} (expected: 2,3,256,256)")

    criterion = CombinedLossV2()
    gt = torch.randint(0, 4, (2, 256, 256))
    losses = criterion(seg, gt, morph, morph)
    print(f"  Loss total:    {losses['total']:.4f}")
    print(f"  Loss CE:       {losses['ce']:.4f}")
    print(f"  Loss Tversky:  {losses['tversky']:.4f}")
    print("\nSmoke test PASSED.")
