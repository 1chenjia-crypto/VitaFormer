"""
train_v2_stratified.py — Training with stratified (day × has_dead) split.

Exp-S0: Stratified split + original config (baseline for new split).
  python train_v2_stratified.py \
    --exp_name expS0_stratified_baseline \
    --out_dir runs/expS0_stratified_baseline \
    --lr 1e-4 --lr_encoder 1e-5 --max_steps 20000 --val_every 500 \
    --freeze_blocks 4 --use_dead_center_crop --dead_weight_min 30

Exp-SB: Stratified split + Focal Loss + dead_weight=10 (main experiment).
  python train_v2_stratified.py \
    --exp_name expSB_stratified_focal \
    --out_dir runs/expSB_stratified_focal \
    --lr 1e-4 --lr_encoder 1e-5 --max_steps 40000 --val_every 500 \
    --freeze_blocks 4 --use_dead_center_crop \
    --focal_loss --gamma 2.0 --dead_weight_min 10

DO NOT tune on test set.
"""

import os, sys, time, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (build_dataset_index, stratified_train_val_test_split,
                     print_split_stats, NUM_CLASSES)
from dataset_v2 import PatchOrganoidDataset
from model_v3_multiscale import OrganoidSegMultiScale  # arch exploration 2026-11-04
from model_v2 import (
    OrganoidSegV2, TverskyLoss,
    compute_metrics_v2, compute_instance_metrics_fast,
)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# ── Lovász-Softmax Loss (Dead class only) ────────────────────────────────────

class LovaszDeadLoss(nn.Module):
    """Lovász-Softmax loss for the Dead class only.

    Directly optimises (1 - IoU_dead) as a differentiable surrogate.
    Reference: Berman et al., "The Lovász-Softmax Loss", CVPR 2018.
    """

    def __init__(self, dead_class=2, ignore_index=255):
        super().__init__()
        self.dead_class   = dead_class
        self.ignore_index = ignore_index

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Lovász extension gradient for a sorted binary error vector."""
        p    = len(gt_sorted)
        gts  = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union        = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(self, logits, targets):
        valid = (targets != self.ignore_index)
        if valid.sum() == 0:
            return logits.sum() * 0.

        probas      = F.softmax(logits, dim=1)           # (B,C,H,W)
        probas_flat = probas.permute(0,2,3,1).reshape(-1, probas.size(1))  # (N,C)
        targets_flat = targets.reshape(-1)
        valid_flat   = valid.reshape(-1)

        probas_flat  = probas_flat[valid_flat]
        targets_flat = targets_flat[valid_flat]

        fg = (targets_flat == self.dead_class).float()
        if fg.sum() == 0:
            return logits.sum() * 0.

        class_pred     = probas_flat[:, self.dead_class]
        errors         = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted      = fg[perm]
        return torch.dot(errors_sorted, self._lovasz_grad(fg_sorted))


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
        )
        valid = targets != self.ignore_index
        p_t = torch.exp(-ce.detach())
        focal = ((1 - p_t) ** self.gamma) * ce
        n_valid = valid.float().sum().clamp(min=1)
        return (focal * valid.float()).sum() / n_valid


# ── Combined loss supporting CE or Focal ─────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, use_focal=False, gamma=2.0,
                 ce_w=0.4, tversky_w=0.6, lovasz_w=0.0):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        if use_focal:
            self.ce_or_focal = FocalLoss(gamma=gamma, weight=w, ignore_index=255)
        else:
            self.ce_or_focal = nn.CrossEntropyLoss(weight=w, ignore_index=255)
        self.tversky  = TverskyLoss()
        self.ce_w     = ce_w
        self.tversky_w = tversky_w
        self.lovasz_w  = lovasz_w
        self.lovasz    = LovaszDeadLoss(dead_class=2, ignore_index=255) if lovasz_w > 0 else None

    def forward(self, seg_logits, targets, **_):
        mask = targets != 255
        targets_clean = targets.clone()
        targets_clean[~mask] = 0

        loss_ce = self.ce_or_focal(seg_logits, targets)
        loss_tv = self.tversky(seg_logits, targets_clean)
        total   = self.ce_w * loss_ce + self.tversky_w * loss_tv

        loss_lv = 0.0
        if self.lovasz is not None and self.lovasz_w > 0:
            loss_lv = self.lovasz(seg_logits, targets)
            total   = total + self.lovasz_w * loss_lv

        return {
            'total':   total,
            'ce':      loss_ce.item(),
            'tversky': loss_tv.item(),
            'lovasz':  loss_lv.item() if isinstance(loss_lv, torch.Tensor) else 0.0,
        }


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='OrganoidSeg-v2 stratified split experiments')

    p.add_argument('--exp_name', default='expS_stratified',
                   help='Experiment name (used in summary)')

    # Paths
    p.add_argument('--data_root',
                   default='3D_OrganoidSeg/vco/data/VOC2012')
    p.add_argument('--biomedclip_weights',
                   default='pretrained/biomedclip/open_clip_pytorch_model.bin')
    p.add_argument('--out_dir', required=True)

    # Optional resume checkpoint
    p.add_argument('--resume', default=None,
                   help='Path to checkpoint for fine-tuning (loads model weights only, fresh optimizer)')
    p.add_argument('--resume_full', default=None,
                   help='Path to full checkpoint (model + optimizer + scheduler + step) for true resume')

    # Training hyperparams
    p.add_argument('--crop_size',    type=int,   default=224)
    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--max_steps',    type=int,   default=5000)
    p.add_argument('--lr',           type=float, default=1e-5)
    p.add_argument('--lr_encoder',   type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--arch', choices=('m0', 'm1', 'm2'), default='m0',
                   help='arch exploration 2026-11-04: m0=model_v2, m1=multiscale zero-CNN-input, m2=multiscale')
    p.add_argument('--val_every',    type=int,   default=100)
    p.add_argument('--oversample',   type=float, default=0.4)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--split_seed',   type=int,   default=42,
                   help='Fixed dataset split seed; keep constant across training seeds')
    p.add_argument('--gpu',          type=int,   default=0)

    # Model freeze
    p.add_argument('--freeze_blocks', type=int, default=4)
    p.add_argument('--no_mpm',        action='store_true', default=False,
                   help='Ablation: remove MPM morphology prior injection')
    p.add_argument('--no_bam',        action='store_true', default=False,
                   help='Ablation: remove boundary-aware refinement module')
    p.add_argument('--use_imagenet',  action='store_true', default=False,
                   help='Ablation: use ImageNet ViT-B/16 instead of BiomedCLIP')

    # Dead-center crop
    p.add_argument('--use_dead_center_crop',    action='store_true', default=False)
    p.add_argument('--dead_center_crop_prob',   type=float, default=0.6)
    p.add_argument('--min_dead_pixels_in_crop', type=int,   default=20)
    p.add_argument('--max_dead_crop_retry',     type=int,   default=20)

    # Dead pixel dropout augmentation
    p.add_argument('--dead_pixel_dropout_prob', type=float, default=0.0,
                   help='Prob of dropping 60-90%% of Dead pixels per crop (0=off)')

    # Sparse Dead zoom crop
    p.add_argument('--sparse_dead_zoom_prob',      type=float, default=0.0,
                   help='Prob of zoom-crop for images with <sparse_dead_threshold Dead px')
    p.add_argument('--sparse_dead_threshold',      type=int,   default=5000,
                   help='Total Dead pixel count below which zoom crop applies')
    p.add_argument('--sparse_dead_zoom_factor',    type=int,   default=2,
                   help='Zoom factor: crops crop_size/factor window then upsamples')

    # 5-Fu Dead oversampling
    p.add_argument('--oversample_5fu_factor', type=int, default=1,
                   help='Duplicate 5-Fu Dead training samples N times (1=no extra)')

    # Condition-aware split
    p.add_argument('--use_condition_split', action='store_true', default=False,
                   help='Use day × condition × has_dead stratification')

    # Method B: Focal Loss
    p.add_argument('--no_class_weights', action='store_true', default=False,
                   help='Ablation: use unweighted CE/Focal loss')
    p.add_argument('--no_rare_dead', action='store_true', default=False,
                   help='Ablation: disable all rare-Dead loss and sampling strategies')
    p.add_argument('--focal_loss',      action='store_true', default=False)
    p.add_argument('--gamma',           type=float, default=2.0)
    p.add_argument('--dead_weight_min', type=float, default=30.0,
                   help='Min class weight for Dead (30=original, 10=Method B)')

    # Lovász loss (directly optimises Dead IoU)
    p.add_argument('--lovasz_weight', type=float, default=0.0,
                   help='Weight for Lovász-Softmax Dead IoU loss (0=off, 0.5=recommended)')

    # Copy-Paste Dead augmentation
    p.add_argument('--copy_paste_dead_prob', type=float, default=0.0,
                   help='Probability of pasting a Dead crop onto current patch (0=off, 0.5=recommended)')

    # Instance metrics
    p.add_argument('--inst_min_px',     type=int,   default=20)
    p.add_argument('--inst_iou_thresh', type=float, default=0.5)

    return p.parse_args()


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(train_samples, num_classes=4, dead_weight_min=30.0):
    from PIL import Image as _PIL
    counts = np.zeros(num_classes, dtype=np.float64)
    for s in train_samples[:200]:
        mask = np.array(_PIL.open(s['mask']))
        for c in range(num_classes):
            counts[c] += (mask == c).sum()
    total = counts.sum()
    if total == 0:
        return None
    freq = counts / total
    w = 1.0 / (freq + 1e-6)
    w = w / w.mean()
    w = np.clip(w, 0.5, 50.0)
    w[2] = max(w[2], dead_weight_min)
    print(f"Class weights: BG={w[0]:.1f} Alive={w[1]:.1f} Dead={w[2]:.1f} Bubble={w[3]:.1f}")
    print(f"  (dead_weight_min={dead_weight_min})")
    return w.tolist()


# ── Validation ────────────────────────────────────────────────────────────────

def evaluate(model, val_loader, criterion, device, inst_min_px=20, inst_iou_thresh=0.5):
    model.eval()
    all_preds, all_gts = [], []
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            seg_logits, _ = model(imgs)
            loss_dict = criterion(seg_logits, masks)
            total_loss += loss_dict['total'].item()
            all_preds.append(seg_logits.argmax(1).cpu())
            all_gts.append(masks.cpu())
            n += 1

    all_preds_cat = torch.cat(all_preds)
    all_gts_cat   = torch.cat(all_gts)

    metrics = compute_metrics_v2(all_preds_cat, all_gts_cat, num_classes=NUM_CLASSES)
    metrics['val_loss'] = total_loss / max(n, 1)

    inst = compute_instance_metrics_fast(
        all_preds, all_gts,
        cls=2, min_px=inst_min_px, iou_thresh=inst_iou_thresh,
    )
    metrics['dead_instance'] = inst
    return metrics


# ── Config banner ─────────────────────────────────────────────────────────────

def print_config(args, trainable_p, frozen_p):
    print("=" * 60)
    print(f"  OrganoidSeg-v2 Stratified Exp: {args.exp_name}")
    print("=" * 60)
    print(f"  out_dir          : {args.out_dir}")
    print(f"  exp_name         : {args.exp_name}")
    if args.resume:
        print(f"  resume from      : {args.resume}")
    print(f"  train / split seed: {args.seed} / {args.split_seed}")
    print(f"  freeze_blocks    : {args.freeze_blocks}")
    print(f"  trainable params : {trainable_p / 1e6:.2f} M")
    print(f"  frozen params    : {frozen_p / 1e6:.2f} M")
    print(f"  use_mpm / use_bam: {not args.no_mpm} / {not args.no_bam}")
    print(f"  encoder init     : {'ImageNet' if args.use_imagenet else 'BiomedCLIP'}")
    print(f"  focal_loss       : {args.focal_loss}  gamma={args.gamma}")
    print(f"  dead_weight_min  : {args.dead_weight_min}")
    print(f"  class_weights    : {not args.no_class_weights}")
    print(f"  use_dead_center_crop: {args.use_dead_center_crop}")
    print(f"  dead_pixel_dropout_prob: {args.dead_pixel_dropout_prob}")
    print(f"  sparse_dead_zoom_prob  : {args.sparse_dead_zoom_prob}  "
          f"threshold={args.sparse_dead_threshold}px  factor={args.sparse_dead_zoom_factor}x")
    print(f"  use_condition_split    : {args.use_condition_split}")
    print(f"  oversample_5fu_factor  : {args.oversample_5fu_factor}")
    print(f"  lovasz_weight          : {args.lovasz_weight}")
    print(f"  copy_paste_dead_prob   : {args.copy_paste_dead_prob}")
    print(f"  max_steps        : {args.max_steps}")
    print(f"  val_every        : {args.val_every}")
    print(f"  lr / lr_encoder  : {args.lr} / {args.lr_encoder}")
    print(f"  gpu              : {args.gpu}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.no_rare_dead:
        args.focal_loss = False
        args.no_class_weights = True
        args.use_dead_center_crop = False
        args.sparse_dead_zoom_prob = 0.0
        args.oversample_5fu_factor = 1
        args.oversample = 0.0
        args.dead_pixel_dropout_prob = 0.0
        args.copy_paste_dead_prob = 0.0

    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset / split
    all_samples, well_groups = build_dataset_index(args.data_root)
    split_desc = "day × condition × has_dead" if args.use_condition_split else "day × has_dead"
    print(f"Building stratified split ({split_desc})...")
    train_s, val_s, test_s = stratified_train_val_test_split(
        well_groups, 0.6, 0.2, seed=args.split_seed, use_condition=args.use_condition_split,
    )
    print_split_stats(train_s, val_s, test_s, "Stratified Split verification")
    print(f"Split: train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")

    # 5-Fu Dead oversampling
    if args.oversample_5fu_factor > 1:
        from PIL import Image as _PIL
        extras = []
        for s in train_s:
            cond = s.get('condition', '')
            if '5' in cond:  # matches '5-Fu', '5FU', etc.
                mask = np.array(_PIL.open(s['mask']))
                if (mask == 2).any():
                    extras.extend([s] * (args.oversample_5fu_factor - 1))
        print(f"5-Fu Dead oversampling: added {len(extras)} copies "
              f"(factor={args.oversample_5fu_factor})")
        train_s = train_s + extras

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(
            train_s, dead_weight_min=args.dead_weight_min)

    train_ds = PatchOrganoidDataset(
        train_s, args.crop_size, augment=True,
        minority_oversample=args.oversample,
        use_dead_center_crop=args.use_dead_center_crop,
        dead_center_crop_prob=args.dead_center_crop_prob,
        min_dead_pixels_in_crop=args.min_dead_pixels_in_crop,
        max_dead_crop_retry=args.max_dead_crop_retry,
        dead_pixel_dropout_prob=args.dead_pixel_dropout_prob,
        sparse_dead_zoom_prob=args.sparse_dead_zoom_prob,
        sparse_dead_threshold=args.sparse_dead_threshold,
        sparse_dead_zoom_factor=args.sparse_dead_zoom_factor,
        copy_paste_dead_prob=args.copy_paste_dead_prob,
    )
    val_ds = PatchOrganoidDataset(
        val_s, args.crop_size, augment=False,
        minority_oversample=0.0,
        use_dead_center_crop=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    w_path = args.biomedclip_weights
    if not os.path.exists(w_path):
        print(f"WARNING: BiomedCLIP weights not found at {w_path}, using random init")
        w_path = 'random'

    if args.arch == 'm0':
        model = OrganoidSegV2(
            weight_path=w_path,
            num_classes=NUM_CLASSES,
            freeze_blocks=args.freeze_blocks,
            use_mpm=not args.no_mpm,
            use_bam=not args.no_bam,
            use_imagenet=args.use_imagenet,
        ).to(device)
    else:
        assert args.no_mpm and args.no_bam, 'M1/M2 are defined without MPM/BAM'
        model = OrganoidSegMultiScale(
            w_path, num_classes=NUM_CLASSES, freeze_blocks=args.freeze_blocks,
            use_imagenet=args.use_imagenet, zero_cnn_input=(args.arch == 'm1'),
        ).to(device)
    print(f"  arch             : {args.arch}")

    # Fine-tune resume: load model weights only, fresh optimizer
    if args.resume:
        print(f"\nLoading checkpoint for fine-tune: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        state = ckpt.get('model_state', ckpt)
        model.load_state_dict(state, strict=True)
        src_step = ckpt.get('step', '?')
        print(f"  Loaded from step={src_step}, starting fresh optimizer at lr={args.lr}")

    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print_config(args, trainable_p, frozen_p)

    criterion = CombinedLoss(
        class_weights=class_weights,
        use_focal=args.focal_loss,
        gamma=args.gamma,
        ce_w=0.4,
        tversky_w=0.6,
        lovasz_w=args.lovasz_weight,
    ).to(device)

    enc_params  = [p for p in model.encoder.parameters() if p.requires_grad]
    rest_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in {id(q) for q in enc_params}]
    optimizer = optim.AdamW([
        {'params': enc_params,  'lr': args.lr_encoder},
        {'params': rest_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=1e-7,
    )

    # True resume: restore model + optimizer + scheduler + step
    start_step = 1
    best_miou = best_dead_iou = 0.0
    history = []
    if args.resume_full:
        print(f"\nFull resume from: {args.resume_full}")
        ckpt = torch.load(args.resume_full, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_step = ckpt['step'] + 1
        best_miou     = ckpt.get('best_miou', 0.0)
        best_dead_iou = ckpt.get('best_dead_iou', 0.0)
        history       = ckpt.get('history', [])
        print(f"  Resumed from step={ckpt['step']}  "
              f"best_miou={best_miou*100:.2f}%  best_dead_iou={best_dead_iou*100:.2f}%")

    # Sanity check
    _imgs, _masks, _ = next(iter(train_loader))
    _dead_ratio = (_masks == 2).float().mean().item()
    print(f"\n[Sanity] First batch dead_pixel_ratio = {_dead_ratio:.5f}  "
          f"({'Dead-center crop ON' if args.use_dead_center_crop else 'baseline'})")
    del _imgs, _masks

    # Training loop
    train_iter = iter(train_loader)
    t0 = time.time()
    dead_ratio_acc, dead_ratio_n = 0.0, 0

    print(f"\nTraining for {args.max_steps} steps ({args.exp_name}) "
          f"[starting from step {start_step}]...\n")

    for step in range(start_step, args.max_steps + 1):
        model.train()
        try:
            imgs, masks, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            imgs, masks, _ = next(train_iter)

        imgs, masks = imgs.to(device), masks.to(device)

        dead_ratio_acc += (masks == 2).float().mean().item()
        dead_ratio_n   += 1

        optimizer.zero_grad()
        seg_logits, _ = model(imgs)
        loss_dict = criterion(seg_logits, masks)
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            avg_dead_ratio = dead_ratio_acc / dead_ratio_n
            dead_ratio_acc, dead_ratio_n = 0.0, 0
            lv_str = (f"  Lv={loss_dict['lovasz']:.3f}"
                      if args.lovasz_weight > 0 else "")
            print(f"[{step:>6}/{args.max_steps}]  "
                  f"loss={loss_dict['total'].item():.4f}  "
                  f"CE={loss_dict['ce']:.3f}  Tv={loss_dict['tversky']:.3f}"
                  f"{lv_str}  dead_px_ratio={avg_dead_ratio:.5f}  "
                  f"({(time.time()-t0)/60:.1f} min)")

        if step % args.val_every == 0 or step == args.max_steps:
            m = evaluate(model, val_loader, criterion, device,
                         args.inst_min_px, args.inst_iou_thresh)

            miou      = m['mIoU']
            dead      = m['per_class']['dead']
            alive_iou = m['per_class']['alive']['iou']
            inst      = m['dead_instance']

            print(
                f"\n[VAL {step}]"
                f"  mIoU={miou*100:.2f}%"
                f"  Alive={alive_iou*100:.2f}%"
                f"  Dead IoU={dead['iou']*100:.2f}%"
                f"  Dead Prec={dead['precision']*100:.2f}%"
                f"  Dead Rec={dead['recall']*100:.2f}%"
                f"  Dead F1={dead['f1']*100:.2f}%"
                f"  Dead Dice={dead['dice']*100:.2f}%"
            )
            print(
                f"  [INST-Dead]"
                f"  TP={inst['tp']}  FP={inst['fp']}  FN={inst['fn']}"
                f"  Prec={inst['precision']*100:.1f}%"
                f"  Rec={inst['recall']*100:.1f}%"
                f"  F1={inst['f1']*100:.1f}%\n"
            )

            history.append({
                'step': step,
                'mIoU': miou,
                'mDice': m['mDice'],
                'val_loss': m['val_loss'],
                'per_class': m['per_class'],
                'dead_instance': inst,
            })

            if miou > best_miou:
                best_miou = miou
                torch.save(
                    {'step': step, 'model_state': model.state_dict(), 'metrics': m},
                    os.path.join(args.out_dir, 'best_miou.pth'),
                )
                print(f"  ↳ new best mIoU={miou*100:.2f}%")

            if dead['iou'] > best_dead_iou:
                best_dead_iou = dead['iou']
                torch.save(
                    {'step': step, 'model_state': model.state_dict(), 'metrics': m},
                    os.path.join(args.out_dir, 'best_dead_iou.pth'),
                )
                print(f"  ↳ new best Dead IoU={dead['iou']*100:.2f}%")

        # ── Full resume checkpoint every 2000 steps ───────────────────────────
        if step % 2000 == 0:
            torch.save({
                'step':            step,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_miou':       best_miou,
                'best_dead_iou':   best_dead_iou,
                'history':         history,
            }, os.path.join(args.out_dir, 'checkpoint_resume.pth'))
            print(f"  [RESUME CKPT saved at step {step}]")

    # Save final checkpoint and history
    torch.save(
        {'step': args.max_steps, 'model_state': model.state_dict()},
        os.path.join(args.out_dir, 'final.pth'),
    )
    with open(os.path.join(args.out_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Summary
    best_entry_miou = max(history, key=lambda e: e['mIoU'])
    best_entry_dead = max(history, key=lambda e: e['per_class']['dead']['iou'])

    summary = {
        'experiment':           os.path.basename(args.out_dir),
        'exp_name':            args.exp_name,
        'resume_from':          args.resume,
        'focal_loss':           args.focal_loss,
        'gamma':                args.gamma if args.focal_loss else None,
        'dead_weight_min':      args.dead_weight_min,
        'freeze_blocks':        args.freeze_blocks,
        'trainable_params_M':   round(trainable_p / 1e6, 2),
        'max_steps':            args.max_steps,
        'lr':                   args.lr,
        'lr_encoder':           args.lr_encoder,
        'best_val_miou_step':   best_entry_miou['step'],
        'best_val_mIoU':        round(best_entry_miou['mIoU'] * 100, 2),
        'best_val_dead_iou_step': best_entry_dead['step'],
        'best_val_dead_iou':    round(best_entry_dead['per_class']['dead']['iou'] * 100, 2),
        'dead_precision_at_best': round(best_entry_dead['per_class']['dead']['precision'] * 100, 2),
        'dead_recall_at_best':  round(best_entry_dead['per_class']['dead']['recall'] * 100, 2),
        'dead_f1_at_best':      round(best_entry_dead['per_class']['dead']['f1'] * 100, 2),
        'dead_dice_at_best':    round(best_entry_dead['per_class']['dead']['dice'] * 100, 2),
        'inst_dead_f1_at_best': round(best_entry_dead['dead_instance']['f1'] * 100, 2),
        'inst_dead_prec_at_best': round(best_entry_dead['dead_instance']['precision'] * 100, 2),
        'inst_dead_rec_at_best': round(best_entry_dead['dead_instance']['recall'] * 100, 2),
        'inst_dead_tp':         best_entry_dead['dead_instance']['tp'],
        'inst_dead_fp':         best_entry_dead['dead_instance']['fp'],
        'inst_dead_fn':         best_entry_dead['dead_instance']['fn'],
        'best_checkpoint':      os.path.join(args.out_dir, 'best_dead_iou.pth'),
    }

    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  STRATIFIED EXP SUMMARY: {args.exp_name}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<35s}: {v}")
    print(f"\n  Summary → {summary_path}")
    print(f"  Best Dead checkpoint → {args.out_dir}/best_dead_iou.pth")
    print("=" * 60)
    print("TRAINING_COMPLETE")


if __name__ == '__main__':
    main()
