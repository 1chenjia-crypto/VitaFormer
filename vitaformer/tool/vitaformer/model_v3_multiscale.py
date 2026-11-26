"""VitaFormer architecture-exploration branch (2026-11-04). NEW FILE; model_v2.py is imported, never modified.

M0 (frozen baseline) = model_v2.OrganoidSegV2(use_mpm=False, use_bam=False):
    BiomedCLIP ViT-B/16 -> taps l4/l8/l12 (all 14x14) -> decoder
    dec1 14->28 (+l8) | dec2 28->56 (+l4) | dec3 56->112 (no skip) | dec4 112->224 (no skip) -> 1x1 head.

M2 (candidate) = M0 + a lightweight from-scratch CNN branch on the same normalised 224 input.
    CNN stage 1: 224->112, 32 ch | stage 2: 112->56, 64 ch | stage 3: 56->28, 128 ch
    (each stage: 3x3 conv stride 2 + BN + GELU, 3x3 conv + BN + GELU)
    Fusion by concatenation as an additional skip, at the decoder stage whose OUTPUT has that resolution:
        dec1 14->28  : skip = [l8 (256), c28 (128)]
        dec2 28->56  : skip = [l4 (128), c56 (64)]
        dec3 56->112 : skip = [c112 (32)]
        dec4 112->224: no skip (as M0)
    Decoder channel widths (256/128/64/32), ViT projections, head, encoder freezing: identical to M0.
    No BAM, no MPM, no deep supervision, no nnU-Net weights. 4-class output.

M1 (information control, run only if M2 passes stage B) = M2 with the CNN branch input set to zeros:
    identical modules, parameter count, FLOPs, optimiser groups and decoder input widths; the only difference to
    M2 is that no image information reaches the CNN skips. Note: with a decoder that keeps M0's topology, a
    "decoder-only" variant without high-resolution input is M0 itself; M1 additionally controls for the extra
    decoder capacity introduced by the wider skip inputs.

forward() returns (seg_logits, None) so train/eval code that unpacks `seg, _ = model(x)` works unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v2 import BiomedCLIPEncoder, DecoderBlock


def _stage(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
    )


class CNNDetailBranch(nn.Module):
    """Fixed design (locked 2026-11-04): 3 stride-2 stages, 32/64/128 channels, from scratch."""

    CHANNELS = (32, 64, 128)

    def __init__(self, in_ch=3):
        super().__init__()
        c1, c2, c3 = self.CHANNELS
        self.s1 = _stage(in_ch, c1)  # 1/2  (112 for 224 input)
        self.s2 = _stage(c1, c2)     # 1/4  (56)
        self.s3 = _stage(c2, c3)     # 1/8  (28)

    def forward(self, x):
        f112 = self.s1(x)
        f56 = self.s2(f112)
        f28 = self.s3(f56)
        return f112, f56, f28


class MultiScaleDecoder(nn.Module):
    def __init__(self, vit_ch=768, cnn_ch=CNNDetailBranch.CHANNELS):
        super().__init__()
        c112, c56, c28 = cnn_ch
        self.proj_l12 = nn.Sequential(nn.Conv2d(vit_ch, 256, 1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.proj_l8 = nn.Sequential(nn.Conv2d(vit_ch, 256, 1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.proj_l4 = nn.Sequential(nn.Conv2d(vit_ch, 128, 1, bias=False), nn.BatchNorm2d(128), nn.GELU())
        self.dec1 = DecoderBlock(256, 256 + c28, 256)   # 14 -> 28
        self.dec2 = DecoderBlock(256, 128 + c56, 128)   # 28 -> 56
        self.dec3 = DecoderBlock(128, c112, 64)         # 56 -> 112
        self.dec4 = DecoderBlock(64, 0, 32)             # 112 -> 224

    @staticmethod
    def _fit(t, ref):
        return t if t.shape[-2:] == ref.shape[-2:] else F.interpolate(t, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, feats, cnn):
        f112, f56, f28 = cnn
        l4, l8, l12 = self.proj_l4(feats['l4']), self.proj_l8(feats['l8']), self.proj_l12(feats['l12'])
        x = self.dec1(l12, torch.cat([self._fit(l8, f28), f28], 1))
        x = self.dec2(x, torch.cat([self._fit(l4, f56), f56], 1))
        x = self.dec3(x, f112)
        return self.dec4(x)


class OrganoidSegMultiScale(nn.Module):
    """M2 (zero_cnn_input=False) or M1 (zero_cnn_input=True)."""

    def __init__(self, weight_path, num_classes=4, freeze_blocks=4, use_imagenet=False, zero_cnn_input=False):
        super().__init__()
        self.num_classes = num_classes
        self.zero_cnn_input = zero_cnn_input
        self.encoder = BiomedCLIPEncoder(weight_path, freeze_blocks=freeze_blocks, use_imagenet=use_imagenet)
        self.cnn = CNNDetailBranch(3)
        self.decoder = MultiScaleDecoder(vit_ch=768)
        self.seg_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_224 = x if (H, W) == (224, 224) else F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feats = self.encoder(x_224)
        cnn_in = torch.zeros_like(x_224) if self.zero_cnn_input else x_224
        dec = self.decoder(feats, self.cnn(cnn_in))
        seg = self.seg_head(dec)
        if (H, W) != (224, 224):
            seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=False)
        return seg, None
