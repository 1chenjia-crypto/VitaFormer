"""Frozen VitaFormer M2 inference (no training, no thresholds, no post-processing).

The sliding-window routine is copied from the frozen functional-validation inference
(deepOrganoid_v1/extract_full_features.py `infer`, used in Stages 36-39): ImageNet normalisation, reflect padding,
224 px window, stride 112, softmax averaged over overlaps, argmax. fp16 autocast is used on CUDA only (CPU runs fp32),
so CPU and GPU masks can differ in a small number of boundary pixels.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # model_v3_multiscale imports model_v2 by name (files kept verbatim)
from model_v3_multiscale import OrganoidSegMultiScale  # noqa: E402

CLASSES = ('background', 'alive', 'dead', 'bubble')
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
WINDOW, STRIDE, BATCH = 224, 112, 64

# Frozen M2 checkpoints (MODEL_FREEZE_M2_20261108.md). The default model is seed 42.
KNOWN_WEIGHTS = {
    '0f12be44859d06723ebfa35e8416df8cd4b02ff83f3ac153654d0f68df226101': 'M2 seed 42 (default)',
    '1e4bd6afd0a1925094103c991430246b9f1fef09ce5b480e2c47056a7128f585': 'M2 seed 3407',
    'c268464ee2ae3ee19e2112fc91ae86a95aa5d95086374df526fda9a7aa3225b8': 'M2 seed 2026',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_model(weights: Path, device: torch.device, allow_unknown: bool = False):
    digest = sha256(weights)
    if digest not in KNOWN_WEIGHTS and not allow_unknown:
        raise RuntimeError(f'{weights}: SHA256 {digest} is not a frozen M2 checkpoint (use --allow-unknown-weights to override)')
    # The BiomedCLIP file is only needed for training; every weight comes from the M2 checkpoint (strict load).
    model = OrganoidSegMultiScale('random', num_classes=4, freeze_blocks=0, zero_cnn_input=False)
    ck = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(ck.get('model_state', ck), strict=True)
    return model.eval().to(device), digest, KNOWN_WEIGHTS.get(digest, 'unknown (not frozen M2)')


def to_rgb8(path: Path):
    """Legacy helper kept for compatibility; new code uses vitaformer.io.load_image (validates before converting)."""
    from .io import load_image
    rgb, info = load_image(Path(path))
    return rgb, info['preprocessing']


def _coords(h, w):
    ph = max(h, WINDOW); pw = max(w, WINDOW)
    ys = list(range(0, ph - WINDOW + 1, STRIDE)); xs = list(range(0, pw - WINDOW + 1, STRIDE))
    if not ys or ys[-1] + WINDOW < ph:
        ys.append(ph - WINDOW)
    if not xs or xs[-1] + WINDOW < pw:
        xs.append(pw - WINDOW)
    return [(y, x) for y in ys for x in xs], ph - h, pw - w


@torch.inference_mode()
def predict_probs(model, img: np.ndarray, device: torch.device) -> np.ndarray:
    """Copy of the frozen `infer`: returns (4, H, W) softmax averaged over overlapping windows."""
    h, w = img.shape[:2]
    cs, ph, pw = _coords(h, w)
    imgp = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    a = np.asarray(imgp, np.float32) / 255.0
    chw = torch.from_numpy(a).permute(2, 0, 1)
    chw = (chw - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    patches = torch.stack([chw[:, y:y + WINDOW, x:x + WINDOW] for y, x in cs])
    score = torch.zeros((4, h, w), device=device); cnt = torch.zeros((h, w), device=device)
    for st in range(0, len(cs), BATCH):
        ch = cs[st:st + BATCH]; b = patches[st:st + BATCH].to(device)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            o = model(b); o = o[0] if isinstance(o, tuple) else o
        pr = F.softmax(o.float(), 1)
        for i, (y, x) in enumerate(ch):
            y2 = min(y + WINDOW, h); x2 = min(x + WINDOW, w)
            score[:, y:y2, x:x2] += pr[i, :, :y2 - y, :x2 - x]; cnt[y:y2, x:x2] += 1
    return (score / torch.clamp(cnt, min=1e-6).unsqueeze(0)).cpu().numpy()


RESIZE_MODES = {
    'native': 'native resolution (validated: CRC, DeepOrganoid)',
    'factor_bilinear': 'scale factor f, PIL BILINEAR down-sizing, mask back with NEAREST (validated: GemcitabineScreen, f = 0.25)',
    'square_box': 'resize to S x S with PIL BOX, mask back with NEAREST; square images only (validated: NViR, S = 256)',
}


def predict_mask(model, img: np.ndarray, device: torch.device, mode: str = 'native', value: float | None = None) -> np.ndarray:
    """Hard 4-class mask at the input resolution. Only the three resize procedures used in the validated analyses are offered;
    each reproduces the original code exactly (see RESIZE_MODES). Choose the setting without using any functional readout."""
    h, w = img.shape[:2]
    if mode == 'native':
        return predict_probs(model, img, device).argmax(0).astype(np.uint8)
    if mode == 'factor_bilinear':
        f = float(value)
        if not 0 < f <= 1:
            raise ValueError('scale factor must be in (0, 1]')
        if f == 1.0:
            return predict_probs(model, img, device).argmax(0).astype(np.uint8)
        small = np.asarray(Image.fromarray(img).resize((round(w * f), round(h * f)), Image.BILINEAR))
    elif mode == 'square_box':
        S = int(value)
        if h != w:
            raise ValueError(f'square-side mode needs a square image (got {w} x {h})')
        if S > w:
            raise ValueError(f'target side {S} is larger than the image ({w}); only down-sizing was validated')
        small = np.asarray(Image.fromarray(img).resize((S, S), Image.BOX))
    else:
        raise ValueError(f'unknown resize mode {mode}')
    m = predict_probs(model, small, device).argmax(0).astype(np.uint8)
    return np.asarray(Image.fromarray(m).resize((w, h), Image.NEAREST))
