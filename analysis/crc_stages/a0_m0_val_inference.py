#!/usr/bin/env python3
"""A0 step 1: whole-image M0 (Simplified, 3 seeds) predictions on the 164 CRC *validation* images.

Why: M0's in-training validation used one random 224-px crop per image (dataset_v2.PatchOrganoidDataset,
augment=False), so training_history.json is not a whole-image validation metric and cannot serve as a
Go/No-Go noise band. nnU-Net whole-image validation masks already exist (fold_0/validation, checkpoint_final).

Inference protocol is copied unchanged from
scripts/run_primary_test_specialized_endpoints_20261029.py::infer_hard (224 window, 112 stride, reflect pad,
ImageNet norm, softmax averaging, argmax, autocast). Frozen checkpoints are only read.
Consistency check: before validation inference, 2 test images per seed are re-predicted and compared with the
existing P0-M masks (module_endpoints_multiseed_20261030/simplified_seed*/masks); expected pixel agreement ~100%.
Outputs go to a new directory only.
"""
import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('3D_OrganoidSeg')
DATA = Path('OrganoidSegBaseline/VOC2012')
IMG_DIR = DATA / 'JPEGImages'
VAL = DATA / 'ImageSets/Segmentation/val_cond.txt'
TEST = DATA / 'ImageSets/Segmentation/test_cond.txt'
BCLIP = Path('pretrained/biomedclip/open_clip_pytorch_model.bin')
OUT = ROOT / 'vco/runs/arch_exploration_20261104/A0_gap_decomposition/m0_val_masks'
P0M = ROOT / 'vco/runs/module_endpoints_multiseed_20261030'
sys.path.insert(0, str(ROOT / 'vco/organoid_seg'))
from model_v2 import OrganoidSegV2  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
WINDOW, STRIDE, BATCH = 224, 112, 128
SEEDS = [42, 3407, 2026]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def patch_coords(h, w):
    ph, pw = max(h, WINDOW), max(w, WINDOW)
    ys = list(range(0, ph - WINDOW + 1, STRIDE))
    xs = list(range(0, pw - WINDOW + 1, STRIDE))
    if not ys or ys[-1] + WINDOW < ph:
        ys.append(ph - WINDOW)
    if not xs or xs[-1] + WINDOW < pw:
        xs.append(pw - WINDOW)
    return [(y, x) for y in ys for x in xs], ph - h, pw - w


@torch.inference_mode()
def infer_hard(model, image, device):
    h, w = image.shape[:2]
    coords, pad_h, pad_w = patch_coords(h, w)
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1)
    tensor = (tensor - MEAN) / STD
    patches = torch.stack([tensor[:, y:y + WINDOW, x:x + WINDOW] for y, x in coords])
    score = torch.zeros((4, h, w), dtype=torch.float32, device=device)
    count = torch.zeros((h, w), dtype=torch.float32, device=device)
    for start in range(0, len(coords), BATCH):
        cc = coords[start:start + BATCH]
        batch = patches[start:start + BATCH].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            logits = model(batch)
            logits = logits[0] if isinstance(logits, tuple) else logits
        probs = F.softmax(logits.float(), dim=1)
        for j, (y, x) in enumerate(cc):
            y2, x2 = min(y + WINDOW, h), min(x + WINDOW, w)
            score[:, y:y2, x:x2] += probs[j, :, :y2 - y, :x2 - x]
            count[y:y2, x:x2] += 1
    return (score / count.clamp_min(1e-6).unsqueeze(0)).argmax(0).byte().cpu().numpy()


def main():
    device = torch.device('cuda')
    OUT.mkdir(parents=True, exist_ok=False)
    val = [s.strip() for s in VAL.read_text().splitlines() if s.strip()]
    test = [s.strip() for s in TEST.read_text().splitlines() if s.strip()]
    manifest = {'protocol': 'infer_hard copy (224/112, reflect, ImageNet norm, softmax avg, argmax, autocast)',
                'n_val': len(val), 'val_split_sha256': sha256(VAL), 'seeds': {}}
    for seed in SEEDS:
        ck = ROOT / f'vco/runs/ablation_no_bam_no_mpm_seed{seed}/best_miou.pth'
        model = OrganoidSegV2(weight_path=str(BCLIP), num_classes=4, freeze_blocks=0, use_mpm=False, use_bam=False)
        state = torch.load(ck, map_location='cpu', weights_only=True)
        model.load_state_dict(state.get('model_state', state), strict=True)
        model = model.eval().to(device)
        # consistency check vs existing P0-M test masks
        agree = []
        for stem in test[:2]:
            img = np.asarray(Image.open(IMG_DIR / f'{stem}.jpg').convert('RGB'))
            pred = infer_hard(model, img, device)
            ref = np.asarray(Image.open(P0M / f'simplified_seed{seed}/masks/{stem}.png'))
            agree.append(float((pred == ref).mean()))
        print(f'seed {seed}: consistency vs P0-M test masks = {agree}', flush=True)
        d = OUT / f'seed{seed}'
        d.mkdir()
        t0 = time.time()
        for i, stem in enumerate(val, 1):
            img = np.asarray(Image.open(IMG_DIR / f'{stem}.jpg').convert('RGB'))
            Image.fromarray(infer_hard(model, img, device)).save(d / f'{stem}.png')
            if i % 40 == 0:
                print(f'seed {seed}: {i}/{len(val)} {time.time() - t0:.0f}s', flush=True)
        manifest['seeds'][seed] = {'checkpoint': str(ck), 'checkpoint_sha256': sha256(ck),
                                   'consistency_test2_pixel_agreement': agree,
                                   'val_seconds': round(time.time() - t0, 1)}
        del model
        torch.cuda.empty_cache()
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('done', flush=True)


if __name__ == '__main__':
    main()
