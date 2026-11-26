#!/usr/bin/env python3
"""Stage 39 (2026-11-09): frozen SegFormer-B2 original (best_val_miou.pth, seeds 42/3407/2026) on full DeepOrganoid.
Copy of run_m2_deeporganoid_p1p99_20261109.py (Stage 36); ONLY the model + inference call changed.
Same images, per-image p1-p99 stretch (legacy.load_new), legacy.feats, border metrics.
Inference = SegFormer's own CRC test predict() (train_segformer_b2.py, imported unchanged): reflect pad to >=512, ImageNet norm,
512/256 sliding window, fp16 autocast, mean softmax, argmax. predict() returns only argmax, but feats() needs the softmax, so
predict_prob() below is a line-for-line copy that also returns the averaged softmax and a non-finite count; its argmax is
asserted equal to the unchanged B.predict() on the first 20 images and every 100th image.
Non-finite logits -> image re-run in fp32 (pre-specified). Plan: 39_segformer_b2_deeporganoid_20261109/ANALYSIS_PLAN.md.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage

ROOT = Path('3D_OrganoidSeg')
BASE = ROOT / 'external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment'
STAGE = BASE / '39_segformer_b2_deeporganoid_20261109'
MODELS = {
    'segformer_seed42': {'checkpoint': ROOT / 'baselines_fair_20261030/segformer_b2_runs/seed42/best_val_miou.pth',
                         'sha256': '4613f19815d41ff86374401f64e2c4bd023de085eef54c65ee426474dd77f130', 'seed': 42},
    'segformer_seed3407': {'checkpoint': ROOT / 'baselines_fair_20261030/segformer_b2_runs/seed3407/best_val_miou.pth',
                         'sha256': '45f18201e8825cc4f479f181011a0ddab9412e73dfed962d450e7176d46d2527', 'seed': 3407},
    'segformer_seed2026': {'checkpoint': ROOT / 'baselines_fair_20261030/segformer_b2_runs/seed2026/best_val_miou.pth',
                         'sha256': 'ce4f301bf67f688641028ee5c51897a21fb1cd1744e7d1050a9ef844171b1613', 'seed': 2026},
}
COLORS = np.array([[0, 0, 0], [45, 180, 75], [220, 50, 47], [60, 130, 220]], dtype=np.uint8)

sys.path.insert(0, str(ROOT / 'deepOrganoid_v1'))
import extract_full_features as legacy  # noqa: E402  (load_new = p1-p99, infer, feats)
sys.path.insert(0, str(ROOT / 'baselines_fair_20261030'))
import train_segformer_b2 as B  # noqa: E402  (unchanged; SHA256 checked below)
import torch.nn.functional as F  # noqa: E402
B_SHA256 = 'ffff4bfb11e254b8f2b12c44063c28daee6c728344c933bc97b17c09d0207f25'


@torch.inference_mode()
def predict_prob(model, img, device, fp32=False):
    """Copy of B.predict returning (argmax, mean softmax [NC,h,w], non-finite logit count)."""
    WIN, STRIDE, NC = B.WIN, B.STRIDE, B.NC
    h, w = img.shape[:2]
    ph, pw = max(WIN - h, 0), max(WIN - w, 0)
    x = B.norm(np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect'))
    H, W = x.shape[1:]
    ys = list(range(0, H - WIN + 1, STRIDE)); xs = list(range(0, W - WIN + 1, STRIDE))
    if ys[-1] + WIN < H: ys.append(H - WIN)
    if xs[-1] + WIN < W: xs.append(W - WIN)
    score = torch.zeros((NC, H, W), device=device); cnt = torch.zeros((H, W), device=device)
    coords = [(y, xx) for y in ys for xx in xs]
    nonfinite = 0
    for b in range(0, len(coords), 16):
        cc = coords[b:b + 16]
        batch = torch.stack([x[:, y:y + WIN, xx:xx + WIN] for y, xx in cc]).to(device)
        with torch.amp.autocast('cuda', enabled=not fp32):
            lg = model(pixel_values=batch).logits
        nonfinite += int((~torch.isfinite(lg)).sum())
        pr = F.softmax(F.interpolate(lg.float(), size=(WIN, WIN), mode='bilinear', align_corners=False), 1)
        for j, (y, xx) in enumerate(cc):
            score[:, y:y + WIN, xx:xx + WIN] += pr[j]; cnt[y:y + WIN, xx:xx + WIN] += 1
    prob = (score / cnt.clamp_min(1))[:, :h, :w]
    return prob.argmax(0).byte().cpu().numpy(), prob.float().cpu().numpy(), nonfinite


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def overlay(img, pred):
    out = img.copy()
    for cls in (1, 2, 3):
        m = pred == cls
        if m.any():
            out[m] = (0.4 * out[m] + 0.6 * COLORS[cls]).astype(np.uint8)
    return out


def border_metrics(fg):
    labels, n = ndimage.label(fg, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return 0, 0.0, 0.0
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    border = border[border != 0]
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    return int(len(border)), float(sizes[border].sum() / max(fg.sum(), 1)), float(len(border) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    args = ap.parse_args()
    spec = MODELS[args.model]
    CKPT, CKPT_SHA256 = spec['checkpoint'], spec['sha256']
    OUT = STAGE / args.model
    MASKS, OVERLAYS = OUT / 'hard_class_masks', OUT / 'overlays_jpg'
    for p in (MASKS, OVERLAYS):
        p.mkdir(parents=True, exist_ok=True)
    if sha256(CKPT) != CKPT_SHA256:
        raise RuntimeError('Frozen checkpoint hash mismatch')
    if sha256(B.__file__) != B_SHA256:
        raise RuntimeError('train_segformer_b2.py hash mismatch')
    device = torch.device('cuda')
    model, _ = B.build_model()
    ck = torch.load(CKPT, map_location='cpu', weights_only=True)
    model.load_state_dict(ck['model'], strict=True)
    model.eval().to(device)
    nonfinite_log, n_checked = [], 0

    rows = list(csv.DictReader(open(legacy.TGT_CSV)))
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    records, reused, t0 = [], 0, time.time()
    for i, r in enumerate(rows):
        sid = legacy.sid(r)
        mask_path = MASKS / f'{sid}_classes.png'
        img = legacy.load_new(r['IXM_BestFocus_resolved'])
        was_cached = mask_path.exists()
        if was_cached:
            pred = np.asarray(Image.open(mask_path)).astype(np.uint8)
            prob = np.eye(4, dtype=np.float32)[pred].transpose(2, 0, 1)  # soft features unavailable on reuse
            reused += 1
        else:
            pred, prob, nf = predict_prob(model, img, device)
            if nf:
                nonfinite_log.append({'sample_id': sid, 'nonfinite_fp16': nf})
                pred, prob, nf32 = predict_prob(model, img, device, fp32=True)
                nonfinite_log[-1]['nonfinite_fp32'] = nf32
            elif i < 20 or i % 100 == 0:
                if not np.array_equal(pred, B.predict(model, img, device)):
                    raise RuntimeError(f'{sid}: predict_prob argmax != B.predict')
                n_checked += 1
            pred = pred.astype(np.uint8)
            Image.fromarray(pred).save(mask_path)
            Image.fromarray(overlay(img, pred)).save(OVERLAYS / f'{sid}_overlay.jpg', quality=88)
        gray = np.asarray(Image.fromarray(img).convert('L')).astype(np.float32)
        fg = (pred == 1) | (pred == 2)
        bn, bpf, bcf = border_metrics(fg)
        records.append({
            'sample_id': sid, 'CCL': legacy.clean(r['CCL']), 'Row': legacy.clean(r['Row']),
            'Col': legacy.clean(r['Col']), 'Func': legacy.clean(r['Func']),
            'Cmpd1': legacy.clean(r['Cmpd1']), 'Conc1': legacy.fnum(r['Conc1']),
            'Lum': legacy.fnum(r['Lum']), 'Lum_FA': legacy.fnum(r['Lum_FA']),
            **legacy.feats(prob, gray),
            'border_touch_component_count': bn, 'border_touch_pixel_fraction': bpf,
            'border_touch_component_fraction': bcf, 'mask_reused': was_cached,
        })
        if (i + 1) % 100 == 0:
            print(f'shard {args.shard}: {i + 1}/{len(rows)} {time.time() - t0:.0f}s', flush=True)

    pd.DataFrame(records).to_csv(OUT / f'features_shard{args.shard}of{args.nshards}.csv', index=False)
    (OUT / f'manifest_shard{args.shard}of{args.nshards}.json').write_text(json.dumps({
        'created': '2026-11-09', 'script': __file__, 'script_sha256': sha256(__file__),
        'checkpoint': str(CKPT), 'checkpoint_sha256': CKPT_SHA256,
        'model': args.model, 'architecture': {'arch': 'SegFormer-B2 (HF transformers, mit-b2 init)', 'version': 'original fp16 run, best_val_miou', 'seed': spec['seed']},
        'inference': 'B.predict logic: reflect pad>=512, ImageNet norm, 512/256 sliding, fp16 autocast, mean softmax, argmax',
        'train_segformer_b2_sha256': B_SHA256, 'argmax_equal_checks_vs_B_predict': n_checked,
        'nonfinite_images': nonfinite_log, 'checkpoint_val_mIoU': float(ck['val_mIoU']), 'checkpoint_iter': int(ck['iter']),
        'preprocessing': 'per-image 1st-99th percentile stretch of 16-bit IXM_BestFocus to uint8 RGB',
        'window': B.WIN, 'stride': B.STRIDE, 'source_csv': str(legacy.TGT_CSV),
        'source_csv_sha256': sha256(legacy.TGT_CSV), 'n_images': len(records),
        'new_inferences': len(records) - reused, 'reused_masks': reused,
        'training_rerun': False, 'stage12_full_model_rerun': False,
        'torch_version': torch.__version__, 'elapsed_seconds': time.time() - t0,
    }, indent=2))


if __name__ == '__main__':
    main()
