#!/usr/bin/env python3
"""Stage 36 (2026-11-09): frozen M2 (MODEL_FREEZE_M2_20261108.md, seeds 42/3407/2026 final.pth) on full DeepOrganoid.
Copy of run_final_model_deeporganoid_p1p99_full_20261029.py (Stage 31); ONLY the model changed (OrganoidSegMultiScale, M2 checkpoints).
Same images, per-image p1-p99 stretch, legacy.infer (224/112, reflect pad, ImageNet norm, fp16 autocast, softmax average), argmax,
legacy.feats. Plan: analysis_ctg_alignment/36_m2_deeporganoid_20261109/ANALYSIS_PLAN.md.
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
STAGE = BASE / '36_m2_deeporganoid_20261109'
MODELS = {
    'm2_seed42': {'checkpoint': ROOT / 'vco/runs/arch_exploration_20261104' / 'stageB_m2_seed42/final.pth',
                  'sha256': '0f12be44859d06723ebfa35e8416df8cd4b02ff83f3ac153654d0f68df226101', 'seed': 42},
    'm2_seed3407': {'checkpoint': ROOT / 'vco/runs/arch_exploration_20261104' / 'stageC_m2_seed3407/final.pth',
                    'sha256': '1e4bd6afd0a1925094103c991430246b9f1fef09ce5b480e2c47056a7128f585', 'seed': 3407},
    'm2_seed2026': {'checkpoint': ROOT / 'vco/runs/arch_exploration_20261104' / 'stageC_m2_seed2026/final.pth',
                    'sha256': 'c268464ee2ae3ee19e2112fc91ae86a95aa5d95086374df526fda9a7aa3225b8', 'seed': 2026},
}
COLORS = np.array([[0, 0, 0], [45, 180, 75], [220, 50, 47], [60, 130, 220]], dtype=np.uint8)

sys.path.insert(0, str(ROOT / 'deepOrganoid_v1'))
import extract_full_features as legacy  # noqa: E402  (load_new = p1-p99, infer, feats)
sys.path.insert(0, str(ROOT / 'vco/organoid_seg'))
from model_v3_multiscale import OrganoidSegMultiScale  # noqa: E402


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
    device = torch.device('cuda')
    model = OrganoidSegMultiScale(str(legacy.BCLIP), num_classes=4, freeze_blocks=0, zero_cnn_input=False)
    ck = torch.load(CKPT, map_location='cpu', weights_only=True)
    model.load_state_dict(ck.get('model_state', ck), strict=True)
    model.eval().to(device)

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
            prob = legacy.infer(model, img, device)
            pred = prob.argmax(0).astype(np.uint8)
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
        'model': args.model, 'architecture': {'arch': 'M2', 'use_mpm': False, 'use_bam': False, 'rare_dead': True, 'seed': spec['seed']},
        'preprocessing': 'per-image 1st-99th percentile stretch of 16-bit IXM_BestFocus to uint8 RGB',
        'window': legacy.WINDOW, 'stride': legacy.STRIDE, 'source_csv': str(legacy.TGT_CSV),
        'source_csv_sha256': sha256(legacy.TGT_CSV), 'n_images': len(records),
        'new_inferences': len(records) - reused, 'reused_masks': reused,
        'training_rerun': False, 'stage12_full_model_rerun': False,
        'torch_version': torch.__version__, 'elapsed_seconds': time.time() - t0,
    }, indent=2))


if __name__ == '__main__':
    main()
