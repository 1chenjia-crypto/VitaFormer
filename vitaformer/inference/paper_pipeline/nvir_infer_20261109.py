#!/usr/bin/env python3
"""NViR (functional_validation_nvir_20261109/ANALYSIS_PLAN.md + Amendment 1): frozen-model inference on the authors'
cropped liver-spheroid images (images_labeled_cropped_0.2, 8-bit grayscale, used as is, replicated to 3 channels).

Scale: the crop is resized to S x S (PIL BOX = area averaging; every candidate S is <= the native crop side) or kept
native; the hard class mask is resized back to the native crop size with nearest neighbour. Saved: uint8 class masks.
This script never reads the ATP `value` column (usecols below excludes it).

Families (frozen weights, inference functions as in their DeepOrganoid stages, unchanged):
  m2        : OrganoidSegMultiScale final.pth (MODEL_FREEZE_M2_20261108), legacy.infer (224/112, ImageNet norm, fp16, softmax avg)
  segformer : SegFormer-B2 original best_val_miou.pth, train_segformer_b2.predict (512/256, reflect pad, fp16)
  nnunet    : nnU-Net run-2 checkpoint_final (Stage 35 make_predictor/predict, no mirroring)  [run with env_totalseg2]
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('nnUNet_compile', 'f')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path('3D_OrganoidSeg')
DATA = ROOT / 'external_datasets/NViR_zenodo15374443'
CROPS = DATA / 'extracted/images_labeled_cropped_0.2'
CSV = DATA / 'raw/data_labeled.csv'
USECOLS = ['location_in_bucket', 'study', 'location', 'microscope', 'plate', 'day', 'split', 'domain']  # no 'value'
OUTROOT = ROOT / 'functional_validation_nvir_20261109'

CKPT = {
    'm2': {42: ROOT / 'vco/runs/arch_exploration_20261104/stageB_m2_seed42/final.pth',
           3407: ROOT / 'vco/runs/arch_exploration_20261104/stageC_m2_seed3407/final.pth',
           2026: ROOT / 'vco/runs/arch_exploration_20261104/stageC_m2_seed2026/final.pth'},
    'segformer': {k: ROOT / f'baselines_fair_20261030/segformer_b2_runs/seed{k}/best_val_miou.pth' for k in (42, 3407, 2026)},
}
SHA = {('m2', 42): '0f12be44859d06723ebfa35e8416df8cd4b02ff83f3ac153654d0f68df226101',
       ('m2', 3407): '1e4bd6afd0a1925094103c991430246b9f1fef09ce5b480e2c47056a7128f585',
       ('m2', 2026): 'c268464ee2ae3ee19e2112fc91ae86a95aa5d95086374df526fda9a7aa3225b8',
       ('segformer', 42): '4613f19815d41ff86374401f64e2c4bd023de085eef54c65ee426474dd77f130',
       ('segformer', 3407): '45f18201e8825cc4f479f181011a0ddab9412e73dfed962d450e7176d46d2527',
       ('segformer', 2026): 'ce4f301bf67f688641028ee5c51897a21fb1cd1744e7d1050a9ef844171b1613'}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def build(family, seed, device):
    if family == 'm2':
        sys.path.insert(0, str(ROOT / 'deepOrganoid_v1')); sys.path.insert(0, str(ROOT / 'vco/organoid_seg'))
        import extract_full_features as legacy
        from model_v3_multiscale import OrganoidSegMultiScale
        if sha256(CKPT['m2'][seed]) != SHA[('m2', seed)]:
            raise RuntimeError('M2 checkpoint hash mismatch')
        m = OrganoidSegMultiScale(str(legacy.BCLIP), num_classes=4, freeze_blocks=0, zero_cnn_input=False)
        ck = torch.load(CKPT['m2'][seed], map_location='cpu', weights_only=True)
        m.load_state_dict(ck.get('model_state', ck), strict=True); m.eval().to(device)
        return lambda img: legacy.infer(m, img, device).argmax(0).astype(np.uint8), str(CKPT['m2'][seed])
    if family == 'segformer':
        sys.path.insert(0, str(ROOT / 'baselines_fair_20261030'))
        import train_segformer_b2 as B
        if sha256(CKPT['segformer'][seed]) != SHA[('segformer', seed)]:
            raise RuntimeError('SegFormer checkpoint hash mismatch')
        m, _ = B.build_model()
        m.load_state_dict(torch.load(CKPT['segformer'][seed], map_location='cpu', weights_only=True)['model'], strict=True)
        m.eval().to(device)
        return lambda img: B.predict(m, img, device).astype(np.uint8), str(CKPT['segformer'][seed])
    if family == 'nnunet':
        sys.path.insert(0, str(ROOT / 'scripts'))
        import run_nnunet_deeporganoid_20261102 as nn
        p, ck = nn.make_predictor(seed)
        return lambda img: nn.predict(p, img)[0], str(ck)
    raise ValueError(family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=['m2', 'segformer', 'nnunet'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--scale', required=True, help='target side in px, or "native"')
    ap.add_argument('--rows', choices=['train_val', 'all'], required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--outroot', default=str(OUTROOT))
    a = ap.parse_args()
    df = pd.read_csv(CSV, usecols=USECOLS)
    if a.rows == 'train_val':
        df = df[df.split == 'train_val']
    if a.limit:
        df = df.head(a.limit)
    out = Path(a.outroot) / 'masks' / f'{a.family}_s{a.seed}_S{a.scale}'
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    run, ckpt = build(a.family, a.seed, device)
    t0, n_new = time.time(), 0
    for i, name in enumerate(df.location_in_bucket):
        dst = out / name
        if dst.exists():
            continue
        g = Image.open(CROPS / name).convert('L')
        W0, H0 = g.size
        gi = g if a.scale == 'native' else g.resize((int(a.scale), int(a.scale)), Image.BOX)
        img = np.repeat(np.asarray(gi, np.uint8)[:, :, None], 3, axis=2)
        pred = run(np.ascontiguousarray(img))
        if a.scale != 'native':
            pred = np.asarray(Image.fromarray(pred).resize((W0, H0), Image.NEAREST))
        Image.fromarray(pred.astype(np.uint8)).save(dst)
        n_new += 1
        if (i + 1) % 100 == 0:
            print(f'{a.family} s{a.seed} S{a.scale}: {i + 1}/{len(df)} {time.time() - t0:.0f}s', flush=True)
    (out.parent / f'{out.name}__manifest_{a.rows}.json').write_text(json.dumps({
        'created': '2026-11-09', 'script': __file__, 'script_sha256': sha256(__file__), 'family': a.family, 'seed': a.seed,
        'scale': a.scale, 'rows': a.rows, 'n_rows': len(df), 'new_masks': n_new, 'checkpoint': ckpt,
        'input': 'authors images_labeled_cropped_0.2, 8-bit L used as is, replicated to 3 channels',
        'resize': 'PIL BOX to SxS, mask back with NEAREST' if a.scale != 'native' else 'none',
        'atp_value_read': False, 'torch': torch.__version__, 'elapsed_seconds': time.time() - t0}, indent=2))


if __name__ == '__main__':
    main()
