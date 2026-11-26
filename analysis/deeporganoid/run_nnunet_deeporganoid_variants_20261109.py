#!/usr/bin/env python3
"""Stage 38A (POST HOC, 2026-11-09): nnU-Net input-configuration variants on DeepOrganoid. Copy of the Stage 35 script
(run_nnunet_deeporganoid_20261102.py); only --variant added. Plan: 38_posthoc_robustness_20261109/PLAN_A_nnunet_config_sensitivity.md.
  V1 reflect-tile the 682x682 uint8 image to >= patch (1024x1280), centred; predict; keep the central 682x682 prob.
  V2 upsample x1.27 (bilinear, PIL) before prediction; probabilities resized back to 682x682 (bilinear); argmax.
  V3 mirroring TTA on (nnU-Net default); everything else as V0.
Original docstring: Stage 35: frozen nnU-Net v2 (run 2, checkpoint_final, no TTA) on all 3,456 DeepOrganoid IXM_BestFocus wells.
Plan: 35_nnunet_external_comparison_20261102/ANALYSIS_PLAN.md (written before this inference).

Input identical to Stage 31: deepOrganoid_v1 `load_new()` (per-image p1-p99 stretch to uint8, 3 channels, native
scale). nnU-Net then applies its own frozen plans (per-image z-score, patch 1024x1280, Gaussian sliding window,
tile step 0.5, no mirroring). Features: the unchanged Stage 31 `feats(prob, gray)` + border metrics, from the softmax
in memory; only hard masks and overlays are written. No training; nothing on DeepOrganoid informs any setting.
Consistency gate (--check-crc): masks from this API on CRC test images must equal the official nnUNetv2_predict
masks in baselines_fair_20261030/nnunet/pred_run2_seed{N}_noTTA.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('nnUNet_compile', 'f')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k, '1')
NN = Path('3D_OrganoidSeg/baselines_fair_20261030/nnunet')
os.environ['nnUNet_raw'] = str(NN / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(NN / 'nnUNet_preprocessed')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path('3D_OrganoidSeg')
STAGE = ROOT / 'external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment/38_posthoc_robustness_20261109'
PATCH = (1024, 1280)
SCALE_V2 = 1.27
sys.path.insert(0, str(ROOT / 'deepOrganoid_v1'))
sys.path.insert(0, str(ROOT / 'scripts'))
import extract_full_features as legacy  # noqa: E402
import run_final_model_deeporganoid_p1p99_full_20261029 as s31run  # noqa: E402  (overlay, border_metrics)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def make_predictor(seed, mirroring=False):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    folder = NN / f'nnUNet_results_run2_seed{seed}/Dataset801_CRCOrganoid/nnUNetTrainer__nnUNetPlans__2d'
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=mirroring, perform_everything_on_device=True,
                        device=torch.device('cuda'), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(folder), use_folds=(0,), checkpoint_name='checkpoint_final.pth')
    return p, folder / 'fold_0/checkpoint_final.pth'


def predict(p, img_u8):
    x = img_u8.transpose(2, 0, 1)[:, None].astype(np.float32)  # (3, 1, H, W), as NaturalImage2DIO
    seg, prob = p.predict_single_npy_array(x, {'spacing': (999, 1, 1)}, None, None, True)
    return np.asarray(seg[0], np.uint8), np.asarray(prob[:, 0], np.float32)


def predict_variant(p, img_u8, variant):
    h, w = img_u8.shape[:2]
    if variant in ('V0', 'V3'):
        return predict(p, img_u8)
    if variant == 'V1':
        ph, pw = max(PATCH[0] - h, 0), max(PATCH[1] - w, 0)
        t, l = ph // 2, pw // 2
        big = np.pad(img_u8, ((t, ph - t), (l, pw - l), (0, 0)), mode='reflect')
        _, prob = predict(p, big)
        prob = prob[:, t:t + h, l:l + w]
        return prob.argmax(0).astype(np.uint8), np.ascontiguousarray(prob)
    if variant == 'V2':
        H, W = round(h * SCALE_V2), round(w * SCALE_V2)
        big = np.asarray(Image.fromarray(img_u8).resize((W, H), Image.BILINEAR))
        _, prob = predict(p, big)
        t = torch.nn.functional.interpolate(torch.from_numpy(prob)[None], size=(h, w), mode='bilinear', align_corners=False)[0]
        prob = t.numpy().astype(np.float32)
        return prob.argmax(0).astype(np.uint8), prob
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--variant', choices=['V1', 'V2', 'V3'], required=True)
    ap.add_argument('--check-crc', type=int, default=0, help='only run the CRC consistency gate on N test images')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: first N wells into a scratch --out')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    p, ckpt = make_predictor(args.seed, mirroring=(args.variant == 'V3'))
    if args.check_crc:
        voc = Path('OrganoidSegBaseline/VOC2012')
        stems = (voc / 'ImageSets/Segmentation/test_cond.txt').read_text().split()[:args.check_crc]
        for s in stems:
            img = np.asarray(Image.open(voc / 'JPEGImages' / f'{s}.jpg').convert('RGB'))
            seg, _ = predict(p, img)
            ref = np.asarray(Image.open(NN / f'pred_run2_seed{args.seed}_noTTA/{s}.png'))
            print(f'CRC gate {s}: pixel agreement {float((seg == ref).mean()):.6f}', flush=True)
        return
    out = Path(args.out) if args.out else STAGE / f'A_nnunet_{args.variant}' / f'nnunet_seed{args.seed}'
    masks, overlays = out / 'hard_class_masks', out / 'overlays_jpg'
    masks.mkdir(parents=True, exist_ok=False); overlays.mkdir()
    rows = list(csv.DictReader(open(legacy.TGT_CSV)))
    if args.limit:
        rows = rows[:args.limit]
    records, t0 = [], time.time()
    for i, r in enumerate(rows):
        sid = legacy.sid(r)
        img = legacy.load_new(r['IXM_BestFocus_resolved'])
        pred, prob = predict_variant(p, img, args.variant)
        if not np.array_equal(pred, prob.argmax(0)):
            raise RuntimeError(f'{sid}: argmax(prob) != nnU-Net segmentation')
        Image.fromarray(pred).save(masks / f'{sid}_classes.png')
        Image.fromarray(s31run.overlay(img, pred)).save(overlays / f'{sid}_overlay.jpg', quality=88)
        gray = np.asarray(Image.fromarray(img).convert('L')).astype(np.float32)
        fg = (pred == 1) | (pred == 2)
        bn, bpf, bcf = s31run.border_metrics(fg)
        records.append({
            'sample_id': sid, 'CCL': legacy.clean(r['CCL']), 'Row': legacy.clean(r['Row']), 'Col': legacy.clean(r['Col']),
            'Func': legacy.clean(r['Func']), 'Cmpd1': legacy.clean(r['Cmpd1']), 'Conc1': legacy.fnum(r['Conc1']),
            'Lum': legacy.fnum(r['Lum']), 'Lum_FA': legacy.fnum(r['Lum_FA']), **legacy.feats(prob, gray),
            'border_touch_component_count': bn, 'border_touch_pixel_fraction': bpf,
            'border_touch_component_fraction': bcf, 'mask_reused': False})
        if (i + 1) % 100 == 0:
            print(f'seed {args.seed}: {i + 1}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(records).to_csv(out / 'features.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps({
        'created': '2026-11-09', 'variant': args.variant, 'posthoc': True, 'script': __file__, 'script_sha256': sha256(__file__),
        'checkpoint': str(ckpt), 'checkpoint_sha256': sha256(ckpt), 'seed': args.seed, 'tta': args.variant == 'V3',
        'tile_step_size': 0.5, 'preprocessing': 'load_new(): per-image p1-p99 stretch of 16-bit IXM_BestFocus to uint8 x3; '
        'then nnU-Net plans (per-image z-score)', 'source_csv': str(legacy.TGT_CSV),
        'source_csv_sha256': sha256(legacy.TGT_CSV), 'n_images': len(records), 'training_rerun': False,
        'torch_version': torch.__version__, 'elapsed_seconds': time.time() - t0}, indent=2))
    print('DONE', len(records), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
