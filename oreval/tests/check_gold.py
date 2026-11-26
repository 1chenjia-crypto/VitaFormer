#!/usr/bin/env python3
"""Gold-standard check: does a software build reproduce the paper's frozen M2 masks?

Usage (run from anywhere; paths are relative to this file):
    python check_gold.py --device cpu  [--seeds 42,3407,2026] [--out results_cpu]
    python check_gold.py --device cuda [--seeds 42]            [--out results_gpu]
    python check_gold.py --readouts-only        # only (re)compute expected readouts from the paper masks

What it does
1. SHA256 of the three checkpoints in ../A_model_core/weights against Table S6 (frozen values below).
2. strict=True load of each checkpoint into the frozen M2 definition (no BiomedCLIP file needed: 'random' init + full checkpoint).
3. Inference with the shipped tool core (vitaformer_tool v1.1.0: io.load_image -> infer.predict_mask), which is a copy of the
   paper's infer_hard (224 window / 112 stride / reflect pad / ImageNet mean-std / softmax average over overlaps / argmax;
   fp16 autocast on CUDA, fp32 on CPU).
     CRC (5 test images): resize 'native'.         Reference = Stage F masks (stageF_m2_test/test_masks/M2_seed*).
     NViR (1 image):      resize 'square_box' 256. Reference = functional_validation_nvir_20261109/masks/m2_s*_S256.
4. Pixel agreement per image and seed -> <out>/agreement.csv. Pass criterion of the software plan: agreement >= 0.999.
5. Expected readouts computed from the PAPER masks (not from the new run) with
   - oreval.core.readouts_from_mask (foreground labels 1,2; min region 50 px; 8-connectivity)  -> expected_readouts_oreval.csv
   - vitaformer.readouts.image_readouts (the tool's own CSV columns)                           -> expected_readouts_tool.csv
   - per-class pixel counts                                                                    -> expected_class_pixels.csv
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / 'C_readouts_and_tool' / 'vitaformer_tool_v1.1.0'))
sys.path.insert(0, str(PKG / 'D_oreval' / 'organoid_readout_eval'))

FROZEN = {  # Table S6 / MODEL_FREEZE_M2_20261108.md
    42: '0f12be44859d06723ebfa35e8416df8cd4b02ff83f3ac153654d0f68df226101',
    3407: '1e4bd6afd0a1925094103c991430246b9f1fef09ce5b480e2c47056a7128f585',
    2026: 'c268464ee2ae3ee19e2112fc91ae86a95aa5d95086374df526fda9a7aa3225b8',
}
CASES = [  # (set, image path, reference-mask dir pattern, resize mode, resize value)
    *[('crc', p, 'crc_test_5/paper_masks_M2_seed{seed}', 'native', None)
      for p in sorted((HERE / 'crc_test_5' / 'images').glob('*.jpg'))],
    *[('nvir', p, 'nvir_1/paper_masks_M2_seed{seed}_S256', 'square_box', 256)
      for p in sorted((HERE / 'nvir_1' / 'images').glob('*.png'))],
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def expected_readouts():
    from oreval.core import readouts_from_mask
    from vitaformer.io import load_image
    from vitaformer.readouts import image_readouts
    ro, rt, rc = [], [], []
    for st, img, refpat, mode, val in CASES:
        rgb, _ = load_image(img)
        for seed in FROZEN:
            ref = HERE / refpat.format(seed=seed) / (img.stem + '.png')
            m = np.asarray(Image.open(ref))
            key = {'set': st, 'image': img.name, 'seed': seed, 'reference_mask': str(ref.relative_to(HERE))}
            ro.append({**key, **readouts_from_mask(m, (1, 2), 50)})
            rt.append({**key, **image_readouts(m, rgb, 50)})
            rc.append({**key, 'height': m.shape[0], 'width': m.shape[1],
                       **{f'px_{n}': int((m == i).sum()) for i, n in enumerate(['background', 'alive', 'dead', 'bubble'])}})
    pd.DataFrame(ro).to_csv(HERE / 'expected_readouts_oreval.csv', index=False)
    pd.DataFrame(rt).to_csv(HERE / 'expected_readouts_tool.csv', index=False)
    pd.DataFrame(rc).to_csv(HERE / 'expected_class_pixels.csv', index=False)
    print('expected readouts written (from paper masks)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu'); ap.add_argument('--seeds', default='42,3407,2026')
    ap.add_argument('--out', default=None); ap.add_argument('--readouts-only', action='store_true')
    a = ap.parse_args()
    expected_readouts()
    if a.readouts_only:
        return
    import torch
    from vitaformer.infer import load_model, predict_mask
    from vitaformer.io import load_image
    dev = torch.device(a.device)
    out = HERE / (a.out or f'results_{dev.type}'); out.mkdir(exist_ok=True)
    rows, log = [], []
    for seed in [int(s) for s in a.seeds.split(',')]:
        w = PKG / 'A_model_core' / 'weights' / f'vitaformer_m2_seed{seed}_final.pth'
        d = sha256(w); ok = d == FROZEN[seed]
        log.append(f'seed {seed}: sha256 {d} {"MATCHES Table S6" if ok else "MISMATCH"}')
        assert ok, log[-1]
        model, _, label = load_model(w, dev)  # strict=True inside
        log.append(f'seed {seed}: strict=True load OK ({label}), params {sum(p.numel() for p in model.parameters()):,}')
        for st, img, refpat, mode, val in CASES:
            rgb, _ = load_image(img); t0 = time.time()
            m = predict_mask(model, rgb, dev, mode, val)
            ref = np.asarray(Image.open(HERE / refpat.format(seed=seed) / (img.stem + '.png')))
            assert ref.shape == m.shape, (img, ref.shape, m.shape)
            diff = int((m != ref).sum())
            rows.append({'set': st, 'image': img.name, 'seed': seed, 'device': dev.type, 'resize': mode, 'resize_value': val,
                         'pixels': m.size, 'differing_pixels': diff, 'agreement': 1 - diff / m.size,
                         'pass_ge_0.999': (1 - diff / m.size) >= 0.999, 'seconds': round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
        del model
    df = pd.DataFrame(rows); df.to_csv(out / 'agreement.csv', index=False)
    import platform
    log += [f'torch {torch.__version__}, python {platform.python_version()}, device {dev}',
            f'min agreement {df.agreement.min():.7f}; all pass >=0.999: {bool(df["pass_ge_0.999"].all())}']
    (out / 'check_log.txt').write_text('\n'.join(log) + '\n'); print('\n'.join(log))


if __name__ == '__main__':
    main()
