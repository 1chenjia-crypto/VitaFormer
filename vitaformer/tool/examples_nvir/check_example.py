#!/usr/bin/env python3
"""Check a run of the NViR example against the shipped expected output (made on CPU, fp32).
Usage: python examples_nvir/check_example.py "<run folder>"
  (the run folder is the one created by VitaFormer, e.g. VitaFormer_results/20261110-..._run)
CPU runs should be identical. GPU runs use fp16 and may differ in a few boundary pixels: the check then reports the agreement
(the developers measured >= 0.99972 per image between GPU fp16 and CPU fp32 masks on these images) and passes at >= 0.999."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

run = Path(sys.argv[1]); exp = Path(__file__).parent / 'expected_output'
a = pd.read_csv(run / 'readouts_image_measurements.csv').set_index('image')
b = pd.read_csv(exp / 'readouts_image_measurements.csv').set_index('image')
if set(a.index) != set(b.index):
    sys.exit(f'FAIL: different images. Missing: {sorted(set(b.index) - set(a.index))}')
worst, identical = 1.0, True
for img in b.index:
    m1 = np.asarray(Image.open(run / a.loc[img, 'mask_file'])); m2 = np.asarray(Image.open(exp / b.loc[img, 'mask_file']))
    agree = float((m1 == m2).mean()); worst = min(worst, agree); identical &= agree == 1.0
    print(f'{img}: mask agreement {agree:.6f}')
if a.resize_mode.iloc[0] != 'square_box' or float(a.resize_value.iloc[0]) != 256:
    print('NOTE: the expected output used "square side 256"; your run used', a.resize_mode.iloc[0], a.resize_value.iloc[0])
num = ['foreground_area_px', 'alive_area_px', 'n_regions', 'region_area_median_px']
rel = ((a.loc[b.index, num] - b[num]).abs() / b[num].replace(0, np.nan)).max()
print('max relative difference in readouts:', rel.round(6).to_dict())
if identical:
    print('PASS (identical)')
elif worst >= 0.999:
    print(f'PASS (not identical: lowest per-image agreement {worst:.6f}; expected on GPU because of fp16)')
else:
    print(f'FAIL (lowest per-image agreement {worst:.6f})'); sys.exit(1)
