#!/usr/bin/env python3
"""Compare a fresh run with the shipped expected output.
Usage: python examples/check_example.py <new_output_dir> [<expected_dir>]
Pass criteria: same images; mask pixel agreement >= 99.9% per image (CPU vs CPU should be 100%); readout differences reported."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

new = Path(sys.argv[1]); exp = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / 'expected_output'
a, b = pd.read_csv(new / 'readouts.csv'), pd.read_csv(exp / 'readouts.csv')
assert list(a.image) == list(b.image), 'different image lists'
ok = True
for img in a.image:
    stem = Path(img).stem
    m1 = np.asarray(Image.open(new / 'masks' / f'{stem}.png')); m2 = np.asarray(Image.open(exp / 'masks' / f'{stem}.png'))
    agree = float((m1 == m2).mean())
    ok &= agree >= 0.999
    print(f'{img}: mask agreement {agree:.6f}')
num = [c for c in a.columns if a[c].dtype.kind in 'fi']
diff = (a[num] - b[num]).abs().max()
print('max |difference| per numeric readout:'); print(diff[diff > 0].to_string() if (diff > 0).any() else '  none (identical)')
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
