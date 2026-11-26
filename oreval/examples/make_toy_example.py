#!/usr/bin/env python3
"""Synthetic toy example (no third-party data): 2 'models' x 4 plates x 3 lines x 2 drugs x 5 concentrations.
Model A draws organoid discs whose area follows the simulated viability; model B adds random edge false positives.
Writes examples/toy/{masks_modelA,masks_modelB}/*.png and examples/toy/metadata.csv. Deterministic (seed 7)."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

rng = np.random.default_rng(7)
T = Path(__file__).parent / 'toy'; T.mkdir(exist_ok=False)
for m in ('modelA', 'modelB'):
    (T / f'masks_{m}').mkdir()
rows = []
sens = {'L1': 1.0, 'L2': 0.6, 'L3': 0.3}  # lower = more sensitive
for plate in range(4):
    for line, s in sens.items():
        for drug in ('D1', 'D2'):
            for i, conc in enumerate([0, 0.01, 0.1, 1, 10]):
                via = 1.0 if conc == 0 else 1 / (1 + (conc / (s * (1.5 if drug == 'D2' else 1))) ** 1.2)
                atp = max(via * 1000 * rng.normal(1, 0.05), 1)
                img = f'p{plate}_{line}_{drug}_c{i}.png'
                for m in ('modelA', 'modelB'):
                    im = Image.new('L', (256, 256), 0); dr = ImageDraw.Draw(im)
                    for _ in range(12):
                        r = max(2, int(10 * np.sqrt(via) * rng.normal(1, 0.1))); x, y = rng.integers(30, 226, 2)
                        dr.ellipse([x - r, y - r, x + r, y + r], fill=1)
                    if m == 'modelB':
                        w = int(rng.integers(0, 60)); dr.rectangle([0, 0, w, 255], fill=1)  # edge false positive
                    im.save(T / f'masks_{m}' / img)
                rows.append({'image': img, 'plate': plate, 'line': line, 'drug': drug, 'conc': conc, 'ATP': atp,
                             'role': 'control' if conc == 0 else 'treated'})
pd.DataFrame(rows).to_csv(T / 'metadata.csv', index=False); print(len(rows), 'images')
