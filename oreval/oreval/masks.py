"""python -m oreval.masks --masks DIR --foreground-labels 1,2 --output readouts.csv [--min-region-px 50]
Turns a folder of integer label masks (PNG/TIF; any segmenter) into the per-image readout table used by oreval.run.
The `image` column is the mask file name with its extension replaced by --image-ext (default: keep the mask name)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .core import readouts_from_mask


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--foreground-labels', default='1,2', help='comma-separated label values counted as foreground')
    ap.add_argument('--min-region-px', type=int, default=50); ap.add_argument('--image-ext', default=None)
    a = ap.parse_args(argv)
    labels = tuple(int(v) for v in a.foreground_labels.split(','))
    rows = []
    for p in sorted(Path(a.masks).iterdir()):
        if p.suffix.lower() not in {'.png', '.tif', '.tiff'}:
            continue
        name = p.name if a.image_ext is None else p.stem + a.image_ext
        rows.append({'image': name, **readouts_from_mask(np.asarray(Image.open(p)), labels, a.min_region_px)})
    pd.DataFrame(rows).to_csv(a.output, index=False); print(f'{len(rows)} masks -> {a.output}')


if __name__ == '__main__':
    main()
