#!/usr/bin/env python
"""Export label masks (PNG, values 0-3) for the 167 CRC test images with mmseg inference_segmentor
(config test pipeline: whole image, resize 1446 keep-ratio, logits resized to original). Run in venv-orgvit-py36."""
import argparse, os
import numpy as np
from PIL import Image
from mmseg.apis import init_segmentor, inference_segmentor

VOC = 'OrganoidSegBaseline/VOC2012'
ap = argparse.ArgumentParser(); ap.add_argument('--cfg'); ap.add_argument('--ckpt'); ap.add_argument('--out')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
stems = [s.strip() for s in open(VOC + '/ImageSets/Segmentation/test_cond.txt') if s.strip()]
model = init_segmentor(a.cfg, a.ckpt, device='cuda:0')
for i, s in enumerate(stems):
    seg = inference_segmentor(model, '%s/JPEGImages/%s.jpg' % (VOC, s))[0].astype(np.uint8)
    assert seg.max() < 4
    Image.fromarray(seg).save('%s/%s.png' % (a.out, s))
    if i % 20 == 0:
        print(i, s, flush=True)
print('exported', len(stems))
