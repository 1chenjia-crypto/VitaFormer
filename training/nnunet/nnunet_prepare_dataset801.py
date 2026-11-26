#!/usr/bin/env python3
"""Convert the frozen CRC split to nnU-Net v2 raw format (Dataset801_CRCOrganoid), following the official
natural-image example (Dataset120_RoadSegmentation): RGB image as one case_0000.png, channel_names R/G/B,
labels as single-channel uint8 PNG (0 bg, 1 alive, 2 dead, 3 bubble).
imagesTr = train_cond + val_cond (645); imagesTs = test_cond (167, never used for planning or training).
Also writes the custom splits_final.json (fold 0 = 481 train / 164 val, identical to VitaFormer).
"""

from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image

VOC = Path('OrganoidSegBaseline/VOC2012')
BASE = Path('3D_OrganoidSeg/baselines_fair_20261030/nnunet')
RAW = BASE / 'nnUNet_raw/Dataset801_CRCOrganoid'


def ids(split):
    return [s.strip() for s in (VOC / f'ImageSets/Segmentation/{split}_cond.txt').read_text().splitlines() if s.strip()]


def main():
    for d in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs_reference_only'):
        (RAW / d).mkdir(parents=True, exist_ok=True)
    tr, va, te = ids('train'), ids('val'), ids('test')
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    for s in tr + va + te:
        img = Image.open(VOC / 'JPEGImages' / f'{s}.jpg').convert('RGB')
        lab = np.asarray(Image.open(VOC / 'SegmentationClass' / f'{s}.png'))
        assert lab.max() <= 3
        sub_i, sub_l = ('imagesTs', 'labelsTs_reference_only') if s in te else ('imagesTr', 'labelsTr')
        if not (RAW / sub_i / f'{s}_0000.png').exists():
            img.save(RAW / sub_i / f'{s}_0000.png')
            Image.fromarray(lab.astype(np.uint8), mode='L').save(RAW / sub_l / f'{s}.png')
    (RAW / 'dataset.json').write_text(json.dumps({
        'channel_names': {'0': 'R', '1': 'G', '2': 'B'},
        'labels': {'background': 0, 'alive': 1, 'dead': 2, 'bubble': 3},
        'numTraining': len(tr) + len(va), 'file_ending': '.png',
        'name': 'CRCOrganoid', 'description': 'VitaFormer frozen condition-aware split; test in imagesTs only'}, indent=2))
    splits = [{'train': tr, 'val': va}]
    (BASE / 'splits_final_fold0.json').write_text(json.dumps(splits, indent=1))
    (BASE / 'prepare_manifest.json').write_text(json.dumps({
        'created': '2026-10-30', 'script': __file__,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'split_sha256': {k: hashlib.sha256((VOC / f'ImageSets/Segmentation/{k}_cond.txt').read_bytes()).hexdigest()
                         for k in ('train', 'val', 'test')},
        'note': 'JPEG decoded once and stored losslessly as PNG; labels copied unchanged'}, indent=2))
    print(len(tr), len(va), len(te))


if __name__ == '__main__':
    main()
