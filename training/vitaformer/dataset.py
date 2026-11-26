"""
Organoid Segmentation Dataset
Handles VOC2012-format data with temporal structure.

Filename format: D{day}_{rep}-{condition}_Well{well}_Channel{ch}_Seq{seq}_RGB_C1
Labels: 0=background, 1=alive, 2=dead, 3=bubble
"""

import os
import re
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


CLASS_NAMES = ['background', 'alive', 'dead', 'bubble']
NUM_CLASSES = 4


def parse_filename(name):
    """Parse organoid image filename into metadata dict.

    Handles all naming variants in the dataset:
      Standard:  D{day}_{rep}-{condition}_Well{X}_Channel{ch}_Seq{n}_RGB_C1
      New1:      D{day}_{rep}-{condition}_Well{X}_ChannelNew1_Seq{n}_RGB_New1
      2X-BF:     D{day}_{id}_Well{X}_Channel2X-BF_Seq{n}_RGB_2X-BF
      No suffix: D{day}_{id}_Well{X}_Channel{ch}_Seq{n}_RGB
      Single id: D{day}_{id}_Well{X}_Channel{ch}_Seq{n}_RGB_C1  (id has no hyphen)

    well_id = "{identifier}-{Well}" so same physical well across days shares the same id.
    """
    parts = name.split('_')
    # Minimum: Day / identifier / Well / Channel... / Seq... / RGB
    if len(parts) < 6:
        return None
    if not re.match(r'D\d+$', parts[0]):
        return None

    # Find the Well field (first field matching WellXNN pattern)
    well_idx = None
    for i, p in enumerate(parts):
        if re.match(r'Well[A-Z]\d+$', p):
            well_idx = i
            break
    if well_idx is None or well_idx < 2:
        return None

    remaining = parts[well_idx + 1:]
    # Expect Channel / Seq / RGB [/ suffix]
    if (len(remaining) < 3 or not remaining[0].startswith('Channel')
            or not remaining[1].startswith('Seq') or remaining[2] != 'RGB'):
        return None

    day        = parts[0]
    identifier = '_'.join(parts[1:well_idx])   # e.g. "01R-FC", "CGC", "2348Y6"
    well       = parts[well_idx]               # e.g. "WellB4", "WellF03"
    channel    = remaining[0][len('Channel'):]
    seq        = remaining[1][len('Seq'):]

    # Split identifier into rep / condition when hyphen is present
    if '-' in identifier:
        rep, condition = identifier.split('-', 1)
    else:
        rep, condition = identifier, 'FC'

    return {
        'day':       day,
        'replicate': rep,
        'condition': condition,
        'well':      well,
        'channel':   channel,
        'seq':       seq,
        # well_id must be stable across days: same physical well → same id
        'well_id': f"{identifier}-{well}",
    }


def build_dataset_index(root):
    """Build index of all samples grouped by well_id."""
    img_dir = os.path.join(root, 'JPEGImages')
    mask_dir = os.path.join(root, 'SegmentationClass')

    samples = []
    well_groups = defaultdict(list)

    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith('.jpg'):
            continue
        stem = fname[:-4]
        mask_path = os.path.join(mask_dir, stem + '.png')
        if not os.path.exists(mask_path):
            continue

        meta = parse_filename(stem)
        if meta is None:
            continue

        entry = {
            'image': os.path.join(img_dir, fname),
            'mask': mask_path,
            'stem': stem,
            **meta,
        }
        samples.append(entry)
        well_groups[meta['well_id']].append(entry)

    return samples, well_groups


def train_val_split(well_groups, val_ratio=0.2, seed=42):
    """Split by well (not by image) to avoid data leakage."""
    rng = random.Random(seed)
    well_ids = sorted(well_groups.keys())
    rng.shuffle(well_ids)
    n_val = max(1, int(len(well_ids) * val_ratio))
    val_wells = set(well_ids[:n_val])
    train_wells = set(well_ids[n_val:])

    train_samples, val_samples = [], []
    for wid, entries in well_groups.items():
        if wid in val_wells:
            val_samples.extend(entries)
        else:
            train_samples.extend(entries)

    return train_samples, val_samples


def train_val_test_split(well_groups, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Three-way well-level split: train / val / test.

    test_ratio = 1 - train_ratio - val_ratio (≈0.2 by default).
    Returns (train_samples, val_samples, test_samples).
    """
    rng = random.Random(seed)
    well_ids = sorted(well_groups.keys())
    rng.shuffle(well_ids)
    n = len(well_ids)
    n_test = max(1, int(n * (1.0 - train_ratio - val_ratio)))
    n_val  = max(1, int(n * val_ratio))
    # Assign: first n_test → test, next n_val → val, rest → train
    test_wells  = set(well_ids[:n_test])
    val_wells   = set(well_ids[n_test:n_test + n_val])
    # train_wells = everything else (implicit)

    train_samples, val_samples, test_samples = [], [], []
    for wid, entries in well_groups.items():
        if wid in test_wells:
            test_samples.extend(entries)
        elif wid in val_wells:
            val_samples.extend(entries)
        else:
            train_samples.extend(entries)

    return train_samples, val_samples, test_samples


def stratified_train_val_test_split(well_groups, train_ratio=0.6, val_ratio=0.2,
                                     seed=42, use_condition=False):
    """Stratified well-level 6:2:2 split.

    use_condition=False (default): strata = max_dead_day
      'dead_D12', 'dead_D9', 'dead_D6', 'alive'

    use_condition=True: strata = max_dead_day × drug_condition
      e.g. 'D12_5-Fu', 'D9_CPT-11', 'alive_FC'
      Ensures val/test have the same drug-condition mix as train for Dead wells.
      For strata with n < 3 wells the extras fall into train to avoid empty splits.

    Returns (train_samples, val_samples, test_samples).
    """
    from PIL import Image as _PIL

    def _well_stratum(entries):
        dead_days = set()
        condition = entries[0].get('condition', 'FC')
        for e in entries:
            mask = np.array(_PIL.open(e['mask']))
            if (mask == 2).any():
                dead_days.add(e['day'])
        if 'D12' in dead_days:   dead_key = 'D12'
        elif 'D9' in dead_days:  dead_key = 'D9'
        elif 'D6' in dead_days:  dead_key = 'D6'
        else:                    dead_key = 'alive'
        if use_condition:
            return f"{dead_key}_{condition}"
        return f"dead_{dead_key}" if dead_key != 'alive' else 'alive'

    strata = defaultdict(list)
    for wid, entries in well_groups.items():
        strata[_well_stratum(entries)].append(wid)

    train_wells, val_wells, test_wells = set(), set(), set()
    for stratum_name in sorted(strata.keys()):
        wids = sorted(strata[stratum_name])
        import hashlib as _hl
        _h = int(_hl.md5(stratum_name.encode()).hexdigest(), 16) % 99991
        rng  = random.Random(seed + _h)
        rng.shuffle(wids)
        n = len(wids)
        if n == 1:
            # Can't split — put in train
            train_wells.update(wids)
            continue
        if n == 2:
            # One for val, one for train (skip test contribution)
            val_wells.add(wids[0])
            train_wells.add(wids[1])
            continue
        n_test  = max(1, round(n * (1.0 - train_ratio - val_ratio)))
        n_val   = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train, n_test = 1, max(0, n - 1 - n_val)
        test_wells.update(wids[:n_test])
        val_wells.update(wids[n_test:n_test + n_val])
        train_wells.update(wids[n_test + n_val:])

    train_s, val_s, test_s = [], [], []
    for wid, entries in well_groups.items():
        if wid in train_wells:
            train_s.extend(entries)
        elif wid in val_wells:
            val_s.extend(entries)
        else:
            test_s.extend(entries)

    return train_s, val_s, test_s


def print_split_stats(train_s, val_s, test_s, label="Split stats"):
    """Print Dead-class distribution across train/val/test for verification."""
    from PIL import Image as _PIL
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for split, name in [(train_s, 'TRAIN'), (val_s, 'VAL'), (test_s, 'TEST')]:
        dead_by_day = defaultdict(int)
        total_by_day = defaultdict(int)
        dead_px = 0
        total_px = 0
        for s in split:
            day = s['day']
            mask = np.array(_PIL.open(s['mask']))
            has_dead = (mask == 2).any()
            total_by_day[day] += 1
            dead_px += int((mask == 2).sum())
            total_px += mask.size
            if has_dead:
                dead_by_day[day] += 1
        n_dead_imgs = sum(dead_by_day.values())
        print(f"\n  {name} ({len(split)} images, {n_dead_imgs} with Dead):")
        for d in sorted(total_by_day):
            nd = dead_by_day.get(d, 0)
            n  = total_by_day[d]
            bar = f"  [{nd}/{n} Dead-imgs]" if nd > 0 else ""
            print(f"    {d}: {n:4d} imgs{bar}")
        pct = 100 * dead_px / max(total_px, 1)
        print(f"    Dead pixels: {dead_px:,} / {total_px:,}  ({pct:.4f}%)")
    print(f"{'='*55}\n")


class OrganoidDataset(Dataset):
    """Single-frame organoid segmentation dataset."""

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(self, samples, img_size=512, augment=False):
        self.samples = samples
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['image']).convert('RGB')
        mask = Image.open(s['mask'])  # palette/grayscale with values 0-3

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, self.IMG_MEAN, self.IMG_STD)

        mask_np = np.array(mask, dtype=np.int64)
        # Values 0-3 are valid classes; 253/254/255 are boundary/void → ignore (-1)
        ignore = (mask_np >= 250) | (mask_np > NUM_CLASSES - 1)
        mask_np[ignore] = -1
        mask_t = torch.from_numpy(mask_np)

        return img_t, mask_t, s['stem']

    def _augment(self, img, mask):
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
        # brightness/contrast only on image
        img = TF.adjust_brightness(img, 1.0 + random.uniform(-0.2, 0.2))
        img = TF.adjust_contrast(img, 1.0 + random.uniform(-0.2, 0.2))
        return img, mask


class TemporalOrganoidDataset(Dataset):
    """
    Temporal dataset: groups frames by well and returns all available
    time points together as (T, C, H, W).

    Days are sorted chronologically: D0 < D6 < D9 < D12.
    Missing days are filled with zeros and indicated by a validity mask.
    """

    DAY_ORDER = {'D0': 0, 'D6': 1, 'D9': 2, 'D12': 3}
    MAX_FRAMES = 4

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(self, well_groups, well_ids, img_size=512, augment=False):
        self.well_ids = well_ids
        self.well_groups = well_groups
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.well_ids)

    def __getitem__(self, idx):
        wid = self.well_ids[idx]
        entries = sorted(
            self.well_groups[wid],
            key=lambda e: self.DAY_ORDER.get(e['day'], 99)
        )

        imgs = torch.zeros(self.MAX_FRAMES, 3, self.img_size, self.img_size)
        masks = torch.full((self.MAX_FRAMES, self.img_size, self.img_size), -1, dtype=torch.long)
        valid = torch.zeros(self.MAX_FRAMES, dtype=torch.bool)
        days = [''] * self.MAX_FRAMES

        for e in entries:
            t = self.DAY_ORDER.get(e['day'], None)
            if t is None:
                continue
            img = Image.open(e['image']).convert('RGB')
            mask = Image.open(e['mask'])
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

            img_t = TF.to_tensor(img)
            img_t = TF.normalize(img_t, self.IMG_MEAN, self.IMG_STD)
            mask_np = np.array(mask, dtype=np.int64)
            ignore = (mask_np >= 250) | (mask_np > NUM_CLASSES - 1)
            mask_np[ignore] = -1

            imgs[t] = img_t
            masks[t] = torch.from_numpy(mask_np)
            valid[t] = True
            days[t] = e['day']

        return imgs, masks, valid, wid, days
