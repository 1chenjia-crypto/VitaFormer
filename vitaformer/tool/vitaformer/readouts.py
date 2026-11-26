"""Per-image readouts from a hard VitaFormer mask (tier 1: image measurements), and control normalisation (tier 2).

Definitions (unchanged from v1.0.0)
- foreground = Alive + Dead (Bubble and background excluded). All areas are in pixels (no physical calibration is applied).
- n_regions = number of 8-connected foreground regions with >= min_region_px pixels. This is a REGION count, not an organoid count:
  touching organoids merge, and fragments split.
- dead_area_px_unvalidated = pixels predicted as the Dead class. The Dead class has not been externally validated (it did not activate
  on GemcitabineScreen or NViR); it must not be reported as a percentage of dead cells.
- circularity = 4*pi*area / perimeter^2 with perimeter = boundary-pixel count; equivalent diameter = sqrt(4*area/pi).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), bool)
TIER1_COLUMNS = ['foreground_area_px', 'foreground_area_fraction', 'alive_area_px', 'dead_area_px_unvalidated', 'bubble_area_px',
                 'n_regions', 'region_area_median_px', 'region_area_max_px', 'region_eq_diameter_median_px',
                 'region_circularity_mean', 'foreground_mean_intensity']
NORMALISABLE = ['foreground_area_px', 'alive_area_px', 'n_regions', 'region_area_median_px', 'region_eq_diameter_median_px']


def image_readouts(mask: np.ndarray, rgb: np.ndarray, min_region_px: int = 50) -> dict:
    fg = (mask == 1) | (mask == 2)
    gray = rgb.astype(np.float32).mean(axis=2)
    lab, n = ndi.label(fg, structure=EIGHT)
    areas, circ, eqd = [], [], []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        obj = lab[sl] == i
        a = int(obj.sum())
        if a < min_region_px:
            continue
        per = max(float((obj & ~ndi.binary_erosion(obj)).sum()), 1.0)
        areas.append(a); circ.append(4 * math.pi * a / per ** 2); eqd.append(math.sqrt(4 * a / math.pi))
    h, w = mask.shape
    return {
        'height_px': h, 'width_px': w,
        'foreground_area_px': int(fg.sum()), 'foreground_area_fraction': float(fg.mean()),
        'alive_area_px': int((mask == 1).sum()),
        'dead_area_px_unvalidated': int((mask == 2).sum()),
        'bubble_area_px': int((mask == 3).sum()),
        'n_regions': len(areas), 'min_region_px': min_region_px,
        'region_area_median_px': float(np.median(areas)) if areas else 0.0,
        'region_area_max_px': int(max(areas)) if areas else 0,
        'region_eq_diameter_median_px': float(np.median(eqd)) if eqd else 0.0,
        'region_circularity_mean': float(np.mean(circ)) if circ else float('nan'),
        'foreground_mean_intensity': float(gray[fg].mean()) if fg.any() else float('nan'),
    }


def control_summary(readouts: pd.DataFrame, plate_map: pd.DataFrame, feature: str = 'foreground_area_px') -> pd.DataFrame:
    """Per group: number of images, number of controls, control median and a status; shown to the user before normalising."""
    d = readouts[['image', feature]].merge(plate_map, on='image', how='inner')
    rows = []
    for g, x in d.groupby('group', dropna=False):
        c = x[x.role == 'control'][feature].dropna()
        med = float(c.median()) if len(c) else float('nan')
        status = 'ok' if len(c) and med > 0 else ('NO CONTROLS in this group' if not len(c) else 'CONTROL MEDIAN IS ZERO')
        rows.append({'group': g, 'n_images': len(x), 'n_controls': len(c), f'control_median_{feature}': med,
                     'controls': '; '.join(x[x.role == 'control'].image), 'status': status})
    return pd.DataFrame(rows)


def control_normalise(readouts: pd.DataFrame, plate_map: pd.DataFrame, features=('foreground_area_px',)) -> pd.DataFrame:
    """Tier 2: each feature divided by the median of the control images of the same group.
    plate_map columns: image, group, role ('control' or 'sample'). Rows whose group has no control, a zero/undefined control median, or an
    undefined readout get NaN and an explicit `<feature>_rel_status`; nothing is replaced by a default value."""
    pm = plate_map.copy()
    bad_role = ~pm.role.isin(['control', 'sample'])
    if bad_role.any():
        raise ValueError(f'role must be "control" or "sample"; found {sorted(pm.role[bad_role].astype(str).unique())}')
    if pm.image.duplicated().any():
        raise ValueError('an image appears more than once in the plate map')
    d = readouts.merge(pm[['image', 'group', 'role']], on='image', how='left')
    d['in_plate_map'] = d.group.notna()
    for f in features:
        ctrl = d[(d.role == 'control') & d[f].notna()].groupby('group')[f].agg(['median', 'size'])
        med = d.group.map(ctrl['median']); n = d.group.map(ctrl['size']).fillna(0).astype(int)
        val = d[f] / med
        status = np.where(~d.in_plate_map, 'not in plate map',
                 np.where(n == 0, 'no control in group',
                 np.where(med == 0, 'control median is zero',
                 np.where(d[f].isna(), 'readout undefined', 'ok'))))
        d[f + '_control_median'] = med
        d[f + '_n_controls'] = n
        d[f + '_rel_control'] = np.where(status == 'ok', val, np.nan)
        d[f + '_rel_status'] = status
    return d
