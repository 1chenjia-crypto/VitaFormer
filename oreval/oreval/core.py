"""Model-agnostic evaluation of segmentation-derived functional readouts.

Inputs are plain tables, so any segmenter can be evaluated:
  - readouts: one row per image (image id + numeric readouts, e.g. foreground area), from `readouts_from_masks` or your own pipeline;
  - metadata: one row per image with experimental annotations (control flag, grouping keys, assay value, drug, concentration, line).

Endpoints (each reported with its tier):
  association        Spearman rho(readout, assay) with a cluster bootstrap CI           [control-relative, no fitting]
  dose_direction     sign agreement of rho(conc, readout) and rho(conc, assay)          [control-relative, no fitting]
  within_group_rank  per drug: Spearman over lines of concentration-averaged responses  [control-relative, no fitting]
  general_baseline   ranking from each line's mean assay response to the other drugs    [assay-only reference]
Calibrated predictions (fitting readouts to assay labels) are intentionally left to the user's own model; the tier is recorded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr

EIGHT = np.ones((3, 3), bool)


# ----------------------------------------------------------------------------- readouts from masks
def readouts_from_mask(mask: np.ndarray, foreground_labels=(1, 2), min_region_px: int = 50) -> dict:
    """Image-only readouts from an integer mask. `foreground_labels` lists the label values counted as foreground."""
    fg = np.isin(mask, foreground_labels)
    lab, _ = ndi.label(fg, structure=EIGHT)
    areas, circ = [], []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        obj = lab[sl] == i
        a = int(obj.sum())
        if a < min_region_px:
            continue
        per = max(float((obj & ~ndi.binary_erosion(obj)).sum()), 1.0)
        areas.append(a); circ.append(4 * math.pi * a / per ** 2)
    return {'foreground_area_px': int(fg.sum()), 'foreground_area_fraction': float(fg.mean()), 'n_regions': len(areas),
            'region_area_median_px': float(np.median(areas)) if areas else 0.0,
            'region_circularity_mean': float(np.mean(circ)) if circ else float('nan')}


# ----------------------------------------------------------------------------- normalisation
def control_normalise(d: pd.DataFrame, cols, group_cols, control_col='is_control', mode='ratio', zero='nan'):
    """Normalise `cols` to the median of control rows (control_col True) within group_cols.
    mode: 'ratio' (x / median) or 'difference' (x - median). zero: what to do when a ratio is undefined ('nan' or a number).
    Returns a copy with `<col>_rel` columns and a count of undefined values per column in d.attrs['undefined']."""
    d = d.copy(); undefined = {}
    for c in cols:
        med = d[d[control_col].astype(bool)].groupby(group_cols)[c].median().rename('_m')
        x = d.join(med, on=group_cols)['_m']
        if mode == 'ratio':
            with np.errstate(divide='ignore', invalid='ignore'):
                v = d[c] / x
        elif mode == 'difference':
            v = d[c] - x
        else:
            raise ValueError(mode)
        bad = ~np.isfinite(v)
        undefined[c] = int(bad.sum())
        d[f'{c}_rel'] = v.where(~bad, np.nan if zero == 'nan' else float(zero))
    d.attrs['undefined'] = undefined
    return d


# ----------------------------------------------------------------------------- bootstrap helpers
def cluster_draws(d: pd.DataFrame, cluster_cols, nboot=2000, seed=20261109):
    """Index arrays for a cluster bootstrap: clusters are resampled with replacement, all rows of a cluster kept together.
    Rows that share a physical unit (e.g. the same well imaged twice) must share a cluster."""
    rng = np.random.default_rng(seed)
    gid = d.groupby(cluster_cols, sort=True).ngroup().to_numpy()
    groups = [np.where(gid == g)[0] for g in np.unique(gid)]
    return [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(nboot)], len(groups)


def _ci(b):
    b = np.asarray(b, float)
    return float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


# ----------------------------------------------------------------------------- endpoints
def association(d, x, y, cluster_cols, nboot=2000, seed=20261109, strata_col=None):
    """Spearman rho(x, y) overall and per stratum, with cluster-bootstrap CIs. Rows must be sorted identically across runs
    when results are to be compared with paired draws (see `paired_association`)."""
    out = []
    parts = [('pooled', d)] + ([(s, g) for s, g in d.groupby(strata_col)] if strata_col else [])
    for name, g in parts:
        g = g.reset_index(drop=True)
        dr, ncl = cluster_draws(g, cluster_cols, nboot, seed)
        xv, yv = g[x].to_numpy(float), g[y].to_numpy(float)
        b = [spearmanr(xv[i], yv[i], nan_policy='omit').statistic for i in dr]
        lo, hi = _ci(b)
        out.append({'stratum': name, 'n_rows': len(g), 'n_clusters': ncl, 'rho': spearmanr(xv, yv, nan_policy='omit').statistic,
                    'ci_lo': lo, 'ci_hi': hi, 'tier': 'control-relative, no fitting'})
    return pd.DataFrame(out)


def paired_association(runs: dict, key, x, y, cluster_cols, nboot=2000, seed=20261109):
    """runs: {name: DataFrame}; rows are aligned on `key`. Returns rho per run and paired delta |rho| (a - b) on shared draws."""
    names = list(runs)
    base = runs[names[0]].sort_values(key).reset_index(drop=True)
    dr, _ = cluster_draws(base, cluster_cols, nboot, seed)
    vals, boots = {}, {}
    for n in names:
        g = runs[n].set_index(key).loc[base[key]].reset_index()
        xv, yv = g[x].to_numpy(float), g[y].to_numpy(float)
        vals[n] = spearmanr(xv, yv, nan_policy='omit').statistic
        boots[n] = np.array([spearmanr(xv[i], yv[i], nan_policy='omit').statistic for i in dr])
    rows = []
    for a in names:
        for b in names:
            if a < b:
                dl = np.abs(boots[a]) - np.abs(boots[b]); lo, hi = _ci(dl)
                rows.append({'a': a, 'b': b, 'delta_abs_rho': abs(vals[a]) - abs(vals[b]), 'ci_lo': lo, 'ci_hi': hi})
    return pd.Series(vals, name='rho'), pd.DataFrame(rows)


def dose_direction(d, series_cols, conc_col, x, y, min_conc=4, min_abs_rho=0.5, aggregate='median'):
    """For each series (e.g. line x drug x day) with >= min_conc concentrations and |rho(conc, y)| >= min_abs_rho:
    concordant if sign(rho(conc, x)) == sign(rho(conc, y)). aggregate='median': correlations on per-concentration medians;
    'none': on individual wells (as in the NViR analysis)."""
    rows = []
    d = d[d[conc_col].notna()]
    for k, g in d.groupby(series_cols):
        z = g.groupby(conc_col)[[x, y]].median().reset_index() if aggregate == 'median' else g[[conc_col, x, y]]
        if z[conc_col].nunique() < min_conc:
            continue
        ry = spearmanr(z[conc_col], z[y]).statistic; rx = spearmanr(z[conc_col], z[x]).statistic
        if not np.isfinite(ry) or abs(ry) < min_abs_rho:
            continue
        rows.append({**dict(zip(series_cols, k if isinstance(k, tuple) else (k,))), 'rho_conc_assay': ry, 'rho_conc_readout': rx,
                     'concordant': bool(np.isfinite(rx) and np.sign(rx) == np.sign(ry))})
    t = pd.DataFrame(rows)
    return t, {'n_series': len(t), 'concordant': int(t.concordant.sum()) if len(t) else 0}


def response_summary(d, unit_col, group_col, conc_col, x, y):
    """Concentration-averaged response per unit x group: mean over concentrations of per-concentration medians."""
    z = d.groupby([unit_col, group_col, conc_col])[[x, y]].median().reset_index()
    return z.groupby([unit_col, group_col], as_index=False)[[x, y]].mean()


def within_group_rank(d, unit_col, group_col, conc_col, x, y, min_units=5, nboot=2000, seed=20261030):
    """Per group (drug): Spearman over units (lines) of concentration-averaged readout vs assay; mean over groups,
    with a bootstrap over groups. Note: groups sharing a mechanism are not independent, so this CI can be too narrow."""
    s = response_summary(d, unit_col, group_col, conc_col, x, y)
    per = {g: spearmanr(h[x], h[y]).statistic for g, h in s.groupby(group_col) if h[unit_col].nunique() >= min_units}
    per = pd.Series(per)
    rng = np.random.default_rng(seed)
    b = [per.iloc[rng.integers(0, len(per), len(per))].mean() for _ in range(nboot)]
    lo, hi = _ci(b)
    return {'mean_rho': float(per.mean()), 'ci_lo': lo, 'ci_hi': hi, 'n_groups': len(per),
            'tier': 'control-relative, no fitting'}, per


def general_baseline(d, unit_col, group_col, conc_col, y, min_units=5):
    """Assay-only reference: rank units within each group by their mean assay summary over the OTHER groups."""
    s = response_summary(d.assign(_x=d[y]), unit_col, group_col, conc_col, '_x', y)
    tot = s.groupby(unit_col)[y].transform('sum'); n = s.groupby(unit_col)[y].transform('count')
    s['G'] = (tot - s[y]) / (n - 1)
    per = pd.Series({g: spearmanr(h['G'], h[y]).statistic for g, h in s.groupby(group_col) if h[unit_col].nunique() >= min_units})
    return {'mean_rho': float(per.mean()), 'n_groups': len(per), 'tier': 'assay-only reference'}, per


def seed_range(values):
    v = np.asarray(list(values), float)
    return {'min': float(v.min()), 'max': float(v.max()), 'observed_range': float(v.max() - v.min()), 'n_seeds': len(v)}
