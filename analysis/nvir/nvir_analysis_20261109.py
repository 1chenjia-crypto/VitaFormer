#!/usr/bin/env python3
"""NViR main analysis (functional_validation_nvir_20261109/ANALYSIS_PLAN.md + Amendment 1). Runs only after
scale_selection_choice.json exists (scale fixed without ATP). Reads masks, computes features, endpoints N1/N1b/N2/N3.

Fixed implementation details (written before any ATP is read):
- Foreground = classes 1|2 (Alive|Dead); Bubble excluded. Otsu: global Otsu on the crop, foreground = darker side, holes
  filled, largest component kept. Reference = authors' masks cropped as crop_spheroid.py.
- Set A per image: fg_area (crop px), and on the largest component: eq_diam, circularity 4piA/P^2 (P = skimage perimeter),
  solidity, eccentricity; n_comp = components >= 50 px. Empty foreground -> area 0, shape features 0, n_comp 0.
  Set B: mean intensity in fg, in a ring of width max(1, round(0.1*eq_diam)) outside fg, ratio in/ring.
  Dead fraction (exploratory) = Dead px / fg px.
- Control normalisation: feature / median over medium wells of the same (location, microscope, study, plate, day);
  non-finite results (0/0, x/0) -> 1.0 (neutral).
- Label: value / median(medium value) in the same group (authors' normalize_by_compound). Test rows incl. controls.
- N1: Spearman(norm fg_area, norm ATP), pooled test + per domain; 2,000 cluster-bootstrap draws over plate groups
  (seed 20261109; draws shared by all representations -> paired deltas of |rho|).
- N1b: test, treatment == drug, per (study, compound, day) with >= 4 distinct concentrations and |rho(conc, ATP)| >= 0.5:
  share with sign(rho(conc, area)) == sign(rho(conc, ATP)).
- N2: sklearn GradientBoostingRegressor(random_state=0) and RidgeCV(alphas=logspace(-3,3,13), cv=5) on standardised
  features; fit on train_val rows, evaluated on test rows; R^2 / MAE pooled + per domain (authors' evaluate.py logic).
- N3: fg Dice vs reference per test image.
- Supplement: N1 at all 5 candidate scales for seed 42 of each family.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label as sklabel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nvir_select_scale_20261109 as SS

ROOT = SS.ROOT; DATA = SS.DATA; OUT = SS.OUT
RES = OUT / 'analysis'
GROUP = ['location', 'microscope', 'study', 'plate', 'day']
SETA = ['fg_area', 'eq_diam', 'circularity', 'solidity', 'eccentricity', 'n_comp']
SETB = SETA + ['int_in', 'int_ring', 'int_ratio']
SEEDS = (42, 3407, 2026); NBOOT, BSEED = 2000, 20261109


def feats(cls_or_fg, gray, is_fg=False):
    fg = cls_or_fg if is_fg else np.isin(cls_or_fg, (1, 2))
    dead = 0.0 if is_fg else float((cls_or_fg == 2).sum())
    out = {'fg_area': float(fg.sum()), 'dead_frac': dead / fg.sum() if fg.sum() else 0.0}
    lab = sklabel(fg, connectivity=2)
    props = regionprops(lab)
    out['n_comp'] = sum(p.area >= 50 for p in props)
    if not props:
        out.update(eq_diam=0.0, circularity=0.0, solidity=0.0, eccentricity=0.0, int_in=0.0, int_ring=0.0, int_ratio=0.0)
        return out
    p = max(props, key=lambda q: q.area)
    out.update(eq_diam=p.equivalent_diameter_area, circularity=4 * np.pi * p.area / max(p.perimeter, 1) ** 2,
               solidity=p.solidity, eccentricity=p.eccentricity)
    w = max(1, int(round(0.1 * p.equivalent_diameter_area)))
    ring = ndimage.binary_dilation(fg, iterations=w) & ~fg
    out['int_in'] = float(gray[fg].mean()); out['int_ring'] = float(gray[ring].mean()) if ring.any() else 0.0
    out['int_ratio'] = out['int_in'] / out['int_ring'] if out['int_ring'] else 0.0
    return out


def otsu_fg(gray):
    fg = gray < threshold_otsu(gray)
    fg = ndimage.binary_fill_holes(fg)
    lab, n = ndimage.label(fg)
    if n == 0:
        return fg
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == sizes.argmax()


def representations(choice):
    reps = {}
    for fam in ('m2', 'segformer', 'nnunet'):
        if fam in choice:
            for s in SEEDS:
                reps[f'{fam}_s{s}'] = ('mask', OUT / 'masks' / f'{fam}_s{s}_S{choice[fam]}')
            for S in SS.SCALES:  # supplement (seed 42, all scales)
                reps[f'{fam}_s42_S{S}'] = ('mask', OUT / 'masks' / f'{fam}_s42_S{S}')
    reps['otsu'] = ('otsu', None); reps['reference'] = ('ref', None)
    return reps


def build_features(df, reps):
    rows = []
    for r in df.itertuples():
        gray = np.asarray(Image.open(DATA / 'extracted/images_labeled_cropped_0.2' / r.location_in_bucket).convert('L'), np.float32)
        ref = SS.ref_crop(r)
        for name, (kind, d) in reps.items():
            if kind == 'mask':
                f = d / r.location_in_bucket
                if not f.exists():
                    continue
                c = np.asarray(Image.open(f)); fg = np.isin(c, (1, 2)); ft = feats(c, gray)
            elif kind == 'otsu':
                fg = otsu_fg(gray); ft = feats(fg, gray, is_fg=True)
            else:
                fg = ref; ft = feats(fg, gray, is_fg=True)
            s = fg.sum() + ref.sum()
            rows.append({'rep': name, 'image': r.location_in_bucket, **ft,
                         'dice_vs_ref': 2 * (fg & ref).sum() / s if s else np.nan})
    return pd.DataFrame(rows)


def ctrl_norm(d, cols):
    d = d.copy()
    for c in cols:
        med = d[d.compound == 'medium'].groupby(['rep'] + GROUP)[c].median().rename(f'{c}_ctrl')
        d = d.join(med, on=['rep'] + GROUP)
        with np.errstate(divide='ignore', invalid='ignore'):
            v = d[c] / d[f'{c}_ctrl']
        d[f'{c}_n'] = v.where(np.isfinite(v), 1.0)
        d = d.drop(columns=f'{c}_ctrl')
    return d


def cluster_draws(test):
    rng = np.random.default_rng(BSEED)
    gid = test.groupby(GROUP, sort=True).ngroup().to_numpy()
    groups = {k: np.where(gid == k)[0] for k in np.unique(gid)}
    out = {}
    for dom in ['pooled'] + sorted(test.domain.unique()):
        keys = [k for k, idx in groups.items() if dom == 'pooled' or (test.domain.to_numpy()[idx[0]] == dom)]
        out[dom] = [np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]) for _ in range(NBOOT)]
    return out


def n1(test, reps, draws, order):
    res, boot = [], {}
    for name in reps:
        t = test[test.rep == name].set_index('image').reindex(order)
        if t.fg_area.isna().any():
            continue
        x, y, dom = t.fg_area_n.to_numpy(), t.atp_n.to_numpy(), t.domain.to_numpy()
        for dname, dr in draws.items():
            sel = np.ones(len(t), bool) if dname == 'pooled' else dom == dname
            rho = spearmanr(x[sel], y[sel]).statistic
            b = np.array([spearmanr(x[i], y[i]).statistic for i in dr])
            boot[(name, dname)] = b
            res.append({'rep': name, 'domain': dname, 'n': int(sel.sum()), 'rho': rho,
                        'ci_lo': np.nanpercentile(b, 2.5), 'ci_hi': np.nanpercentile(b, 97.5)})
    return pd.DataFrame(res), boot


def paired(boot, res):
    rows = []
    obs = res.set_index(['rep', 'domain']).rho
    for a in [f'm2_s{s}' for s in SEEDS]:
        for b in ['otsu', 'reference'] + [f'{f}_s{s}' for f in ('nnunet', 'segformer') for s in SEEDS]:
            for dname in sorted({k[1] for k in boot}):
                if (a, dname) not in boot or (b, dname) not in boot:
                    continue
                d = np.abs(boot[(a, dname)]) - np.abs(boot[(b, dname)])
                rows.append({'a': a, 'b': b, 'domain': dname, 'delta_abs_rho': abs(obs[(a, dname)]) - abs(obs[(b, dname)]),
                             'ci_lo': np.nanpercentile(d, 2.5), 'ci_hi': np.nanpercentile(d, 97.5)})
    return pd.DataFrame(rows)


def n1b(test):
    rows = []
    d = test[test.treatment.eq('drug') & test.concentration_uM.notna()]
    for name, g in d.groupby('rep'):
        ok = tot = 0
        for _, h in g.groupby(['study', 'compound', 'day']):
            if h.concentration_uM.nunique() < 4:
                continue
            ra = spearmanr(h.concentration_uM, h.atp_n).statistic
            if not np.isfinite(ra) or abs(ra) < 0.5:
                continue
            rx = spearmanr(h.concentration_uM, h.fg_area_n).statistic
            tot += 1; ok += int(np.isfinite(rx) and np.sign(rx) == np.sign(ra))
        rows.append({'rep': name, 'n_series': tot, 'concordant': ok, 'share': ok / tot if tot else np.nan})
    return pd.DataFrame(rows)


def n2(all_rows, reps):
    rows = []
    for name in reps:
        d = all_rows[all_rows.rep == name]
        if d.fg_area.isna().any() or d.split.nunique() < 2 or len(d) != 2359:
            continue
        tr, te = d[d.split == 'train_val'], d[d.split == 'test']
        for fset, cols in (('A', SETA), ('AB', SETB)):
            X = [f'{c}_n' for c in cols]
            for mname, model in (('gbr', GradientBoostingRegressor(random_state=0)),
                                 ('ridge', make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)))):
                model.fit(tr[X].to_numpy(float), tr.atp_n.to_numpy(float))
                p = model.predict(te[X].to_numpy(float))
                y = te.atp_n.to_numpy(float)
                rows.append({'rep': name, 'features': fset, 'model': mname, 'domain': 'pooled',
                             'r2': r2_score(y, p), 'mae': mean_absolute_error(y, p), 'n': len(y)})
                for dom in sorted(te.domain.unique()):
                    s = te.domain.to_numpy() == dom
                    rows.append({'rep': name, 'features': fset, 'model': mname, 'domain': dom,
                                 'r2': r2_score(y[s], p[s]), 'mae': mean_absolute_error(y[s], p[s]), 'n': int(s.sum())})
    return pd.DataFrame(rows)


def main():
    choice = json.loads((OUT / 'scale_selection_choice.json').read_text())
    RES.mkdir(parents=True, exist_ok=False)
    meta = pd.read_csv(DATA / 'raw/data_labeled.csv')  # ATP read here, after the scale choice is fixed
    meta = meta.rename(columns={'location_in_bucket': 'image'})
    reps = representations(choice)
    F = build_features(meta.rename(columns={'image': 'location_in_bucket'}), reps)
    F.to_csv(RES / 'features_raw.csv', index=False)
    d = F.merge(meta, on='image', how='left', validate='many_to_one')
    lab = d[(d.rep == 'reference') & (d.compound == 'medium')].groupby(GROUP).value.median().rename('atp_ctrl')
    d = d.join(lab, on=GROUP); d['atp_n'] = d.value / d.atp_ctrl
    d = ctrl_norm(d, SETB + ['dead_frac'])
    d.to_csv(RES / 'features_normalised.csv', index=False)
    test = d[d.split == 'test'].copy()
    ref_test = test[test.rep == 'reference'].sort_values('image').reset_index(drop=True)
    draws = cluster_draws(ref_test)
    r1, boot = n1(test, reps, draws, ref_test.image.to_numpy()); r1.to_csv(RES / 'N1_area_vs_atp.csv', index=False)
    pr = paired(boot, r1); pr.to_csv(RES / 'N1_paired_deltas.csv', index=False)
    r1b = n1b(test); r1b.to_csv(RES / 'N1b_dose_concordance.csv', index=False)
    r2 = n2(d, reps); r2.to_csv(RES / 'N2_supervised.csv', index=False)
    r3 = test.groupby(['rep', 'domain']).dice_vs_ref.median().reset_index(); r3.to_csv(RES / 'N3_dice_vs_reference_test.csv', index=False)
    dead = test.groupby(['rep', 'domain']).dead_frac.median().reset_index(); dead.to_csv(RES / 'exploratory_dead_fraction.csv', index=False)
    (RES / 'run_manifest.json').write_text(json.dumps({'created': '2026-11-09', 'scale_choice': choice, 'nboot': NBOOT,
                                                       'boot_seed': BSEED, 'script': __file__}, indent=2))
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
    print(r1[r1.rep.str.match(r'^(m2|segformer|nnunet)_s\d+$|otsu|reference')].round(3).to_string(index=False))
    print(r1b.round(3).to_string(index=False))
    print(r2[(r2.features == 'A') & (r2.model == 'gbr')].round(3).to_string(index=False))
    print(r3.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
