#!/usr/bin/env python3
"""NViR N1 correction (2026-11-10, after verification; not a new hypothesis).

Finding: test domains A-test (Echo Rebel) and C (Cytation 5) are the SAME 245 physical wells (same location/study/plate/day/well,
identical ATP), imaged on two microscopes. The original analysis (nvir_analysis_20261109.py) clustered the bootstrap on
(location, microscope, study, plate, day), so the pooled CI treated the two images of one well as independent clusters.
Per-domain results are unaffected (no duplicates within a domain).

This script, reading the already-computed analysis/features_normalised.csv (no re-inference, no feature change):
  1. sample-count table per domain (images, unique wells, plate-day clusters, physical plates);
  2. pooled N1 with clusters = physical (location, study, plate, day) -> both images of a well are always resampled together;
  3. sensitivity: pooled N1 on unique wells only (A-test image kept, Cytation copy dropped; and the reverse);
  4. paired delta |rho| with the corrected clusters (pooled).
Same NBOOT (2000) and seed (20261109) as the plan. Output: analysis_n1_cluster_fix_20261110/ (new dir).
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

O = Path('3D_OrganoidSeg/functional_validation_nvir_20261109')
OUT = O / 'analysis_n1_cluster_fix_20261110'
NBOOT, BSEED = 2000, 20261109
PHYS = ['location', 'study', 'plate', 'day']
WELL = PHYS + ['well']
PRIMARY = ['m2_s42', 'm2_s3407', 'm2_s2026', 'segformer_s42', 'segformer_s3407', 'segformer_s2026',
           'nnunet_s42', 'nnunet_s3407', 'nnunet_s2026', 'otsu', 'reference']


def draws_for(ref, key):
    rng = np.random.default_rng(BSEED)
    gid = ref.groupby(key, sort=True).ngroup().to_numpy()
    groups = [np.where(gid == g)[0] for g in np.unique(gid)]
    return [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(NBOOT)]


def run(test, label, key):
    ref = test[test.rep == 'reference'].sort_values('image').reset_index(drop=True)
    order = ref.image.to_numpy()
    dr = draws_for(ref, key)
    res, boot = [], {}
    for r in PRIMARY:
        t = test[test.rep == r].set_index('image').reindex(order)
        x, y = t.fg_area_n.to_numpy(), t.atp_n.to_numpy()
        b = np.array([spearmanr(x[i], y[i]).statistic for i in dr]); boot[r] = b
        res.append({'analysis': label, 'rep': r, 'n_images': len(t), 'n_clusters': ref.groupby(key).ngroups,
                    'rho': spearmanr(x, y).statistic, 'ci_lo': np.nanpercentile(b, 2.5), 'ci_hi': np.nanpercentile(b, 97.5)})
    pairs = []
    for a in ['m2_s42', 'm2_s3407', 'm2_s2026']:
        for c in PRIMARY:
            if c.startswith('m2'):
                continue
            d = np.abs(boot[a]) - np.abs(boot[c])
            pairs.append({'analysis': label, 'a': a, 'b': c, 'delta_abs_rho': abs(res[PRIMARY.index(a)]['rho']) - abs(res[PRIMARY.index(c)]['rho']),
                          'ci_lo': np.nanpercentile(d, 2.5), 'ci_hi': np.nanpercentile(d, 97.5)})
    return res, pairs


def main():
    OUT.mkdir(exist_ok=False)
    f = pd.read_csv(O / 'analysis/features_normalised.csv', low_memory=False)
    test = f[(f.split == 'test') & f.rep.isin(PRIMARY)].copy()
    ref = test[test.rep == 'reference']
    counts = []
    for dom, g in ref.groupby('domain'):
        counts.append({'domain': dom, 'images': len(g), 'unique_wells': g[WELL].drop_duplicates().shape[0],
                       'plate_day_clusters': g[PHYS].drop_duplicates().shape[0],
                       'physical_plates': g[['location', 'study', 'plate']].drop_duplicates().shape[0],
                       'studies': g.study.nunique(), 'days': ','.join(map(str, sorted(g.day.unique())))})
    counts.append({'domain': 'pooled_test', 'images': len(ref), 'unique_wells': ref[WELL].drop_duplicates().shape[0],
                   'plate_day_clusters': ref[PHYS].drop_duplicates().shape[0],
                   'physical_plates': ref[['location', 'study', 'plate']].drop_duplicates().shape[0],
                   'studies': ref.study.nunique(), 'days': ''})
    pd.DataFrame(counts).to_csv(OUT / 'sample_counts.csv', index=False)
    R, P = [], []
    for label, sub in [('pooled_all_images_physical_clusters', test),
                       ('pooled_unique_wells_keep_echo_rebel', test[test.domain != 'A_cytation_v0']),
                       ('pooled_unique_wells_keep_cytation', test[test.domain != 'A_echo_rebel'])]:
        r, p = run(sub, label, PHYS); R += r; P += p
    pd.DataFrame(R).to_csv(OUT / 'N1_pooled_corrected.csv', index=False)
    pd.DataFrame(P).to_csv(OUT / 'N1_pooled_paired_corrected.csv', index=False)
    (OUT / 'run_manifest.json').write_text(json.dumps({'created': '2026-11-10', 'nboot': NBOOT, 'seed': BSEED,
                                                       'cluster_key': PHYS, 'reason': __doc__}, indent=2))
    print(pd.DataFrame(counts).to_string(index=False)); print(pd.DataFrame(R).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
