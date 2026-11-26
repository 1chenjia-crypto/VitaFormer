#!/usr/bin/env python3
"""Stage G object-level summary. Pooled object precision/recall/F1 per seed (sum over images), seed mean, and paired
hierarchical bootstrap (3 seed positions + val wells, 10,000, RNG 20261107) for the AMENDMENT contrasts.
Usage: stageG_objects_summary.py OUTDIR CSV [CSV ...]"""
import sys, json, contextlib, io
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import stageG_analysis as G  # noqa  (wells_of, classify)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
df = pd.concat([pd.read_csv(f) for f in sys.argv[2:]]); df['key'] = df.arm + '@' + df.delta.map(lambda d: f'{d:g}')
stems = sorted(df.stem.unique()); wid, nw = G.wells_of(stems); s2w = dict(zip(stems, wid))
df['w'] = df.stem.map(s2w); SEEDS = [42, 3407, 2026]
def agg(c):  # c: (..., 3) counts gt,pred,matched -> P,R,F1
    g, p, m = c[..., 0], c[..., 1], c[..., 2]
    P = 100 * m / np.maximum(p, 1); R = 100 * m / np.maximum(g, 1); return P, R, 2 * P * R / np.maximum(P + R, 1e-9)
rows, boots = [], {}
rng = np.random.default_rng(20261107); B = 10000
pos = rng.integers(0, 3, (B, 3)); wb = rng.integers(0, nw, (B, 3, nw))
W = np.stack([np.stack([np.bincount(wb[b, k], minlength=nw) for k in range(3)]) for b in range(B)])  # (B,3,nw)
for (key, mina, tau), g in df.groupby(['key', 'min_area', 'tau']):
    T = np.zeros((3, nw, 3))
    for i, s in enumerate(SEEDS):
        x = g[g.seed == s].groupby('w')[['n_gt', 'n_pred', 'n_matched']].sum()
        T[i, x.index.values] = x.values
    P, R, F = agg(T.sum(1)); rows.append({'arm@delta': key, 'min_area': mina, 'tau': tau, 'obj_P': P.mean(), 'obj_R': R.mean(),
        'obj_F1': F.mean(), 'F1_sd': F.std(ddof=1), 'n_gt_per_seed': T.sum(1)[:, 0].mean(), 'n_pred_per_seed': T.sum(1)[:, 1].mean()})
    bc = np.einsum('bkw,bkwc->bkc', W, T[pos])  # (B,3,3)
    boots[(key, mina, tau)] = [m.mean(1) for m in agg(bc)]
    boots[(key, mina, tau, 'pt')] = [m.mean() for m in (P, R, F)]
    boots[(key, mina, tau, 'seed')] = [m for m in (P, R, F)]
summ = pd.DataFrame(rows); summ.to_csv(out / 'object_summary.csv', index=False)
CON = [('M2@0', 'nnUNet@0'), ('Full@0', 'FullnoRD@0'), ('M2@0', 'M0@0'), ('M0@0', 'nnUNet@0'), ('FullnoRD@0', 'nnUNet@0'),
       ('M2@0', 'nnUNet@-2.9'), ('Full@0', 'FullnoRD@-0.1')]
crow = []
for a, b in CON:
    for mina in (0, 20):
        for tau in (0.5, 0.3, 0.1):
            if (a, mina, tau) not in boots or (b, mina, tau) not in boots: continue
            for j, nm in enumerate(['obj precision', 'obj recall', 'obj F1']):
                d = boots[(a, mina, tau)][j] - boots[(b, mina, tau)][j]
                pt = boots[(a, mina, tau, 'pt')][j] - boots[(b, mina, tau, 'pt')][j]
                sd = boots[(a, mina, tau, 'seed')][j] - boots[(b, mina, tau, 'seed')][j]
                lo95, hi95 = np.percentile(d, [2.5, 97.5]); lo90, hi90 = np.percentile(d, [5, 95])
                crow.append({'contrast': f'{a} - {b}', 'min_area': mina, 'tau': tau, 'metric': nm, 'delta': pt,
                             'ci95': f'[{lo95:.2f}, {hi95:.2f}]', 'seed_deltas': ';'.join(f'{v:+.2f}' for v in sd),
                             'reading': G.classify(lo95, hi95, lo90, hi90, 5.0)})
pd.DataFrame(crow).to_csv(out / 'object_contrasts.csv', index=False)
(out / 'manifest.json').write_text(json.dumps({'inputs': sys.argv[2:], 'bootstrap': '10000, RNG 20261107, seed positions + wells',
    'aggregation': 'counts pooled over images per seed; metric per seed then mean over 3 positions', 'margin': 5.0}, indent=2))
print(summ.round(2).to_string()); print(pd.DataFrame(crow).to_string())
