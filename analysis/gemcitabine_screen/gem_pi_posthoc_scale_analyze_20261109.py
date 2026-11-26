#!/usr/bin/env python3
"""POST HOC (Amendment 1): scale sensitivity analysis for GemcitabineScreen PI. Descriptive only, no gate.
Factors 1.0/0.75/0.5 from posthoc_scale/, factor 0.25 recomputed from the pre-registered runs for side-by-side.
Same estimators as gem_pi_analyze_20261109.py (imported). Output: posthoc_scale/analysis (new).
"""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import gem_pi_analyze_20261109 as A  # noqa: E402

D = A.D; P = D / 'posthoc_scale'; OUT = P / 'analysis'
FACTORS = [1.0, 0.75, 0.5, 0.25]


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    T = json.loads((D / 'analysis/log.json').read_text())['pi_threshold_T']
    pi = pd.read_csv(D / 'analysis/pi_per_frame.csv')
    stacks = {A.xy_of(p.name): A.frames(p) for p in sorted((A.DATA / 'PI').glob('*.tif'))}
    OUT.mkdir(); rows = []
    for f in FACTORS:
        for m in A.MODELS:
            base = D / m if f == 0.25 else P / f'f{f}' / m
            pf = pd.read_csv(base / 'per_frame.csv'); pf = pf[pf.frame.isin([0, 13, 18])]
            pf = pf.merge(pi[['xy', 'frame', 'pi_img']], on=['xy', 'frame'], validate='1:1')
            pf['DF'] = np.where(pf.fg_px > 0, pf.dead_px / pf.fg_px.where(pf.fg_px > 0), np.nan)
            b0 = pf[pf.frame == 0].set_index('xy')
            pf['fg_fc'] = pf.fg_px / pf.xy.map(b0.fg_px).replace(0, np.nan)
            for th in (52, 72):
                w = pf[pf.time_h == th].dropna(subset=['DF', 'fg_fc']).reset_index(drop=True)
                r = dict(factor=f, model=m, time_h=th, n=len(w), frames_with_dead=int((pf.dead_px > 0).sum()),
                         frames_total=len(pf), dice_vs_author_t0=pf[pf.frame == 0].dice_vs_author.mean(),
                         dice_vs_author=w.dice_vs_author.mean())
                for dv, g in w.groupby('dose'): r[f'DF_dose{int(dv)}'] = g.DF.mean()
                for nm, col in [('DF', 'DF'), ('DP', 'dp_mean_fg'), ('fgfc', 'fg_fc')]:
                    v = w.dropna(subset=[col]).reset_index(drop=True)
                    r[f'rho_{nm}_PI'] = spearmanr(v[col], v.pi_img).statistic if np.ptp(v[col]) > 0 else np.nan
                    r[f'rho_{nm}_PI_lo'], r[f'rho_{nm}_PI_hi'] = A.strat_boot(v, lambda d: spearmanr(d[col], d.pi_img).statistic)
                r['prho_DF_PI_given_fgfc'] = A.pspear(w.DF, w.pi_img, w.fg_fc)
                r['prho_DF_lo'], r['prho_DF_hi'] = A.strat_boot(w, lambda d: A.pspear(d.DF, d.pi_img, d.fg_fc))
                fi = th // 4; H = {}; tp = fp = fn = 0
                for xy in sorted(w.xy):
                    z = np.load(base / 'masks' / f'xy{xy:03d}.npz')
                    pred = z['pred'][list(z['frames']).index(fi)] if 'frames' in z else z['pred'][fi]
                    pdead = np.load(base / 'pdead' / f'xy{xy:03d}_t{th}.npy').astype(np.float32)
                    pos = np.asarray(stacks[xy][fi]) > T; fg = (pred == 1) | (pred == 2)
                    bb = np.clip((pdead[fg] * A.NBINS).astype(int), 0, A.NBINS - 1); pp = pos[fg]
                    H[xy] = (np.bincount(bb[pp], minlength=A.NBINS), np.bincount(bb[~pp], minlength=A.NBINS))
                    dead = pred == 2
                    tp += int((dead & pos & fg).sum()); fp += int((dead & ~pos & fg).sum()); fn += int((~dead & pos & fg).sum())
                keys = list(H); rng = np.random.default_rng(A.RNG); est = []
                for _ in range(A.B):
                    s = rng.choice(len(keys), len(keys), replace=True)
                    est.append(A.auc_from_hist(sum(H[keys[i]][0] for i in s), sum(H[keys[i]][1] for i in s)))
                r.update(pooled_auc=A.auc_from_hist(sum(H[k][0] for k in keys), sum(H[k][1] for k in keys)),
                         pooled_auc_lo=float(np.nanpercentile(est, 2.5)), pooled_auc_hi=float(np.nanpercentile(est, 97.5)),
                         dead_precision=tp / max(tp + fp, 1), dead_recall=tp / max(tp + fn, 1))
                rows.append(r); print(f, m, th, flush=True)
    S = pd.DataFrame(rows); S.to_csv(OUT / 'posthoc_scale_summary.csv', index=False)
    pd.set_option('display.width', 250); print(S.set_index(['factor', 'model', 'time_h']).T.round(3).to_string())


if __name__ == '__main__':
    main()
