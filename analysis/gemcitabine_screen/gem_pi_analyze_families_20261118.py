#!/usr/bin/env python3
"""GemcitabineScreen PI Dead-class validation — analysis (plan: external_m2_20261108/gemcitabine_pi/ANALYSIS_PLAN.md).
Writes into gemcitabine_pi/analysis (new directory only).
"""
from pathlib import Path
import json, re, sys, hashlib
import numpy as np, pandas as pd, scipy.optimize
from scipy.stats import spearmanr, rankdata
from PIL import Image

ROOT = Path('3D_OrganoidSeg')
DATA = ROOT / 'external_datasets/OrganoID_Data/GemcitabineScreen'
AUTH = ROOT / 'external_datasets/OrganoID/Publication/Figure3'
D = ROOT / 'baselines_fair_20261030/external_m2_20261108/gemcitabine_pi'; OUT = D / 'analysis_families'
MODELS = [f'{f}_seed{s}' for f in ('m2', 'segformer', 'nnunet') for s in (42, 3407, 2026)]; PRIMARY = 'm2_seed42'
B, RNG = 10000, 20261109; NBINS = 1000
AUTHOR_EC50 = dict(MTS=26.2, PI=16.9, OrganoID_area=12.7, tracked_PI_per_area=24.5)  # author fit code on author CSVs (plan sec. 1)


def frames(p):
    im = Image.open(p); r = []
    for t in range(im.n_frames): im.seek(t); r.append(np.array(im))
    return r


def xy_of(n): return int(re.findall(r'XY(\d+)', n)[0])


def pspear(x, y, z):
    """partial Spearman: Pearson of rank residuals after linear regression on rank(z)."""
    if np.ptp(np.asarray(x, float)) == 0 or np.ptp(np.asarray(y, float)) == 0: return np.nan
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z); Z = np.c_[np.ones_like(rz), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def strat_boot(df, fn, seed=RNG):
    rng = np.random.default_rng(seed); groups = [g.index.to_numpy() for _, g in df.groupby('dose')]; est = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]); est.append(fn(df.loc[idx]))
    est = np.array(est, float); est = est[np.isfinite(est)]
    if len(est) < 0.5 * B: return np.nan, np.nan   # statistic undefined (e.g. constant Dead fraction) -> NA, gate fails
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def auc_from_hist(pos, neg):
    """AUROC from binned score histograms (ties within a bin count 1/2)."""
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0: return np.nan
    cneg = np.cumsum(neg) - neg
    return float((pos * (cneg + 0.5 * neg)).sum() / (P * N))


def fsig(x, a, b, c, d): return c / (1.0 + np.exp(a * (np.log(x) - b))) + d


def ec50(s):
    m = s[s.index != 0].groupby(level=0).mean()
    try:
        p, _ = scipy.optimize.curve_fit(fsig, m.index.to_numpy(float), m.values,
                                        p0=[0, np.mean(np.log(m.index.to_numpy(float))), m.max() - m.min(), m.min()], method='dogbox')
        return float(np.exp(p[1]))
    except Exception as e:
        return f'fit failed: {e}'


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    sel = json.loads((D / 'scale_select/chosen.json').read_text())
    OUT.mkdir(); log = {'scale_select': sel}
    # ---- PI readouts + sanity check vs author PIMeasurements.csv
    pis = sorted((DATA / 'PI').glob('*.tif')); auth = pd.read_csv(AUTH / 'PIMeasurements.csv')
    auth = auth.set_index(['Dosage', 'Replicate', 'Time'])['Fluorescence']
    dos = [np.inf, 0, None, 1000, 300, 100, 30, 10, 3]; pirows = []; stacks = {}
    for p in pis:
        xy = xy_of(p.name); fr = frames(p); stacks[xy] = fr
        for t, f in enumerate(fr):
            pirows.append(dict(xy=xy, dose=dos[(xy - 1) // 6], replicate=xy % 6, frame=t, time_h=4 * t,
                               pi_img=int(np.sum(np.asarray(f) / 1000))))
    pi = pd.DataFrame(pirows)
    mism = sum(int(auth.loc[(r.dose, r.replicate, r.time_h)]) != r.pi_img for r in pi.itertuples())
    log['pi_sanity'] = dict(rows=len(pi), author_rows=len(auth), mismatches=int(mism))
    if mism or len(pi) != len(auth):
        (OUT / 'log.json').write_text(json.dumps(log, indent=2)); raise RuntimeError(f'PI sanity failed: {mism} mismatches')
    ctrl0 = np.concatenate([stacks[xy][0].ravel() for xy in pi.query('dose == 0 and frame == 0').xy])
    T = float(np.percentile(ctrl0, 99)); log['pi_threshold_T'] = T
    pi.to_csv(OUT / 'pi_per_frame.csv', index=False)
    # ---- per model
    summary, timecourse, pix = [], [], []
    for m in MODELS:
        pf = pd.read_csv(D / m / 'per_frame.csv').merge(pi[['xy', 'frame', 'pi_img']], on=['xy', 'frame'], validate='1:1')
        pf['DF'] = np.where(pf.fg_px > 0, pf.dead_px / pf.fg_px.where(pf.fg_px > 0), np.nan)
        base = pf[pf.frame == 0].set_index('xy')
        pf['fg_fc'] = pf.fg_px / pf.xy.map(base.fg_px).replace(0, np.nan)
        pf['author_fc'] = pf.author_fg_px / pf.xy.map(base.author_fg_px).replace(0, np.nan)
        pf.to_csv(OUT / f'{m}_per_frame_merged.csv', index=False)
        for th in (52, 72):
            w = pf[pf.time_h == th].reset_index(drop=True); r = dict(model=m, time_h=th, n_wells=len(w),
                                                                      n_df_na=int(w.DF.isna().sum()))
            ok = w.dropna(subset=['DF', 'fg_fc']).reset_index(drop=True)
            for nm, col in [('DF', 'DF'), ('DP', 'dp_mean_fg'), ('M2_fg_fc', 'fg_fc'), ('author_area_fc', 'author_fc')]:
                v = ok.dropna(subset=[col]).reset_index(drop=True)
                rho = spearmanr(v[col], v.pi_img).statistic; lo, hi = strat_boot(v, lambda d: spearmanr(d[col], d.pi_img).statistic)
                r.update({f'rho_{nm}_PI': rho, f'rho_{nm}_PI_lo': lo, f'rho_{nm}_PI_hi': hi})
            for nm, col in [('DF', 'DF'), ('DP', 'dp_mean_fg')]:
                v = ok.dropna(subset=[col]).reset_index(drop=True)
                r[f'prho_{nm}_PI_given_fgfc'] = pspear(v[col], v.pi_img, v.fg_fc)
                r[f'prho_{nm}_PI_given_fgfc_lo'], r[f'prho_{nm}_PI_given_fgfc_hi'] = strat_boot(v, lambda d: pspear(d[col], d.pi_img, d.fg_fc))
            ld = np.log10(ok.dose.astype(float) + 1)
            r['rho_DF_dose'] = spearmanr(ok.DF, ld).statistic
            r['rho_DF_dose_lo'], r['rho_DF_dose_hi'] = strat_boot(ok.assign(ld=ld), lambda d: spearmanr(d.DF, d.ld).statistic)
            if m == PRIMARY and th == 52:
                rng = np.random.default_rng(RNG); obs = r['rho_DF_PI']
                if np.isfinite(obs):
                    null = np.array([spearmanr(rng.permutation(ok.DF.to_numpy()), ok.pi_img).statistic for _ in range(B)])
                    r['perm_p_two_sided'] = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (B + 1))
                else:
                    r['perm_p_two_sided'] = np.nan
            s = w.set_index('dose')
            r['EC50_DF'] = ec50(s.DF); r['EC50_M2_fg_px'] = ec50(s.fg_px); r['EC50_PI_img'] = ec50(s.pi_img)
            for k in ('DF', 'dp_mean_fg', 'fg_fc', 'dice_vs_author'):
                for dv, g in w.groupby('dose'): r[f'mean_{k}_dose{int(dv)}'] = g[k].mean()
            summary.append(r)
        for xy, g in pf.groupby('xy'):
            g = g.sort_values('frame').dropna(subset=['DF'])
            timecourse.append(dict(model=m, xy=xy, dose=g.dose.iloc[0], n=len(g),
                                   rho_DF_PI_time=spearmanr(g.DF, g.pi_img).statistic if len(g) > 2 else np.nan,
                                   rho_fgfc_PI_time=spearmanr(g.fg_fc, g.pi_img).statistic if len(g) > 2 else np.nan))
        # ---- pixel level (t52, t72)
        for th in (52, 72):
            fi = th // 4; H = {}; tp = fp = fn = 0
            for xy in sorted(pf.xy.unique()):
                pred = np.load(D / m / 'masks' / f'xy{xy:03d}.npz')['pred'][fi]
                pdead = np.load(D / m / 'pdead' / f'xy{xy:03d}_t{th}.npy').astype(np.float32)
                pos = np.asarray(stacks[xy][fi]) > T; fg = (pred == 1) | (pred == 2)
                b = np.clip((pdead[fg] * NBINS).astype(int), 0, NBINS - 1); pp = pos[fg]
                H[xy] = (np.bincount(b[pp], minlength=NBINS), np.bincount(b[~pp], minlength=NBINS))
                dead = pred == 2; tp += int((dead & pos & fg).sum()); fp += int((dead & ~pos & fg).sum()); fn += int((~dead & pos & fg).sum())
                pix.append(dict(model=m, time_h=th, xy=xy, dose=dos[(xy - 1) // 6], fg_px=int(fg.sum()), pi_pos_in_fg=int(pp.sum()),
                                auc_well=auc_from_hist(*H[xy])))
            keys = list(H); P = sum(H[k][0] for k in keys); N = sum(H[k][1] for k in keys); rng = np.random.default_rng(RNG); est = []
            for _ in range(B):
                s = rng.choice(len(keys), len(keys), replace=True)
                est.append(auc_from_hist(sum(H[keys[i]][0] for i in s), sum(H[keys[i]][1] for i in s)))
            summary_pix = dict(model=m, time_h=th, pooled_auc=auc_from_hist(P, N), pooled_auc_lo=float(np.nanpercentile(est, 2.5)),
                               pooled_auc_hi=float(np.nanpercentile(est, 97.5)),
                               dead_precision_vs_pipos=tp / max(tp + fp, 1), dead_recall_vs_pipos=tp / max(tp + fn, 1),
                               pi_pos_fraction_in_fg=float(P.sum() / max(P.sum() + N.sum(), 1)))
            for r in summary:
                if r['model'] == m and r['time_h'] == th: r.update(summary_pix)
    S = pd.DataFrame(summary); S.to_csv(OUT / 'summary.csv', index=False)
    TC = pd.DataFrame(timecourse); TC.to_csv(OUT / 'timecourse_per_well.csv', index=False)
    TC.assign(group=np.where(TC.dose >= 30, 'treated_ge30', 'ctrl_0_10')).groupby(['model', 'group'])[
        ['rho_DF_PI_time', 'rho_fgfc_PI_time']].median().to_csv(OUT / 'timecourse_summary.csv')
    pd.DataFrame(pix).to_csv(OUT / 'pixel_per_well.csv', index=False)
    p = S[(S.model == PRIMARY) & (S.time_h == 52)].iloc[0]
    gate = dict(primary_rho=p.rho_DF_PI, primary=bool(p.rho_DF_PI_lo > 0), S1=bool(p.prho_DF_PI_given_fgfc_lo > 0), S2=bool(p.pooled_auc_lo > 0.5),
                segmentation_failed=bool(sel['segmentation_failed']))
    gate['claim'] = ('segmentation failed - Dead endpoints not interpretable' if gate['segmentation_failed'] else
                     'Dead class tracks an independent dead-cell stain' if all([gate['primary'], gate['S1'], gate['S2']]) else
                     'Dead fraction follows dose/PI, no evidence beyond organoid shrinkage' if gate['primary'] else
                     'Dead class not externally validated')
    log['gate'] = gate; log['author_ec50_reference'] = AUTHOR_EC50
    log['script_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    (OUT / 'log.json').write_text(json.dumps(log, indent=2, default=float))
    pd.set_option('display.width', 250); print(json.dumps(log, indent=2, default=float)); print(S.T.to_string())


if __name__ == '__main__':
    main()
