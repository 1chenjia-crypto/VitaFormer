#!/usr/bin/env python3
"""Stage G analysis (AMENDMENT_1_20261106.md). Validation only. Inputs: stageG_dead_pr_val/export/{arm}_seed{s}/stats.npz.
Usage: stageG_analysis.py OUTDIR ARM [ARM ...]   (arms among M2 nnUNet Full FullnoRD M0; missing arms -> error)

Pixel statistics from per-image score histograms rebinned to 0.05 (exact at every main-grid delta, step 0.1).
Bootstrap: 3 seed positions with replacement, physical val wells with replacement per position, pixels pooled,
10,000 replicates, RNG 20261107, shared draws for all arms. Metric of a replicate = mean over its 3 positions.
"""
import contextlib, io, json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
EXP = Path('3D_OrganoidSeg/vco/runs/arch_exploration_20261104')
sys.path.insert(0, str(EXP / 'scripts')); sys.path.insert(0, '3D_OrganoidSeg/vco/organoid_seg')
import stageG_common as C  # noqa
from dataset import build_dataset_index, stratified_train_val_test_split  # noqa
SEEDS = [42, 3407, 2026]; N_BOOT = 10000; RNG = 20261107; CHUNK = 500
EXPORT = EXP / 'stageG_dead_pr_val/export'
E2 = C.EDGES[::5]                      # 801 coarse edges, step 0.05
NB2 = len(E2) + 1
MAIN_IDX = np.array([int(np.where(np.isclose(E2, d))[0][0]) for d in C.MAIN])  # coarse edge index of each main delta
ZERO = int(np.where(E2 == 0.0)[0][0])
RECALLS = [60.0, 65.0, 70.0, 75.0]
MARGIN = {'prec': 5.0, 'auprc': 0.03, 'dead_iou': 1.0, 'miou': 0.3, 'obj': 5.0}


def rebin(h):  # (..., 4002) fine -> (..., 802) coarse; fine bin j -> floor((j-1)/5)+1, j=0 -> 0
    starts = np.r_[0, np.arange(1, C.NB, 5)]
    assert len(starts) == NB2
    return np.add.reduceat(h, starts, axis=-1)


def wells_of(stems):
    _, wg = build_dataset_index('OrganoidSegBaseline/VOC2012')
    with contextlib.redirect_stdout(io.StringIO()):
        _, val, _ = stratified_train_val_test_split(wg, 0.6, 0.2, seed=42, use_condition=True)
    s2w = {x['stem']: x['well_id'] for x in val}
    assert set(stems) == set(s2w), 'val stems differ from split'
    w = sorted(set(s2w.values())); return np.array([w.index(s2w[s]) for s in stems]), len(w)


def load_arm(arm, ref_stems, wid, nw):
    """-> per seed: dict of per-well arrays H (nw,4,3,NB2), HF (nw,NB2), HP (nw,2,PB)."""
    out = {}
    for s in SEEDS:
        z = np.load(EXPORT / f'{arm}_seed{s}/stats.npz')
        assert list(z['stems']) == ref_stems, (arm, s)
        H, HF, HP = rebin(z['H']), rebin(z['HF']), z['HP']
        agg = lambda a: np.stack([a[wid == k].sum(0) for k in range(nw)])
        out[s] = {'H': agg(H).astype(np.float64), 'HF': agg(HF).astype(np.float64), 'HP': agg(HP).astype(np.float64)}
    return out


# ---------- metrics from pooled histograms (leading batch dims allowed) ----------
def confusion(H, i):
    """H (...,4,3,NB2) pooled; dead iff coarse bin > i. -> conf (...,4 gt,4 pred)."""
    le = H[..., :i + 1].sum(-1)                       # (...,4,3) predicted as 'other' class
    tot = H.sum((-1, -2))                             # (...,4)
    conf = np.zeros(H.shape[:-3] + (4, 4))
    conf[..., :, 0] = le[..., 0]; conf[..., :, 1] = le[..., 1]; conf[..., :, 3] = le[..., 2]
    conf[..., :, 2] = tot - le.sum(-1)
    return conf


def seg_metrics(conf):
    tp = np.diagonal(conf, axis1=-2, axis2=-1); fp = conf.sum(-2) - tp; fn = conf.sum(-1) - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        iou = tp / (tp + fp + fn)
        return {'mIoU': 100 * np.nanmean(iou, -1), 'Alive IoU': 100 * iou[..., 1], 'Dead IoU': 100 * iou[..., 2],
                'Bubble IoU': 100 * iou[..., 3], 'Dead precision': 100 * tp[..., 2] / (tp[..., 2] + fp[..., 2]),
                'Dead recall': 100 * tp[..., 2] / (tp[..., 2] + fn[..., 2])}


def pr_from_hist(pos, neg):
    """pos/neg (...,B) histograms over increasing score bins. Thresholds between bins, from high to low.
    Returns TP, FP cumulative from the top: arrays (...,B+1), index t = number of top bins included."""
    tp = np.concatenate([np.zeros(pos.shape[:-1] + (1,)), np.cumsum(pos[..., ::-1], -1)], -1)
    fp = np.concatenate([np.zeros(neg.shape[:-1] + (1,)), np.cumsum(neg[..., ::-1], -1)], -1)
    return tp, fp


def auprc(pos, neg):
    tp, fp = pr_from_hist(pos, neg)
    P = tp[..., -1:]
    with np.errstate(invalid='ignore', divide='ignore'):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
    dr = np.diff(tp, axis=-1) / P                       # recall gained in each bin
    return np.sum(dr * prec[..., 1:], -1)               # step (AP-style) integration, ties as one block


def prec_at_recall(pos, neg, R):
    """Precision at recall R (percent), linear in TP/FP within the crossing bin."""
    tp, fp = pr_from_hist(pos, neg); P = tp[..., -1]
    target = R / 100.0 * P
    out = np.empty(P.shape)
    flat_tp = tp.reshape(-1, tp.shape[-1]); flat_fp = fp.reshape(-1, fp.shape[-1]); ft = target.reshape(-1)
    res = np.empty(len(ft))
    for n in range(len(ft)):
        t = np.searchsorted(flat_tp[n], ft[n], side='left')
        t = min(max(t, 1), flat_tp.shape[1] - 1)
        a_tp, b_tp = flat_tp[n, t - 1], flat_tp[n, t]
        f = 0.0 if b_tp == a_tp else (ft[n] - a_tp) / (b_tp - a_tp)
        tpv = a_tp + f * (b_tp - a_tp); fpv = flat_fp[n, t - 1] + f * (flat_fp[n, t] - flat_fp[n, t - 1])
        res[n] = 100 * tpv / (tpv + fpv) if tpv + fpv > 0 else np.nan
    return res.reshape(P.shape)


def all_metrics(H, HF, HP, r_ref):
    """Pooled stats (batch dims allowed) -> dict of metric arrays."""
    pos = H[..., 2, :, :].sum(-2); neg = H[..., [0, 1, 3], :, :].sum((-2, -3))
    m = {f'{k}@argmax': v for k, v in seg_metrics(confusion(H, ZERO)).items()}
    m['farbg Dead FP px@argmax'] = HF[..., ZERO + 1:].sum(-1)
    m['AUPRC (log-ratio)'] = auprc(pos, neg)
    m['AUPRC (p_dead)'] = auprc(HP[..., 1, :], HP[..., 0, :])
    m['Precision@R_ref'] = prec_at_recall(pos, neg, r_ref)
    for R in RECALLS: m[f'Precision@R{R:.0f}'] = prec_at_recall(pos, neg, R)
    return m


def delta_curve(H, HF):
    """Per main-grid delta: seg metrics + far-bg FP. H (4,3,NB2) pooled."""
    rows = []
    for d, i in zip(C.MAIN, MAIN_IDX):
        sm = seg_metrics(confusion(H, i)); sm = {k: float(v) for k, v in sm.items()}
        rows.append({'delta': d, **sm, 'farbg Dead FP px': float(HF[i + 1:].sum())})
    return rows


def classify(lo95, hi95, lo90, hi90, margin):
    if lo95 > 0 or hi95 < 0: return 'difference supported'
    if lo90 > -margin and hi90 < margin: return f'equivalent within ±{margin}'
    return 'inconclusive'


def main():
    out = Path(sys.argv[1]); arms = sys.argv[2:]
    out.mkdir(parents=True, exist_ok=False)
    ref = list(np.load(EXPORT / f'{arms[0]}_seed42/stats.npz')['stems'])
    wid, nw = wells_of(ref)
    D = {a: load_arm(a, ref, wid, nw) for a in arms}
    pooled = {a: {s: {k: v.sum(0) for k, v in D[a][s].items()} for s in SEEDS} for a in arms}
    # reference recall = nnU-Net 3-seed mean argmax Dead recall (observed, fixed)
    if 'nnUNet' in arms:
        r_ref = float(np.mean([seg_metrics(confusion(pooled['nnUNet'][s]['H'], ZERO))['Dead recall'] for s in SEEDS]))
    else:
        r_ref = 65.0
    # observed
    obs = {a: {s: all_metrics(pooled[a][s]['H'], pooled[a][s]['HF'], pooled[a][s]['HP'], r_ref) for s in SEEDS} for a in arms}
    if 'FullnoRD' in arms:
        r_nord = float(np.mean([obs['FullnoRD'][s]['Dead recall@argmax'] for s in SEEDS]))
        for a in arms:
            for s in SEEDS:
                p = pooled[a][s]['H']; pos = p[2].sum(0); neg = p[[0, 1, 3]].sum((0, 1))
                obs[a][s]['Precision@R_noRD'] = prec_at_recall(pos, neg, r_nord)
    else:
        r_nord = None
    keys = list(obs[arms[0]][42].keys())
    rows = []
    for a in arms:
        for k in keys:
            v = np.array([float(obs[a][s][k]) for s in SEEDS])
            rows.append({'arm': a, 'metric': k, **{f'seed{s}': x for s, x in zip(SEEDS, v)}, 'mean': v.mean(), 'sd': v.std(ddof=1)})
    pd.DataFrame(rows).to_csv(out / 'seed_level.csv', index=False)
    # delta curves (per arm, 3-seed mean of per-seed pooled metrics)
    crow = []
    for a in arms:
        per = [delta_curve(pooled[a][s]['H'], pooled[a][s]['HF']) for s in SEEDS]
        for j, d in enumerate(C.MAIN):
            r = {'arm': a, 'delta': d}
            for k in per[0][j]:
                if k != 'delta': r[k] = float(np.mean([per[t][j][k] for t in range(3)]))
            crow.append(r)
    curves = pd.DataFrame(crow); curves.to_csv(out / 'delta_curves.csv', index=False)
    # PR curve points for plotting (3-seed mean precision at recall grid)
    prow = []
    for a in arms:
        for R in np.arange(5, 96, 1.0):
            v = [float(prec_at_recall(pooled[a][s]['H'][2].sum(0), pooled[a][s]['H'][[0, 1, 3]].sum((0, 1)), R)) for s in SEEDS]
            prow.append({'arm': a, 'recall': R, 'precision_mean': np.mean(v), **{f'seed{s}': x for s, x in zip(SEEDS, v)}})
    pd.DataFrame(prow).to_csv(out / 'pr_curves.csv', index=False)

    # ---------- bootstrap ----------
    rng = np.random.default_rng(RNG)
    sp = rng.integers(0, 3, (N_BOOT, 3))
    cnt = np.stack([np.stack([np.bincount(rng.integers(0, nw, nw), minlength=nw) for _ in range(3)]) for _ in range(N_BOOT)]).astype(np.float64)
    bkeys = ['Dead IoU@argmax', 'mIoU@argmax', 'Dead precision@argmax', 'Dead recall@argmax', 'AUPRC (log-ratio)',
             'AUPRC (p_dead)', 'Precision@R_ref'] + [f'Precision@R{R:.0f}' for R in RECALLS] + (['Precision@R_noRD'] if r_nord else [])
    boot = {a: {k: np.empty((N_BOOT, 3)) for k in bkeys} for a in arms}
    for a in arms:
        for s_i, s in enumerate(SEEDS):
            Wh = D[a][s]['H'].reshape(nw, -1); Wp = D[a][s]['HP'].reshape(nw, -1); Wf = D[a][s]['HF']
            for b0 in range(0, N_BOOT, CHUNK):
                sl = slice(b0, b0 + CHUNK)
                for j in range(3):
                    sel = np.where(sp[sl, j] == s_i)[0]
                    if not len(sel): continue
                    w = cnt[sl][sel, j]
                    H = (w @ Wh).reshape((len(sel),) + D[a][s]['H'].shape[1:]); HP = (w @ Wp).reshape((len(sel),) + D[a][s]['HP'].shape[1:])
                    HF = w @ Wf
                    m = all_metrics(H, HF, HP, r_ref)
                    if r_nord:
                        m['Precision@R_noRD'] = prec_at_recall(H[:, 2].sum(1), H[:, [0, 1, 3]].sum((1, 2)), r_nord)
                    for k in bkeys: boot[a][k][b0 + sel, j] = m[k]
        print('bootstrap done', a, flush=True)
    np.savez_compressed(out / 'bootstrap.npz', **{f'{a}|{k}': boot[a][k] for a in arms for k in bkeys})
    CON = [('M2 - nnUNet', 'M2', 'nnUNet'), ('Full - FullnoRD', 'Full', 'FullnoRD'), ('M2 - M0', 'M2', 'M0'),
           ('M0 - nnUNet', 'M0', 'nnUNet'), ('FullnoRD - nnUNet', 'FullnoRD', 'nnUNet')]
    mg = lambda k: MARGIN['auprc'] if 'AUPRC' in k else MARGIN['dead_iou'] if 'Dead IoU' in k else MARGIN['miou'] if 'mIoU' in k else MARGIN['prec']
    con = []
    for name, x, y in CON:
        if x not in arms or y not in arms: continue
        for k in bkeys:
            bd = boot[x][k].mean(1) - boot[y][k].mean(1)
            per = np.array([float(obs[x][s][k]) - float(obs[y][s][k]) for s in SEEDS])
            lo95, hi95 = np.nanpercentile(bd, [2.5, 97.5]); lo90, hi90 = np.nanpercentile(bd, [5, 95])
            con.append({'contrast': name, 'metric': k, 'delta_mean': per.mean(), 'ci95': f'[{lo95:.3f}, {hi95:.3f}]',
                        'ci90': f'[{lo90:.3f}, {hi90:.3f}]', 'seed_deltas': ';'.join(f'{v:+.3f}' for v in per),
                        'margin': mg(k), 'reading': classify(lo95, hi95, lo90, hi90, mg(k))})
    pd.DataFrame(con).to_csv(out / 'contrasts.csv', index=False)

    # ---------- threshold: full-val optimum (optimistic) + well-grouped 5-fold CV x 20 ----------
    thr = []
    cv_conf = {}
    for a in arms:
        cm = curves[curves.arm == a].reset_index(drop=True)
        jopt = int(cm['mIoU'].idxmax()); jd = int(cm['Dead IoU'].idxmax())
        thr.append({'arm': a, 'kind': 'full-val optimum (optimistic)', 'delta_mIoU': float(cm.delta[jopt]),
                    'mIoU': cm['mIoU'][jopt], 'Dead IoU': cm['Dead IoU'][jopt], 'delta_DeadIoU': float(cm.delta[jd]),
                    'DeadIoU_at_delta_DeadIoU': cm['Dead IoU'][jd], 'mIoU_argmax': cm['mIoU'][40], 'DeadIoU_argmax': cm['Dead IoU'][40]})
        # per-well confusion at each main delta: (seed, nw, 81, 4, 4)
        WC = np.stack([np.stack([confusion(D[a][s]['H'], i) for i in MAIN_IDX], 1) for s in SEEDS])
        oof = np.zeros((3, nw, 4, 4)); chosen = []
        rr = np.random.default_rng(RNG)
        for rep in range(20):
            fold = rr.permutation(np.arange(nw) % 5)
            for f in range(5):
                tr = fold != f
                mi = np.mean([seg_metrics(WC[t][tr].sum(0))['mIoU'] for t in range(3)], 0)  # (81,)
                j = int(np.nanargmax(mi)); chosen.append(float(C.MAIN[j]))
                oof[:, ~tr] += WC[:, ~tr, j]
        oof /= 20.0
        cv_conf[a] = oof
        base = WC[:, :, 40]
        cvm = [seg_metrics(oof[t].sum(0)) for t in range(3)]; bm = [seg_metrics(base[t].sum(0)) for t in range(3)]
        # well bootstrap of the CV gain (seed positions + wells, same machinery; selection variability not re-run)
        gains = {k: [] for k in ('mIoU', 'Dead IoU', 'Dead precision', 'Dead recall')}
        for b in range(2000):
            g = {k: 0.0 for k in gains}
            for j in range(3):
                t = sp[b, j]; w = cnt[b, j]
                c1 = seg_metrics(np.tensordot(w, oof[t], 1)); c0 = seg_metrics(np.tensordot(w, base[t], 1))
                for k in gains: g[k] += (c1[k] - c0[k]) / 3
            for k in gains: gains[k].append(g[k])
        r = {'arm': a, 'kind': 'well-grouped 5-fold CV x20 (out-of-fold)',
             'chosen_delta_distribution': pd.Series(chosen).value_counts().head(5).to_dict()}
        for k in gains:
            per = [cvm[t][k] - bm[t][k] for t in range(3)]
            lo, hi = np.percentile(gains[k], [2.5, 97.5]); lo9, hi9 = np.percentile(gains[k], [5, 95])
            r[f'gain {k}'] = float(np.mean(per)); r[f'gain {k} ci95'] = f'[{lo:.3f}, {hi:.3f}]'
            r[f'gain {k} seeds'] = ';'.join(f'{v:+.3f}' for v in per)
            r[f'gain {k} reading'] = classify(lo, hi, lo9, hi9, MARGIN['miou'] if k == 'mIoU' else MARGIN['dead_iou'] if k == 'Dead IoU' else MARGIN['prec'])
        thr.append(r)
    (out / 'threshold.json').write_text(json.dumps(thr, indent=1, default=float))
    (out / 'manifest.json').write_text(json.dumps({
        'script': __file__, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'arms': arms,
        'n_val_images': len(ref), 'n_val_wells': nw, 'R_ref_nnUNet_argmax_recall': r_ref, 'R_noRD_argmax_recall': r_nord,
        'bootstrap': f'{N_BOOT}, RNG {RNG}', 'cv': '5-fold by well x 20 partitions, RNG 20261108; gain CI 2000 boot',
        'inputs_sha256': {f'{a}_seed{s}': hashlib.sha256((EXPORT / f'{a}_seed{s}/stats.npz').read_bytes()).hexdigest()
                          for a in arms for s in SEEDS}}, indent=2))
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 60)
    print(pd.DataFrame(rows).round(3).to_string(index=False)); print(pd.DataFrame(con).to_string(index=False))
    print(json.dumps(thr, indent=1, default=float))


if __name__ == '__main__':
    main()
