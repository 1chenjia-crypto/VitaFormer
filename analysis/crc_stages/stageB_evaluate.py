#!/usr/bin/env python3
"""Stage B evaluation of a candidate run on the 164 CRC VALIDATION images (plan §4.3–4.4; fixed before results).

Usage: stageB_evaluate.py --run_dir DIR --arch m2 --out DIR_NEW [--candidate_masks DIR] [--skip_infer]
  * Inference (frozen protocol, a0_m0_val_inference.infer_hard) of run_dir/best_miou.pth -> masks;
    plus the CNN-participation variant (same weights, CNN input zeroed at inference) -> masks_zero_cnn.
  * Reference: M0 seed 42 whole-image validation masks from A0 (paired), M0 3-seed reference from A0.
  * --candidate_masks + --skip_infer: pipeline test on existing masks (no model).
The locked test set, BrainOrganoid and DeepOrganoid are never read.
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path('3D_OrganoidSeg')
EXP = ROOT / 'vco/runs/arch_exploration_20261104'
VOC = Path('OrganoidSegBaseline/VOC2012')
GT_DIR = VOC / 'SegmentationClass'
IMG_DIR = VOC / 'JPEGImages'
VAL = VOC / 'ImageSets/Segmentation/val_cond.txt'
M0_42 = EXP / 'A0_gap_decomposition/m0_val_masks/seed42'
BCLIP = 'pretrained/biomedclip/open_clip_pytorch_model.bin'
sys.path.insert(0, str(ROOT / 'vco/organoid_seg'))
sys.path.insert(0, str(EXP / 'scripts'))
from dataset import parse_filename  # noqa: E402
from a0_m0_val_inference import infer_hard  # noqa: E402
from a0_gap_decomposition import gt_regions, conf, REGIONS  # noqa: E402

NC = 4
EDGES = [139.0, 313.0]  # A0 validation-GT component tertile edges (frozen)
TOLS = (1, 2, 4, 8)
EIGHT = np.ones((3, 3), bool)
RULE = {'mIoU': 82.50, 'iou_dead': 43.88, 'dead_recall': 74.18, 'fg_bf1_2px': 0.830, 'latency_ratio': 1.5,
        'nogo_mIoU_le': 81.62, 'nogo_dead_recall_lt': 69.18, 'orig_dead_iou_plus2': 46.55}


def boundary(mask):
    mask = np.asarray(mask, bool)
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def bmetrics(gt_mask, pred_mask):
    gb, pb = boundary(gt_mask), boundary(pred_mask)
    ng, npd = int(gb.sum()), int(pb.sum())
    if ng == 0 and npd == 0:
        return {f'bf1_{t}': 1.0 for t in TOLS} | {'assd': np.nan, 'hd95': np.nan}
    if ng == 0 or npd == 0:
        return {f'bf1_{t}': 0.0 for t in TOLS} | {'assd': np.nan, 'hd95': np.nan}
    dg = ndi.distance_transform_edt(~gb)
    dp = ndi.distance_transform_edt(~pb)
    pdist, gdist = dg[pb], dp[gb]
    out = {}
    for t in TOLS:
        p, r = float((pdist <= t).mean()), float((gdist <= t).mean())
        out[f'bf1_{t}'] = 2 * p * r / (p + r) if p + r else 0.0
    comb = np.concatenate([pdist, gdist])
    out['assd'] = float(comb.mean())
    out['hd95'] = float(np.percentile(comb, 95))
    return out


def objects(pred, gt, cls):
    lab, n = ndi.label(gt == cls, structure=EIGHT)
    if not n:
        return []
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(lab), lab, idx)
    corr = ndi.sum(pred == cls, lab, idx)
    return [(float(a), float(c / a)) for a, c in zip(area, corr) if a >= 20]


def per_image(args):
    stem, cand_dir, extra = args
    gt = np.asarray(Image.open(GT_DIR / f'{stem}.png'), dtype=np.int16)
    gtv = gt.copy(); gtv[(gtv >= 250) | (gtv > 3)] = -1
    reg, _ = gt_regions(np.asarray(Image.open(GT_DIR / f'{stem}.png')), EDGES)
    out = {'stem': stem, 'well': parse_filename(stem)['well_id'], 'has_dead': bool((gtv == 2).any())}
    for tag, d in (('m0', M0_42), ('cand', cand_dir)) + tuple(extra):
        pred = np.asarray(Image.open(Path(d) / f'{stem}.png'), dtype=np.int16)
        valid = gtv >= 0
        out[f'{tag}_cm'] = np.bincount(gtv[valid] * NC + pred[valid], minlength=NC * NC).reshape(NC, NC)
        if tag == 'zero':
            continue
        fg = bmetrics(np.isin(gtv, (1, 2)), np.isin(pred, (1, 2)))
        al = bmetrics(gtv == 1, pred == 1)
        out |= {f'{tag}_fg_{k}': v for k, v in fg.items()} | {f'{tag}_alive_{k}': v for k, v in al.items()}
        out[f'{tag}_regcm'] = {k: conf(pred.astype(np.uint8), np.asarray(Image.open(GT_DIR / f'{stem}.png')), reg[k]) for k in REGIONS}
        out[f'{tag}_obj'] = {c: objects(pred, gtv, ci) for ci, c in ((1, 'alive'), (2, 'dead'))}
        out[f'{tag}_far_dead_fp'] = int(((pred == 2) & (gtv != 2) & reg['FARBG']).sum())
    return out


def summ(cm):
    cm = np.asarray(cm, float)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    iou = np.divide(tp, tp + fp + fn, out=np.full(NC, np.nan), where=(tp + fp + fn) > 0) * 100
    rec = np.divide(tp, tp + fn, out=np.full(NC, np.nan), where=(tp + fn) > 0) * 100
    pre = np.divide(tp, tp + fp, out=np.full(NC, np.nan), where=(tp + fp) > 0) * 100
    return {'mIoU': float(np.nanmean(iou)), 'iou_background': iou[0], 'iou_alive': iou[1], 'iou_dead': iou[2],
            'iou_bubble': iou[3], 'alive_precision': pre[1], 'alive_recall': rec[1], 'dead_precision': pre[2],
            'dead_recall': rec[2]}


def obj_table(recs, tag):
    rows = []
    for c in ('alive', 'dead'):
        a = np.array([o for r in recs for o in r[f'{tag}_obj'][c]])
        for name, m in (('all', np.ones(len(a), bool)), ('small', a[:, 0] < EDGES[0]),
                        ('mid', (a[:, 0] >= EDGES[0]) & (a[:, 0] < EDGES[1])), ('large', a[:, 0] >= EDGES[1])):
            s = a[m]
            rows.append({'model': tag, 'class': c, 'size': name, 'n': int(m.sum()),
                         'detected(>=50%)': float((s[:, 1] >= .5).mean()) if len(s) else np.nan,
                         'missed(<10%)': float((s[:, 1] < .1).mean()) if len(s) else np.nan})
    return rows


def boot(recs, n=10000, seed=20261104):
    wells = sorted({r['well'] for r in recs})
    wi = np.array([wells.index(r['well']) for r in recs])
    rng = np.random.default_rng(seed)
    keys = ['mIoU', 'iou_alive', 'iou_dead', 'iou_bubble', 'dead_recall', 'dead_precision']
    cm_m0 = np.stack([r['m0_cm'] for r in recs]); cm_c = np.stack([r['cand_cm'] for r in recs])
    wm0 = np.zeros((len(wells), NC, NC)); wc = np.zeros_like(wm0)
    np.add.at(wm0, wi, cm_m0); np.add.at(wc, wi, cm_c)
    img_keys = ['fg_bf1_2', 'alive_bf1_2']
    iv = {k: (np.array([r[f'cand_{k}'] for r in recs]) - np.array([r[f'm0_{k}'] for r in recs])) for k in img_keys}
    wsum = {k: np.bincount(wi, iv[k], len(wells)) for k in img_keys}
    wcnt = np.bincount(wi, minlength=len(wells)).astype(float)
    point = {k: summ(wc.sum(0))[k] - summ(wm0.sum(0))[k] for k in keys} | {k: iv[k].mean() for k in img_keys}
    draws = {k: [] for k in point}
    for _ in range(n):
        idx = rng.integers(0, len(wells), len(wells))
        a, b = summ(wc[idx].sum(0)), summ(wm0[idx].sum(0))
        for k in keys:
            draws[k].append(a[k] - b[k])
        for k in img_keys:
            draws[k].append(wsum[k][idx].sum() / wcnt[idx].sum())
    return [{'endpoint': k, 'delta_cand_minus_m0s42': point[k], 'ci_lo': float(np.nanpercentile(draws[k], 2.5)),
             'ci_hi': float(np.nanpercentile(draws[k], 97.5))} for k in point], len(wells)


def latency(model_cand, device):
    from model_v2 import OrganoidSegV2
    m0 = OrganoidSegV2(weight_path=BCLIP, num_classes=4, freeze_blocks=0, use_mpm=False, use_bam=False)
    st = torch.load(ROOT / 'vco/runs/ablation_no_bam_no_mpm_seed42/best_miou.pth', map_location='cpu', weights_only=True)
    m0.load_state_dict(st.get('model_state', st), strict=True); m0 = m0.eval().to(device)
    stem = [s.strip() for s in VAL.read_text().splitlines() if s.strip()][0]
    img = np.asarray(Image.open(IMG_DIR / f'{stem}.jpg').convert('RGB'))
    res = {}
    for tag, m in (('m0', m0), ('cand', model_cand), ('m0_again', m0)):
        t = []
        for _ in range(5):
            torch.cuda.synchronize(); s = time.time(); infer_hard(m, img, device); torch.cuda.synchronize(); t.append(time.time() - s)
        res[tag] = float(np.median(t))
    del m0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir'); ap.add_argument('--arch', default='m2'); ap.add_argument('--out', required=True)
    ap.add_argument('--candidate_masks'); ap.add_argument('--skip_infer', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    stems = [s.strip() for s in VAL.read_text().splitlines() if s.strip()]
    meta = {'rule': RULE, 'plan_sha256_frozen': (EXP / 'stageB_frozen_hashes.txt').read_text()}
    extra = []
    if a.skip_infer:
        cand = Path(a.candidate_masks)
        meta['mode'] = 'PIPELINE TEST on existing masks (not a Stage B result)'
    else:
        from model_v3_multiscale import OrganoidSegMultiScale
        dev = torch.device('cuda')
        ck = Path(a.run_dir) / 'best_miou.pth'
        model = OrganoidSegMultiScale(BCLIP, num_classes=4, freeze_blocks=0, zero_cnn_input=(a.arch == 'm1'))
        st = torch.load(ck, map_location='cpu', weights_only=True)
        model.load_state_dict(st.get('model_state', st), strict=True); model = model.eval().to(dev)
        meta['checkpoint'] = str(ck); meta['checkpoint_step'] = int(st.get('step', -1)) if isinstance(st, dict) else None
        cand, zero = out / 'masks', out / 'masks_zero_cnn'
        cand.mkdir(); zero.mkdir()
        for i, s in enumerate(stems, 1):
            img = np.asarray(Image.open(IMG_DIR / f'{s}.jpg').convert('RGB'))
            model.zero_cnn_input = (a.arch == 'm1')
            Image.fromarray(infer_hard(model, img, dev)).save(cand / f'{s}.png')
            model.zero_cnn_input = True
            Image.fromarray(infer_hard(model, img, dev)).save(zero / f'{s}.png')
            if i % 40 == 0:
                print(f'infer {i}/{len(stems)}', flush=True)
        model.zero_cnn_input = (a.arch == 'm1')
        meta['latency_s_median'] = latency(model, dev)
        meta['latency_ratio'] = meta['latency_s_median']['cand'] / np.mean([meta['latency_s_median']['m0'], meta['latency_s_median']['m0_again']])
        extra = [('zero', zero)]
        del model; torch.cuda.empty_cache()
    with Pool(24) as pool:
        recs = pool.map(per_image, [(s, cand, extra) for s in stems], chunksize=2)
    # pooled metrics
    tab = []
    for tag in ['m0', 'cand'] + [e[0] for e in extra]:
        row = {'model': tag} | summ(np.sum([r[f'{tag}_cm'] for r in recs], axis=0))
        if tag != 'zero':
            for k in [f'fg_bf1_{t}' for t in TOLS] + [f'alive_bf1_{t}' for t in TOLS]:
                row[k] = float(np.mean([r[f'{tag}_{k}'] for r in recs]))
            for k in ('fg_hd95', 'fg_assd', 'alive_hd95', 'alive_assd'):
                row[k] = float(np.nanmean([r[f'{tag}_{k}'] for r in recs]))
            row['far_bg_dead_fp_px'] = int(sum(r[f'{tag}_far_dead_fp'] for r in recs))
        tab.append(row)
    met = pd.DataFrame(tab); met.to_csv(out / 'val_metrics.csv', index=False)
    pd.DataFrame(obj_table(recs, 'm0') + obj_table(recs, 'cand')).to_csv(out / 'val_objects.csv', index=False)
    # region error counts (A0 regions), cand vs m0
    reg_rows = []
    for tag in ('m0', 'cand'):
        for k in REGIONS:
            cm = np.sum([r[f'{tag}_regcm'][k] for r in recs], axis=0)
            reg_rows.append({'model': tag, 'region': k, 'error_px': int(cm.sum() - np.trace(cm))})
    pd.DataFrame(reg_rows).to_csv(out / 'val_region_errors.csv', index=False)
    bt, nw = boot(recs); pd.DataFrame(bt).to_csv(out / 'paired_vs_m0_seed42.csv', index=False)
    # CNN participation
    if extra:
        flips = []
        for s in stems:
            p = np.asarray(Image.open(cand / f'{s}.png')); z = np.asarray(Image.open(extra[0][1] / f'{s}.png'))
            fgu = np.isin(p, (1, 2)) | np.isin(z, (1, 2))
            flips.append(((p != z).mean(), (p != z)[fgu].mean() if fgu.any() else 0.0))
        meta['cnn_participation'] = {'pixel_flip_rate_mean': float(np.mean([f[0] for f in flips])),
                                     'foreground_union_flip_rate_mean': float(np.mean([f[1] for f in flips]))}
    # examples (fixed rule): 3 largest / 3 smallest per-image fg BF1 change
    d = pd.DataFrame({'stem': [r['stem'] for r in recs], 'delta_fg_bf1_2': [r['cand_fg_bf1_2'] - r['m0_fg_bf1_2'] for r in recs]})
    d.sort_values('delta_fg_bf1_2').to_csv(out / 'per_image_fg_bf1_delta.csv', index=False)
    meta['examples'] = {'worst3': d.nsmallest(3, 'delta_fg_bf1_2').stem.tolist(), 'best3': d.nlargest(3, 'delta_fg_bf1_2').stem.tolist()}
    # decision
    c = met[met.model == 'cand'].iloc[0]
    lat_ok = meta.get('latency_ratio', np.nan) <= RULE['latency_ratio'] if 'latency_ratio' in meta else None
    checks = {'mIoU>=82.50': bool(c.mIoU >= RULE['mIoU']), 'deadIoU>=43.88': bool(c.iou_dead >= RULE['iou_dead']),
              'deadRecall>=74.18': bool(c.dead_recall >= RULE['dead_recall']), 'fgBF1_2px>0.830': bool(c.fg_bf1_2 > RULE['fg_bf1_2px']),
              'latency<=1.5x': lat_ok, 'orig_rule_deadIoU>=46.55 (history only)': bool(c.iou_dead >= RULE['orig_dead_iou_plus2'])}
    nogo = bool(c.mIoU <= RULE['nogo_mIoU_le'] or c.dead_recall < RULE['nogo_dead_recall_lt'])
    go = all(v for k, v in checks.items() if 'history' not in k and v is not None) and lat_ok is not False
    meta['checks'] = checks; meta['n_wells_bootstrap'] = nw
    meta['outcome'] = 'GO' if go else ('NO-GO' if nogo else 'INCONCLUSIVE')
    (out / 'stageB_summary.json').write_text(json.dumps(meta, indent=2, default=float))
    print(met.round(3).to_string(index=False)); print(json.dumps({'checks': checks, 'outcome': meta['outcome']}, indent=1))


if __name__ == '__main__':
    main()
