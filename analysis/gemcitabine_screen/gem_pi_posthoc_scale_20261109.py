#!/usr/bin/env python3
"""POST HOC (Amendment 1): scale sensitivity. Copy of gem_pi_infer_20261109.py; --factor given explicitly, frames {0,13,18} only,
output posthoc_scale/f<factor>/<model>.
GemcitabineScreen PI Dead-class validation — M2 zero-shot inference (plan: external_m2_20261108/gemcitabine_pi/ANALYSIS_PLAN.md).

--stage scale_select : seed42, t=0 frame of all 42 wells, factors {1.0,0.75,0.5,0.25}; Dice of M2 foreground (Alive|Dead) vs author
                       OrganoID label>0 at native resolution; chooses factor (ties within 0.005 -> larger). Never opens PI files.
--stage full         : one model, chosen factor, all 42 wells x 19 frames. Saves argmax masks (native), per-frame stats, and
                       P(Dead) maps (float16, native) for t=52 h and t=72 h frames. Never opens PI files.
Frozen inference = legacy.infer (224/112, reflect pad, ImageNet norm, fp16 autocast, softmax average), argmax.
Input = per-frame p1-p99 stretch of the uint16 BF frame to uint8, replicated to RGB (same rule as DeepOrganoid Stage 31/36).
"""
from pathlib import Path
import argparse, hashlib, json, re, sys, time
import numpy as np, pandas as pd, torch
from PIL import Image

ROOT = Path('3D_OrganoidSeg')
DATA = ROOT / 'external_datasets/OrganoID_Data/GemcitabineScreen'
OUTROOT = ROOT / 'baselines_fair_20261030/external_m2_20261108/gemcitabine_pi'
RUNS = ROOT / 'vco/runs/arch_exploration_20261104'
MODELS = {
    'm2_seed42': (RUNS / 'stageB_m2_seed42/final.pth', '0f12be44859d06723ebfa35e8416df8cd4b02ff83f3ac153654d0f68df226101'),
    'm2_seed3407': (RUNS / 'stageC_m2_seed3407/final.pth', '1e4bd6afd0a1925094103c991430246b9f1fef09ce5b480e2c47056a7128f585'),
    'm2_seed2026': (RUNS / 'stageC_m2_seed2026/final.pth', 'c268464ee2ae3ee19e2112fc91ae86a95aa5d95086374df526fda9a7aa3225b8'),
}
FACTORS = [1.0, 0.75, 0.5, 0.25]
DOSAGES = [np.inf, 0, None, 1000, 300, 100, 30, 10, 3]   # author ProcessGemcitabineExperiment.py
SAVE_PROB_FRAMES = {13: 52, 18: 72}                     # frame index -> hours (t = 4 h * frame)

sys.path.insert(0, str(ROOT / 'deepOrganoid_v1'))
import extract_full_features as legacy  # noqa: E402
sys.path.insert(0, str(ROOT / 'vco/organoid_seg'))
from model_v3_multiscale import OrganoidSegMultiScale  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()


def xy_of(name): return int(re.findall(r'XY(\d+)', name)[0])


def wells():
    bf = sorted((DATA / 'BF').glob('*.tif')); out = []
    for p in bf:
        xy = xy_of(p.name); lab = DATA / 'OrganoIDProcessed' / (p.stem + '_labeled.tif')
        if not lab.exists(): raise FileNotFoundError(lab)
        out.append(dict(xy=xy, dose=DOSAGES[(xy - 1) // 6], replicate=xy % 6, bf=p, lab=lab))
    if len(out) != 42: raise RuntimeError(f'expected 42 wells, found {len(out)}')
    return out


def frames(path):
    im = Image.open(path); res = []
    for t in range(im.n_frames):
        im.seek(t); res.append(np.array(im))
    return res


def stretch(raw):
    raw = raw.astype(np.float32); lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
    return np.stack([(np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)] * 3, -1)


def predict(model, img, factor, device):
    h, w = img.shape[:2]
    if factor != 1.0:
        small = np.asarray(Image.fromarray(img).resize((round(w * factor), round(h * factor)), Image.BILINEAR))
        prob = legacy.infer(model, small, device)
        pred = np.asarray(Image.fromarray(prob.argmax(0).astype(np.uint8)).resize((w, h), Image.NEAREST))
        pdead = np.asarray(Image.fromarray(prob[2].astype(np.float32)).resize((w, h), Image.BILINEAR))
    else:
        prob = legacy.infer(model, img, device); pred = prob.argmax(0).astype(np.uint8); pdead = prob[2]
    return pred, pdead.astype(np.float32)


def dice(a, b):
    s = a.sum() + b.sum(); return 1.0 if s == 0 else 2 * (a & b).sum() / s


def load_model(name, device):
    ck, sha = MODELS[name]
    if sha256(ck) != sha: raise RuntimeError(f'checkpoint hash mismatch {name}')
    m = OrganoidSegMultiScale(str(legacy.BCLIP), num_classes=4, freeze_blocks=0, zero_cnn_input=False)
    st = torch.load(ck, map_location='cpu', weights_only=True)
    m.load_state_dict(st.get('model_state', st), strict=True)
    return m.eval().to(device)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['scale_select', 'full'], required=True)
    ap.add_argument('--model', choices=sorted(MODELS), default='m2_seed42')
    ap.add_argument('--factor', type=float, choices=[1.0, 0.75, 0.5], required=True); a = ap.parse_args()
    device = torch.device('cuda'); t0 = time.time(); W = wells()
    if a.stage == 'scale_select':
        if a.model != 'm2_seed42': raise ValueError('scale selection uses seed42 only (plan section 2)')
        out = OUTROOT / 'scale_select'; out.mkdir(exist_ok=False); model = load_model(a.model, device); rows = []
        for w in W:
            img = stretch(frames(w['bf'])[0]); gt = frames(w['lab'])[0] > 0
            for f in FACTORS:
                pred, _ = predict(model, img, f, device); fg = (pred == 1) | (pred == 2)
                rows.append(dict(xy=w['xy'], factor=f, dice=dice(fg, gt), fg_px=int(fg.sum()), gt_px=int(gt.sum()),
                                 dead_px=int((pred == 2).sum())))
            print(f"xy{w['xy']} {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(rows); df.to_csv(out / 'scale_select_t0.csv', index=False)
        s = df.groupby('factor')['dice'].mean().sort_index(ascending=False); best = s.max()
        chosen = float([f for f in s.index if s[f] >= best - 0.005][0])   # largest factor within 0.005 of best
        res = dict(mean_dice_by_factor={str(k): float(v) for k, v in s.items()}, chosen_factor=chosen, best_mean_dice=float(best),
                   segmentation_failed=bool(s[chosen] < 0.5), script_sha256=sha256(__file__), elapsed_s=time.time() - t0)
        (out / 'chosen.json').write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2)); return
    chosen = a.factor
    out = OUTROOT / 'posthoc_scale' / f'f{chosen}' / a.model; out.mkdir(parents=True, exist_ok=False); (out / 'masks').mkdir(); (out / 'pdead').mkdir()
    model = load_model(a.model, device); rows = []
    for w in W:
        bfs, labs = frames(w['bf']), frames(w['lab'])
        if len(bfs) != 19 or len(labs) != 19: raise RuntimeError(f"xy{w['xy']} frames {len(bfs)}/{len(labs)}")
        stack = []
        for t, (raw, lab) in enumerate(zip(bfs, labs)):
            if t not in (0, 13, 18): continue
            pred, pdead = predict(model, stretch(raw), chosen, device); fg = (pred == 1) | (pred == 2); gt = lab > 0
            stack.append(pred)
            rows.append(dict(xy=w['xy'], dose=w['dose'], replicate=w['replicate'], frame=t, time_h=4 * t,
                             alive_px=int((pred == 1).sum()), dead_px=int((pred == 2).sum()), bubble_px=int((pred == 3).sum()),
                             fg_px=int(fg.sum()), dp_mean_fg=float(pdead[fg].mean()) if fg.any() else np.nan,
                             author_fg_px=int(gt.sum()), dice_vs_author=dice(fg, gt)))
            if t in SAVE_PROB_FRAMES:
                np.save(out / 'pdead' / f"xy{w['xy']:03d}_t{SAVE_PROB_FRAMES[t]}.npy", pdead.astype(np.float16))
        np.savez_compressed(out / 'masks' / f"xy{w['xy']:03d}.npz", pred=np.stack(stack).astype(np.uint8), frames=np.array([0, 13, 18]))
        print(f"xy{w['xy']} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out / 'per_frame.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(dict(model=a.model, checkpoint=str(MODELS[a.model][0]),
        checkpoint_sha256=MODELS[a.model][1], factor=chosen, n_frames=len(rows), script_sha256=sha256(__file__),
        torch=torch.__version__, elapsed_s=time.time() - t0), indent=2))


if __name__ == '__main__':
    main()
