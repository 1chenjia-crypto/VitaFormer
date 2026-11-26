#!/usr/bin/env python3
"""GemcitabineScreen Amendment 2: SegFormer-B2 / nnU-Net inference with the same input, scale rule and outputs as
gem_pi_infer_20261109.py (M2). Models from nvir_infer_20261109 (hash-checked). Never opens PI files."""
from pathlib import Path
import argparse, json, sys, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from PIL import Image
ROOT = Path('3D_OrganoidSeg')
sys.path.insert(0, str(ROOT / 'scripts'))
import nvir_infer_20261109 as NV  # noqa
import gem_pi_infer_20261109 as GM  # noqa  (wells, frames, stretch, dice, FACTORS, SAVE_PROB_FRAMES, OUTROOT)
OUT = GM.OUTROOT


def load(family, seed, dev):
    if family == 'segformer':
        sys.path.insert(0, str(ROOT / 'baselines_fair_20261030'))
        import train_segformer_b2 as B
        ck = NV.CKPT['segformer'][seed]
        if NV.sha256(ck) != NV.SHA[('segformer', seed)]: raise RuntimeError('hash mismatch')
        m, _ = B.build_model()
        m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True)['model'], strict=True); m.eval().to(dev)

        def prob(img):  # train_segformer_b2.predict, returning the averaged softmax instead of its argmax
            h, w = img.shape[:2]; ph, pw = max(B.WIN - h, 0), max(B.WIN - w, 0)
            x = B.norm(np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect')); H, W = x.shape[1:]
            ys = list(range(0, H - B.WIN + 1, B.STRIDE)); xs = list(range(0, W - B.WIN + 1, B.STRIDE))
            if ys[-1] + B.WIN < H: ys.append(H - B.WIN)
            if xs[-1] + B.WIN < W: xs.append(W - B.WIN)
            score = torch.zeros((B.NC, H, W), device=dev); cnt = torch.zeros((H, W), device=dev)
            cc_all = [(y, xx) for y in ys for xx in xs]
            with torch.no_grad():
                for b in range(0, len(cc_all), 16):
                    cc = cc_all[b:b + 16]
                    batch = torch.stack([x[:, y:y + B.WIN, xx:xx + B.WIN] for y, xx in cc]).to(dev)
                    with torch.amp.autocast('cuda'):
                        lg = m(pixel_values=batch).logits
                    pr = F.softmax(F.interpolate(lg.float(), size=(B.WIN, B.WIN), mode='bilinear', align_corners=False), 1)
                    for j, (y, xx) in enumerate(cc):
                        score[:, y:y + B.WIN, xx:xx + B.WIN] += pr[j]; cnt[y:y + B.WIN, xx:xx + B.WIN] += 1
            return (score / cnt.clamp_min(1))[:, :h, :w].cpu().numpy()
        return prob, str(ck)
    if family == 'nnunet':
        import run_nnunet_deeporganoid_20261102 as nn
        p, ck = nn.make_predictor(seed)
        return (lambda img: nn.predict(p, img)[1]), str(ck)
    raise ValueError(family)


def predict(prob_fn, img, factor):
    h, w = img.shape[:2]
    if factor != 1.0:
        small = np.asarray(Image.fromarray(img).resize((round(w * factor), round(h * factor)), Image.BILINEAR))
        pr = prob_fn(np.ascontiguousarray(small))
        pred = np.asarray(Image.fromarray(pr.argmax(0).astype(np.uint8)).resize((w, h), Image.NEAREST))
        pdead = np.asarray(Image.fromarray(pr[2].astype(np.float32)).resize((w, h), Image.BILINEAR))
    else:
        pr = prob_fn(np.ascontiguousarray(img)); pred = pr.argmax(0).astype(np.uint8); pdead = pr[2]
    return pred, pdead.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--family', choices=['segformer', 'nnunet'], required=True)
    ap.add_argument('--stage', choices=['scale_select', 'full'], required=True); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(); dev = torch.device('cuda'); t0 = time.time(); W = GM.wells()
    sel_dir = OUT / f'scale_select_{a.family}'
    if a.stage == 'scale_select':
        if a.seed != 42: raise ValueError('scale selection uses seed 42 only')
        sel_dir.mkdir(exist_ok=False); fn, ck = load(a.family, 42, dev); rows = []
        for w in W:
            img = GM.stretch(GM.frames(w['bf'])[0]); gt = GM.frames(w['lab'])[0] > 0
            for f in GM.FACTORS:
                pred, _ = predict(fn, img, f); fg = (pred == 1) | (pred == 2)
                rows.append(dict(xy=w['xy'], factor=f, dice=GM.dice(fg, gt), fg_px=int(fg.sum()), gt_px=int(gt.sum()),
                                 dead_px=int((pred == 2).sum())))
            print(f"xy{w['xy']} {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(rows); df.to_csv(sel_dir / 'scale_select_t0.csv', index=False)
        s = df.groupby('factor')['dice'].mean().sort_index(ascending=False); best = s.max()
        chosen = float([f for f in s.index if s[f] >= best - 0.005][0])
        res = dict(family=a.family, mean_dice_by_factor={str(k): float(v) for k, v in s.items()}, chosen_factor=chosen,
                   best_mean_dice=float(best), segmentation_failed=bool(s[chosen] < 0.5), checkpoint=ck,
                   script_sha256=NV.sha256(__file__), elapsed_s=time.time() - t0)
        (sel_dir / 'chosen.json').write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2)); return
    chosen = json.loads((sel_dir / 'chosen.json').read_text())['chosen_factor']
    out = OUT / f'{a.family}_seed{a.seed}'; out.mkdir(exist_ok=False); (out / 'masks').mkdir(); (out / 'pdead').mkdir()
    fn, ck = load(a.family, a.seed, dev); rows = []
    for w in W:
        bfs, labs = GM.frames(w['bf']), GM.frames(w['lab'])
        if len(bfs) != 19 or len(labs) != 19: raise RuntimeError(f"xy{w['xy']} frames")
        stack = []
        for t, (raw, lab) in enumerate(zip(bfs, labs)):
            pred, pdead = predict(fn, GM.stretch(raw), chosen); fg = (pred == 1) | (pred == 2); gt = lab > 0
            stack.append(pred)
            rows.append(dict(xy=w['xy'], dose=w['dose'], replicate=w['replicate'], frame=t, time_h=4 * t,
                             alive_px=int((pred == 1).sum()), dead_px=int((pred == 2).sum()), bubble_px=int((pred == 3).sum()),
                             fg_px=int(fg.sum()), dp_mean_fg=float(pdead[fg].mean()) if fg.any() else np.nan,
                             author_fg_px=int(gt.sum()), dice_vs_author=GM.dice(fg, gt)))
            if t in GM.SAVE_PROB_FRAMES:
                np.save(out / 'pdead' / f"xy{w['xy']:03d}_t{GM.SAVE_PROB_FRAMES[t]}.npy", pdead.astype(np.float16))
        np.savez_compressed(out / 'masks' / f"xy{w['xy']:03d}.npz", pred=np.stack(stack).astype(np.uint8))
        print(f"xy{w['xy']} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out / 'per_frame.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(dict(family=a.family, seed=a.seed, checkpoint=ck, factor=chosen,
        n_frames=len(rows), script_sha256=NV.sha256(__file__), torch=torch.__version__, elapsed_s=time.time() - t0), indent=2))


if __name__ == '__main__':
    main()
