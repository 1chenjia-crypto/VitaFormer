#!/usr/bin/env python3
"""Stage G (AMENDMENT_1_20261106.md): VitaFormer validation probability export for one arm/seed.
Usage: stageG_export_vf.py {M2|M0|Full|FullnoRD} SEED
Inference = infer_hard (a0_m0_val_inference) unchanged up to the averaged softmax; argmax replaced by returning the
probabilities. Checks (§4): delta=0 decision vs existing frozen masks; best_miou.pth on 2 test images vs saved test counts.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
EXP = Path('3D_OrganoidSeg/vco/runs/arch_exploration_20261104'); RUNS = EXP.parent
sys.path.insert(0, str(EXP / 'scripts')); sys.path.insert(0, '3D_OrganoidSeg/vco/organoid_seg')
import a0_m0_val_inference as A  # noqa
import stageG_common as C  # noqa
from model_v2 import OrganoidSegV2  # noqa
from model_v3_multiscale import OrganoidSegMultiScale  # noqa
sys.path.insert(0, '3D_OrganoidSeg/baselines_fair_20261030')
import eval_masks_shared as EV  # noqa
OUT = EXP / 'stageG_dead_pr_val/export'
FULL = {42: 'expFinal_fromscratch', 3407: 'full_model_seed3407', 2026: 'full_model_seed2026'}


def run_dir(m, s):
    if m == 'M0': return RUNS / f'ablation_no_bam_no_mpm_seed{s}'
    if m == 'M2': return EXP / ('stageB_m2_seed42' if s == 42 else f'stageC_m2_seed{s}')
    if m == 'Full': return RUNS / FULL[s]
    return RUNS / f'ablation_no_rare_dead_seed{s}'


def build(m):
    if m == 'M2': return OrganoidSegMultiScale(str(A.BCLIP), num_classes=4, freeze_blocks=0, zero_cnn_input=False)
    full = m in ('Full', 'FullnoRD')
    return OrganoidSegV2(weight_path=str(A.BCLIP), num_classes=4, freeze_blocks=0, use_mpm=full, use_bam=full)


def load(m, ck, dev):
    st = torch.load(ck, map_location='cpu', weights_only=True)
    model = build(m); model.load_state_dict(st.get('model_state', st), strict=True)
    return model.eval().to(dev), int(st.get('step', -1))


@torch.inference_mode()
def infer_prob(model, image, device):  # = A.infer_hard without the final argmax
    h, w = image.shape[:2]
    coords, pad_h, pad_w = A.patch_coords(h, w)
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1)
    tensor = (tensor - A.MEAN) / A.STD
    patches = torch.stack([tensor[:, y:y + A.WINDOW, x:x + A.WINDOW] for y, x in coords])
    score = torch.zeros((4, h, w), dtype=torch.float32, device=device)
    count = torch.zeros((h, w), dtype=torch.float32, device=device)
    for start in range(0, len(coords), A.BATCH):
        cc = coords[start:start + A.BATCH]
        batch = patches[start:start + A.BATCH].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            logits = model(batch)
            logits = logits[0] if isinstance(logits, tuple) else logits
        probs = F.softmax(logits.float(), dim=1)
        for j, (y, x) in enumerate(cc):
            y2, x2 = min(y + A.WINDOW, h), min(x + A.WINDOW, w)
            score[:, y:y2, x:x2] += probs[j, :, :y2 - y, :x2 - x]
            count[y:y2, x:x2] += 1
    prob = score / count.clamp_min(1e-6).unsqueeze(0)
    return prob.cpu().numpy(), prob.argmax(0).byte().cpu().numpy()


def ref_masks(m, s):
    if m in ('M0', 'M2'): return EXP / f'stageD_selection_free/final_masks/{m}_seed{s}'
    if m == 'Full' and s == 42: return EXP / 'stageE_full_40k/final_masks/Full_seed42'
    return None


def main():
    m, s = sys.argv[1], int(sys.argv[2]); dev = torch.device('cuda')
    val = [x.strip() for x in A.VAL.read_text().splitlines() if x.strip()]
    test = [x.strip() for x in A.TEST.read_text().splitlines() if x.strip()]
    rd = run_dir(m, s); d = OUT / f'{m}_seed{s}'
    (d / 'level').mkdir(parents=True, exist_ok=False); (d / 'other').mkdir()
    checks = {}
    # pipeline check on 2 test images: best_miou reproduces saved per-image test confusion counts (M2 has none)
    tj = rd / 'test_best_miou_uniform.json'
    if tj.exists():
        ref = {x['stem']: x['confusion_counts'] for x in json.loads(tj.read_text())['per_image']}
        mb, _ = load(m, rd / 'best_miou.pth', dev)
        ok = []
        for t in test[:2]:
            pred = A.infer_hard(mb, np.asarray(Image.open(A.IMG_DIR / f'{t}.jpg').convert('RGB')), dev)
            c = EV.counts(pred, C.load_gt(t))
            ok.append(int(sum(abs(a - b) for k in ('tp', 'fp', 'fn') for a, b in zip(c[k], ref[t][k]))))
        checks['best_miou_test2_count_absdiff'] = ok  # 0 = identical; reported, not a stop rule
        del mb; torch.cuda.empty_cache()
    model, step = load(m, rd / 'final.pth', dev); assert step == 40000, (m, s, step)
    rdir = ref_masks(m, s)
    stats = {'H': [], 'HF': [], 'HP': []}; mism = {'argmax_vs_delta0_px': 0, 'ref_vs_delta0_px': 0, 'ref_images': 0}
    t0 = time.time()
    for i, st in enumerate(val, 1):
        img = np.asarray(Image.open(A.IMG_DIR / f'{st}.jpg').convert('RGB'))
        prob, am = infer_prob(model, img, dev)
        sc, oth = C.score_from_probs(prob)
        d0 = C.decide(sc, oth, 0.0)
        mism['argmax_vs_delta0_px'] += int((d0 != am).sum())
        if rdir is not None and (rdir / f'{st}.png').exists():
            mism['ref_vs_delta0_px'] += int((d0 != np.asarray(Image.open(rdir / f'{st}.png'))).sum()); mism['ref_images'] += 1
        res, level = C.summarize(sc, oth, prob[2], C.load_gt(st))
        for k in stats: stats[k].append(res[k])
        Image.fromarray(level).save(d / 'level' / f'{st}.png'); Image.fromarray(oth).save(d / 'other' / f'{st}.png')
        if i % 40 == 0: print(f'{m} s{s}: {i}/{len(val)} {time.time()-t0:.0f}s {mism}', flush=True)
    np.savez_compressed(d / 'stats.npz', stems=np.array(val), **{k: np.stack(v) for k, v in stats.items()})
    checks['delta0'] = mism
    man = {'arm': m, 'seed': s, 'final': str(rd / 'final.pth'), 'final_sha256': A.sha256(rd / 'final.pth'), 'step': step,
           'val_split_sha256': A.sha256(A.VAL), 'n_val': len(val), 'ref_masks': str(rdir), 'checks': checks,
           'seconds': round(time.time() - t0)}
    (d / 'manifest.json').write_text(json.dumps(man, indent=2)); print('done', json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
