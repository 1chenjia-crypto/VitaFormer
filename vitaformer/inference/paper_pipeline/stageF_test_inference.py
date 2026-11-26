#!/usr/bin/env python3
"""Stage F (stageF_m2_test/ANALYSIS_PLAN.md): whole-image CRC TEST inference of final.pth (step 40000) for one model/seed.
Usage: stageF_test_inference.py {M0|M2} SEED
Protocol identical to stageD_final_inference.py (infer_hard from a0_m0_val_inference, unchanged). Outputs: new dir only.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
EXP = Path('3D_OrganoidSeg/vco/runs/arch_exploration_20261104')
RUNS = Path('3D_OrganoidSeg/vco/runs')
sys.path.insert(0, str(EXP / 'scripts')); sys.path.insert(0, '3D_OrganoidSeg/vco/organoid_seg')
from a0_m0_val_inference import infer_hard, sha256, IMG_DIR, VAL, TEST, P0M, BCLIP  # noqa
from model_v2 import OrganoidSegV2  # noqa
from model_v3_multiscale import OrganoidSegMultiScale  # noqa
OUT = EXP / 'stageF_m2_test/test_masks'
STAGED = EXP / 'stageD_selection_free/final_masks'


def run_dir(m, s):
    if m == 'M0': return RUNS / f'ablation_no_bam_no_mpm_seed{s}'
    if s == 42: return EXP / 'stageB_m2_seed42'
    return EXP / f'stageC_m2_seed{s}'


def build(m):
    if m == 'M0': return OrganoidSegV2(weight_path=str(BCLIP), num_classes=4, freeze_blocks=0, use_mpm=False, use_bam=False)
    return OrganoidSegMultiScale(str(BCLIP), num_classes=4, freeze_blocks=0, zero_cnn_input=False)


def load(model, ck, dev):
    st = torch.load(ck, map_location='cpu', weights_only=True)
    model.load_state_dict(st.get('model_state', st), strict=True)
    return model.eval().to(dev), int(st.get('step', -1))


def agree(model, stems, ref_dir, dev):
    out = []
    for s in stems:
        img = np.asarray(Image.open(IMG_DIR / f'{s}.jpg').convert('RGB'))
        out.append(float((infer_hard(model, img, dev) == np.asarray(Image.open(ref_dir / f'{s}.png'))).mean()))
    return out


def main():
    m, s = sys.argv[1], int(sys.argv[2]); assert m in ('M0', 'M2')
    dev = torch.device('cuda')
    test = [x.strip() for x in TEST.read_text().splitlines() if x.strip()]
    val = [x.strip() for x in VAL.read_text().splitlines() if x.strip()]
    assert len(test) == 167, len(test)
    rd = run_dir(m, s)
    checks = {}
    if m == 'M0':  # pipeline check: best_miou re-predicts 2 test images vs existing P0-M masks
        mb, _ = load(build(m), rd / 'best_miou.pth', dev)
        checks['best_miou_vs_P0M_test2'] = agree(mb, test[:2], P0M / f'simplified_seed{s}/masks', dev)
        del mb; torch.cuda.empty_cache()
    mf, step = load(build(m), rd / 'final.pth', dev)
    assert step == 40000, (m, s, step)
    checks['final_vs_stageD_val2'] = agree(mf, val[:2], STAGED / f'{m}_seed{s}', dev)
    print(f'{m} seed{s} checks: {checks}', flush=True)
    assert all(v > 0.9999 for vs in checks.values() for v in vs), checks
    d = OUT / f'{m}_seed{s}'; d.mkdir(parents=True, exist_ok=False); t0 = time.time()
    for i, st_ in enumerate(test, 1):
        img = np.asarray(Image.open(IMG_DIR / f'{st_}.jpg').convert('RGB'))
        Image.fromarray(infer_hard(mf, img, dev)).save(d / f'{st_}.png')
        if i % 40 == 0: print(f'{m} seed{s}: {i}/167 {time.time()-t0:.0f}s', flush=True)
    man = {'protocol': 'infer_hard (a0_m0_val_inference), final.pth, CRC test', 'model': m, 'seed': s,
           'final': str(rd / 'final.pth'), 'final_sha256': sha256(rd / 'final.pth'), 'final_step': step,
           'test_split_sha256': sha256(TEST), 'n_test': len(test), 'checks': checks, 'seconds': round(time.time() - t0)}
    (OUT / f'{m}_seed{s}_manifest.json').write_text(json.dumps(man, indent=2)); print('done', flush=True)


if __name__ == '__main__':
    main()
