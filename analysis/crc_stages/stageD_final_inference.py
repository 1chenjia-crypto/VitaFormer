#!/usr/bin/env python3
"""Amendment 4 (post hoc): whole-image validation inference of final.pth (step 40000) for M0/M1/M2 x 3 seeds."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
EXP = Path('3D_OrganoidSeg/vco/runs/arch_exploration_20261104')
RUNS = Path('3D_OrganoidSeg/vco/runs')
sys.path.insert(0, str(EXP / 'scripts')); sys.path.insert(0, '3D_OrganoidSeg/vco/organoid_seg')
from a0_m0_val_inference import infer_hard, sha256, IMG_DIR, VAL, BCLIP  # noqa
from model_v2 import OrganoidSegV2  # noqa
from model_v3_multiscale import OrganoidSegMultiScale  # noqa
OUT = EXP / 'stageD_selection_free/final_masks'
def run_dir(m, s):
    if m == 'M0': return RUNS / f'ablation_no_bam_no_mpm_seed{s}'
    if m == 'M2' and s == 42: return EXP / 'stageB_m2_seed42'
    return EXP / f'stageC_{m.lower()}_seed{s}'
def best_mask_dir(m, s):
    if m == 'M0': return EXP / f'A0_gap_decomposition/m0_val_masks/seed{s}'
    if m == 'M2' and s == 42: return EXP / 'stageB_m2_seed42_eval/masks'
    return EXP / f'stageC_{m.lower()}_seed{s}_eval/masks'
def build(m):
    if m == 'M0': return OrganoidSegV2(weight_path=str(BCLIP), num_classes=4, freeze_blocks=0, use_mpm=False, use_bam=False)
    return OrganoidSegMultiScale(str(BCLIP), num_classes=4, freeze_blocks=0, zero_cnn_input=(m == 'M1'))
def load(model, ck, dev):
    st = torch.load(ck, map_location='cpu', weights_only=True)
    model.load_state_dict(st.get('model_state', st), strict=True)
    return model.eval().to(dev), int(st.get('step', -1))
def main():
    dev = torch.device('cuda'); OUT.mkdir(parents=True, exist_ok=False)
    stems = [s.strip() for s in VAL.read_text().splitlines() if s.strip()]
    man = {'protocol': 'infer_hard (a0_m0_val_inference), final.pth', 'runs': {}}
    for m in ('M0', 'M1', 'M2'):
        for s in (42, 3407, 2026):
            rd = run_dir(m, s)
            mb, stepb = load(build(m), rd / 'best_miou.pth', dev)
            agree = []
            for st_ in stems[:2]:
                img = np.asarray(Image.open(IMG_DIR / f'{st_}.jpg').convert('RGB'))
                agree.append(float((infer_hard(mb, img, dev) == np.asarray(Image.open(best_mask_dir(m, s) / f'{st_}.png'))).mean()))
            del mb; torch.cuda.empty_cache()
            mf, step = load(build(m), rd / 'final.pth', dev)
            assert step == 40000, (m, s, step)
            d = OUT / f'{m}_seed{s}'; d.mkdir(); t0 = time.time()
            for i, st_ in enumerate(stems, 1):
                img = np.asarray(Image.open(IMG_DIR / f'{st_}.jpg').convert('RGB'))
                Image.fromarray(infer_hard(mf, img, dev)).save(d / f'{st_}.png')
            man['runs'][f'{m}_seed{s}'] = {'final': str(rd / 'final.pth'), 'final_sha256': sha256(rd / 'final.pth'), 'final_step': step,
                                          'best_step': stepb, 'best_reinfer_agreement_2val': agree, 'seconds': round(time.time() - t0)}
            print(f'{m} seed{s}: best step {stepb}, best-consistency {agree}, final done {time.time()-t0:.0f}s', flush=True)
            del mf; torch.cuda.empty_cache()
    (OUT.parent / 'final_inference_manifest.json').write_text(json.dumps(man, indent=2)); print('done', flush=True)
if __name__ == '__main__':
    main()
