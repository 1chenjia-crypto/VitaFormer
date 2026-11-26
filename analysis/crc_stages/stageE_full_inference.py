#!/usr/bin/env python3
"""Amendment 5 (post hoc): Full VitaFormer final.pth (40k) whole-image VALIDATION inference, 3 seeds."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
EXP = Path('3D_OrganoidSeg/vco/runs/arch_exploration_20261104'); RUNS = EXP.parent
sys.path.insert(0, str(EXP / 'scripts')); sys.path.insert(0, '3D_OrganoidSeg/vco/organoid_seg')
from a0_m0_val_inference import infer_hard, sha256, IMG_DIR, VAL, TEST, BCLIP  # noqa
from model_v2 import OrganoidSegV2  # noqa
RUN = {42: 'expFinal_fromscratch', 3407: 'full_model_seed3407', 2026: 'full_model_seed2026'}
OUT = EXP / 'stageE_full_40k/final_masks'
def load(ck, dev):
    m = OrganoidSegV2(weight_path=str(BCLIP), num_classes=4, freeze_blocks=0, use_mpm=True, use_bam=True)
    st = torch.load(ck, map_location='cpu', weights_only=True); m.load_state_dict(st.get('model_state', st), strict=True)
    return m.eval().to(dev), int(st.get('step', -1))
def main():
    dev = torch.device('cuda'); OUT.mkdir(parents=True, exist_ok=False)
    val = [s.strip() for s in VAL.read_text().split() if s.strip()]; test = [s.strip() for s in TEST.read_text().split() if s.strip()]
    man = {}
    for s, r in RUN.items():
        mb, sb = load(RUNS / r / 'best_miou.pth', dev)
        agree = [float((infer_hard(mb, np.asarray(Image.open(IMG_DIR / f'{t}.jpg').convert('RGB')), dev) ==
                        np.asarray(Image.open(RUNS / f'module_endpoints_multiseed_20261030/full_seed{s}/masks/{t}.png'))).mean()) for t in test[:2]]
        del mb; torch.cuda.empty_cache()
        mf, step = load(RUNS / r / 'final.pth', dev); assert step == 40000, (r, step)
        d = OUT / f'Full_seed{s}'; d.mkdir(); t0 = time.time()
        for st_ in val:
            Image.fromarray(infer_hard(mf, np.asarray(Image.open(IMG_DIR / f'{st_}.jpg').convert('RGB')), dev)).save(d / f'{st_}.png')
        man[s] = {'run': r, 'final_sha256': sha256(RUNS / r / 'final.pth'), 'best_step_field': sb, 'test2_consistency_best': agree, 'seconds': round(time.time() - t0)}
        print(s, man[s], flush=True); del mf; torch.cuda.empty_cache()
    (OUT.parent / 'manifest.json').write_text(json.dumps(man, indent=2)); print('done', flush=True)
if __name__ == '__main__':
    main()
