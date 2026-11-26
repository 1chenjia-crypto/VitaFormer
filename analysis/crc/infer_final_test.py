#!/usr/bin/env python3
"""M2 ablations (noRD / ImageNet), final.pth (step 40000), whole-image TEST inference (descriptive only; TEST list imported as VAL so the loop is unchanged).
Same protocol as stageD_final_inference.py (infer_hard from a0_m0_val_inference; M2 architecture OrganoidSegMultiScale,
zero_cnn_input=False). Weights come entirely from final.pth (strict load), so the encoder init source is irrelevant here.
Usage: infer_final_val.py ARM SEED   (ARM = noRD | ImageNet). Refuses an existing output dir."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
EXP = Path('3D_OrganoidSeg/vco/runs/arch_exploration_20261104'); RUNS = Path('3D_OrganoidSeg/vco/runs')
sys.path.insert(0, str(EXP / 'scripts')); sys.path.insert(0, '3D_OrganoidSeg/vco/organoid_seg')
from a0_m0_val_inference import infer_hard, sha256, IMG_DIR, TEST as VAL, BCLIP  # noqa
from model_v3_multiscale import OrganoidSegMultiScale  # noqa
OUT = Path(__file__).resolve().parent / 'test_masks'
RD = {'noRD': RUNS / 'm2_ablation_no_rare_dead_20261112/m2_noRD_seed{}', 'ImageNet': RUNS / 'm2_ablation_imagenet_20261112/m2_imagenet_seed{}'}
arm, s = sys.argv[1], int(sys.argv[2]); rd = Path(str(RD[arm]).format(s))
dev = torch.device('cuda'); d = OUT / f'{arm}_seed{s}'; d.mkdir(parents=True, exist_ok=False)
st = torch.load(rd / 'final.pth', map_location='cpu', weights_only=True)
m = OrganoidSegMultiScale(str(BCLIP), num_classes=4, freeze_blocks=0, zero_cnn_input=False)
m.load_state_dict(st.get('model_state', st), strict=True); m = m.eval().to(dev); step = int(st.get('step', -1))
assert step == 40000, step
stems = [x.strip() for x in VAL.read_text().splitlines() if x.strip()]; t0 = time.time()
for st_ in stems:
    img = np.asarray(Image.open(IMG_DIR / f'{st_}.jpg').convert('RGB'))
    Image.fromarray(infer_hard(m, img, dev)).save(d / f'{st_}.png')
man = {'arm': arm, 'seed': s, 'final': str(rd / 'final.pth'), 'final_sha256': sha256(rd / 'final.pth'), 'step': step,
       'n_test': len(stems), 'seconds': round(time.time() - t0), 'protocol': 'infer_hard (a0_m0_val_inference), final.pth'}
(d.parent / f'{arm}_seed{s}_manifest.json').write_text(json.dumps(man, indent=2)); print(man, flush=True)
