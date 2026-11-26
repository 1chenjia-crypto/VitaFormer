#!/usr/bin/env python3
"""Build M2 freeze manifest (2026-11-08): SHA256 of checkpoints, code, configs, data splits; ckpt step check."""
import hashlib, json, sys, subprocess, platform
from pathlib import Path
import torch
R = Path('3D_OrganoidSeg'); E = R / 'vco/runs/arch_exploration_20261104'
D = Path('OrganoidSegBaseline/VOC2012/ImageSets/Segmentation')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
ck = {42: E/'stageB_m2_seed42/final.pth', 3407: E/'stageC_m2_seed3407/final.pth', 2026: E/'stageC_m2_seed2026/final.pth'}
out = {'checkpoints': {}, 'code': {}, 'configs': {}, 'data': {}, 'pretrained': {}}
for s, p in ck.items():
    st = torch.load(p, map_location='cpu', weights_only=True)
    keys = list(st.keys())[:20] if isinstance(st, dict) else []
    sd = st.get('model_state', st)
    nonfinite = sum(int((~torch.isfinite(v.float())).sum()) for v in sd.values() if torch.is_tensor(v) and v.is_floating_point())
    out['checkpoints'][s] = {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size,
        'step': int(st.get('step', -1)) if isinstance(st, dict) else None, 'top_keys': [k for k in keys if k != 'model_state' or True][:10],
        'n_tensors': len(sd), 'nonfinite_values': nonfinite}
code = [R/'vco/organoid_seg/model_v2.py', R/'vco/organoid_seg/model_v3_multiscale.py', R/'vco/organoid_seg/train_v3_multiscale.py',
        R/'vco/organoid_seg/train_v2_stratified.py', R/'vco/organoid_seg/dataset_v2.py',
        E/'scripts/a0_m0_val_inference.py', E/'scripts/stageD_final_inference.py', E/'scripts/stageF_test_inference.py',
        E/'scripts/stageF_comparison.py', R/'baselines_fair_20261030/eval_masks_shared.py', R/'scripts/unified_comparison_20261102.py']
for p in code: out['code'][str(p)] = sha(p) if p.exists() else 'MISSING'
cfg = [E/'stageB_m2_seed42/launch_command.sh', E/'stageC_m2_seed3407/launch_command.sh', E/'stageC_m2_seed2026/launch_command.sh',
       E/'stageB_m2_seed42/summary.json', E/'stageC_m2_seed3407/summary.json', E/'stageC_m2_seed2026/summary.json',
       E/'ARCHITECTURE_EXPLORATION_PLAN.md', E/'AMENDMENT_4_SELECTION_FREE_20261106.md', E/'stageF_m2_test/ANALYSIS_PLAN.md']
for p in cfg: out['configs'][str(p)] = sha(p) if p.exists() else 'MISSING'
for n in ['train_cond.txt', 'val_cond.txt', 'test_cond.txt']:
    p = D/n; out['data'][str(p)] = {'sha256': sha(p), 'n': len([x for x in p.read_text().splitlines() if x.strip()])} if p.exists() else 'MISSING'
b = Path('pretrained/biomedclip/open_clip_pytorch_model.bin'); out['pretrained'][str(b)] = sha(b)
out['env'] = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'platform': platform.platform()}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps({s: {k: v[k] for k in ('sha256', 'step', 'nonfinite_values')} for s, v in out['checkpoints'].items()}, indent=1))
print('MISSING:', [k for d in ('code', 'configs', 'data') for k, v in out[d].items() if v == 'MISSING'])
