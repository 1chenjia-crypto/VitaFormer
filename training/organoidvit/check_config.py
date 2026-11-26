"""Merged config (source file) must equal the config dumped by the seed-0 run, except work_dir/gpu_ids."""
import sys
from mmcv import Config
a = Config.fromfile('OrganoidSegBaseline/Orgvit/local_configs/2DSegFormer/organoid_voc_b0_1446_40k_cond.py')
b = Config.fromfile('OrganoidSegBaseline/runs/orgvit_b0_1446_40k_cond/organoid_voc_b0_1446_40k_cond.py')
da, db = a._cfg_dict.to_dict(), b._cfg_dict.to_dict()
for k in ('work_dir', 'gpu_ids'):
    da.pop(k, None); db.pop(k, None)
ok = da == db
print('CONFIG_MATCH', ok)
if not ok:
    for k in sorted(set(da) | set(db)):
        if da.get(k) != db.get(k):
            print('DIFF', k, da.get(k), db.get(k))
sys.exit(0 if ok else 1)
