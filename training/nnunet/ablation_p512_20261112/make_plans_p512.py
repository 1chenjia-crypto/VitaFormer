#!/usr/bin/env python3
"""Create nnUNetPlans_p512.json from nnUNetPlans.json. Changes ONLY 2d patch_size -> [512,512] and batch_size -> 10
(pixels per iteration 2*1024*1280 = 2.62M vs 10*512*512 = 2.62M). Network, data_identifier (reuses existing
preprocessed data), normalisation, everything else identical. Refuses to overwrite."""
import json, copy, sys
from pathlib import Path
P = Path('3D_OrganoidSeg/baselines_fair_20261030/nnunet/nnUNet_preprocessed/Dataset801_CRCOrganoid')
src = json.load(open(P / 'nnUNetPlans.json'))
out = P / 'nnUNetPlans_p512.json'
if out.exists(): sys.exit(f'REFUSE: {out} exists')
new = copy.deepcopy(src)
new['plans_name'] = 'nnUNetPlans_p512'
c = new['configurations']['2d']
assert c['patch_size'] == [1024, 1280] and c['batch_size'] == 2
c['patch_size'] = [512, 512]; c['batch_size'] = 10
json.dump(new, open(out, 'w'), indent=4)
diff = [k for k in c if c[k] != src['configurations']['2d'][k]]
print('written', out, 'changed keys:', diff)
