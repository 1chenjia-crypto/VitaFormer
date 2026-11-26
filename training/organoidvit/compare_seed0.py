"""Env check criteria (PLAN.md): tools/test.py mIoU 75.30 +- 0.05; shared-eval areas vs existing CSV <= 0.1% px per image."""
import json, re
import numpy as np, pandas as pd
import sys
D = 'OrganoidSegBaseline/orgvit_3seed_20261111/' + (sys.argv[1] if len(sys.argv) > 1 else 'env_check')
log = open(D + '/tools_test_seed0.full.log').read()
m = re.findall(r'\|\s*global\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', log) or re.findall(r'mIoU:\s*([\d.]+)', log)
print('tools/test.py summary match:', m[-1] if m else 'NOT FOUND')
j = json.load(open(D + '/shared_eval_seed0.json'))
print('shared evaluator mIoU %.3f  alive %.2f dead %.2f bubble %.2f' % (j['mIoU'], j['iou_alive'], j['iou_dead'], j['iou_bubble']))
ref = pd.read_csv('OrganoidSegBaseline/runs/orgvit_b0_1446_40k_cond/test_best_iter_32000_relative_area_errors.csv').set_index('image_id')
worst = 0.0; nd = 0
for p in j['per_image']:
    c = p['confusion_counts']; pred = np.array(c['tp']) + np.array(c['fp']); gt = np.array(c['tp']) + np.array(c['fn'])
    r = ref.loc[p['stem']]
    rp = np.array([r.pred_background_area, r.pred_live_area, r.pred_death_area, r.pred_bubble_area])
    rg = np.array([r.gt_background_area, r.gt_live_area, r.gt_death_area, r.gt_bubble_area])
    assert (gt == rg).all(), p['stem']
    frac = np.abs(pred - rp).sum() / 2 / gt.sum()
    worst = max(worst, frac); nd += int((pred != rp).any())
print('images with any area difference: %d / %d; worst fraction of pixels changed: %.6f' % (nd, len(j['per_image']), worst))
print('AREA_CHECK', 'PASS' if worst <= 0.001 else 'FAIL')
