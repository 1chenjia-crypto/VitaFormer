"""Per-seed OrganoidViT test results (shared evaluator) + selected checkpoint -> orgvit_3seed_summary.{csv,md}."""
import json, re
import numpy as np, pandas as pd
D = 'OrganoidSegBaseline/orgvit_3seed_20261111'
rec = open(D + '/SEED_RECORD.md').read()
rows = []
for s in (42, 3407, 2026):
    j = json.load(open(f'{D}/shared_eval/organoidvit_seed{s}.json'))
    it = re.findall(rf'seed={s} best_val_iter=(\d+) val_mIoU=([\d.]+)', rec)[-1]
    gpu = re.findall(rf'seed={s} host=\S+ gpu_index=(\d+)', rec)[-1]
    rows.append({'seed': s, 'best_val_iter': int(it[0]), 'val_mIoU': float(it[1]) * 100, 'test_mIoU': j['mIoU'],
                 'IoU_background': j['iou_background'], 'IoU_alive': j['iou_alive'], 'IoU_dead': j['iou_dead'],
                 'IoU_bubble': j['iou_bubble'], 'dead_recall': j['recall_dead'], 'dead_precision': j['precision_dead'],
                 'gpu': f'A800 GPU{gpu}'})
d = pd.DataFrame(rows)
num = d.select_dtypes('number').drop(columns=['seed'])
summ = pd.DataFrame({'seed': ['mean', 'SD']} | {c: [num[c].mean(), num[c].std(ddof=1)] for c in num})
out = pd.concat([d, summ], ignore_index=True)
out.to_csv(D + '/orgvit_3seed_summary.csv', index=False)
with open(D + '/orgvit_3seed_summary.md', 'w') as f:
    f.write('| seed | ckpt iter (best val) | val mIoU | test mIoU | Alive IoU | Dead IoU | Bubble IoU | Dead recall | Dead precision |\n|---|---|---|---|---|---|---|---|---|\n')
    for _, r in out.iterrows():
        it = '' if isinstance(r.seed, str) else f'{int(r.best_val_iter):,}'
        f.write(f'| {r.seed} | {it} | {r.val_mIoU:.2f} | {r.test_mIoU:.2f} | {r.IoU_alive:.2f} | {r.IoU_dead:.2f} | {r.IoU_bubble:.2f} | {r.dead_recall:.1f} | {r.dead_precision:.1f} |\n')
print(open(D + '/orgvit_3seed_summary.md').read())
