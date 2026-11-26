#!/usr/bin/env python
"""Best validation mIoU checkpoint from mmseg *.log.json (same rule as watch_orgvit_b0_1446_40k_cond_eval.sh)."""
import glob, json, sys
run = sys.argv[1]; vals = []
for p in sorted(glob.glob(run + '/*.log.json')):
    for line in open(p):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get('mode') == 'val' and 'mIoU' in o:
            vals.append((int(o['iter']), float(o['mIoU'])))
vals = sorted(dict(vals).items())  # dedupe by iteration (resumed runs write a second log.json)
missing = sorted(set(range(2000, 40001, 2000)) - set(i for i, _ in vals))
if missing:
    sys.stderr.write('WARNING missing validation iters (interrupted around a checkpoint): %s\n' % missing)
assert max(i for i, _ in vals) == 40000 and len(vals) >= 18, 'incomplete validation history: %s' % vals
best = max(vals, key=lambda x: x[1])
print(best[0], best[1])
