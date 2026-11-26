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
assert len(vals) == 20, 'expected 20 validation records (2k..40k), got %d' % len(vals)
best = max(vals, key=lambda x: x[1])
print(best[0], best[1])
