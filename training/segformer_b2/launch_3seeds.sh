#!/usr/bin/env bash
# P0-B SegFormer-B2: fixed 3 seeds (42, 3407, 2026), all reported. GPU 7. 42 and 3407 in parallel, 2026 after 42.
cd ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030
PY=${PYTHON:-python}
run() { CUDA_VISIBLE_DEVICES=7 $PY train_segformer_b2.py --seed $1 --out segformer_b2_runs/seed$1 --workers 5 > segformer_b2_runs/logs/seed$1.full.log 2>&1; }
( run 42; run 2026 ) &
( run 3407 ) &
wait
