#!/bin/bash
# V100-2 queue v2 (2026-11-08 13:47). Same commands as run_m2_queue.sh; ONLY change: OMP/MKL threads = 4
# (validation preprocessing was CPU-oversubscribed: 10.8 s vs 0.14 s per image). Interrupted fold0/1 (stopped at step-4000 validation)
# are kept untouched in m2/fold0, m2/fold1; reruns go to m2_rerun/fold{0,1}. Folds 2/3 had not started -> m2/fold{2,3}.
# Fold 4 runs on V100-1 (run_v100_1_queue.sh).
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/brain_organoid2024_m2_20261108
PY=${PYTHON:-python}; S=${WORKSPACE:-.}/3D_OrganoidSeg/scripts/train_brain_m2_20261108.py
lane() { G=$1; shift
  for J in "$@"; do F=${J#*:}; O=$D/${J%%:*}/fold$F
    echo "$(hostname) $(date) START $O GPU$G"
    CUDA_VISIBLE_DEVICES=$G $PY $S --model vitaformer --fold $F --out $O > $D/logs/$(basename $(dirname $O))_fold${F}.full.log 2>&1
    echo "$(date) EXIT $? $O"
  done; }
mkdir -p $D/m2_rerun
lane 0 m2_rerun:0 m2:2 & lane 1 m2_rerun:1 m2:3 & wait
echo "$(date) ALL DONE"
