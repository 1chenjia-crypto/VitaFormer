#!/bin/bash
# V100-1 GPU1 lane (2026-11-08 13:40): M2 fold4, then M0 folds 1-4 (AMENDMENT_M2_20261108.md §2/§4).
# M2 fold4 is also last in the V100-2 GPU0 lane; whichever starts second refuses the existing out dir (FileExistsError) = transferred, not a failure.
# M0 = unchanged 11-04 script (SHA 14a7bbad...), source M0 seed42 best_miou.pth.
D=${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/brain_organoid2024_m2_20261108
PY=${PYTHON:-python}; S=${WORKSPACE:-.}/3D_OrganoidSeg/scripts; G=1
echo "$(hostname) $(date) START m2 fold4 GPU$G"
CUDA_VISIBLE_DEVICES=$G $PY $S/train_brain_m2_20261108.py --model vitaformer --fold 4 --out $D/m2/fold4 > $D/logs/m2_fold4_v100-1.full.log 2>&1
echo "$(date) EXIT $? m2 fold4"
mkdir -p $D/m0
for F in 1 2 3 4; do
  echo "$(date) START m0 fold$F GPU$G"
  CUDA_VISIBLE_DEVICES=$G $PY $S/train_brain_strong_baseline_20261104.py --model vitaformer --fold $F --out $D/m0/fold$F > $D/logs/m0_fold$F.full.log 2>&1
  echo "$(date) EXIT $? m0 fold$F"
done
echo "$(date) LANE DONE"
