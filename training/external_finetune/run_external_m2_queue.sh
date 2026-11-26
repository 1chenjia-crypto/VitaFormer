#!/bin/bash
# V100-2: OrganoIDNet on GPU1 now; OrgaSegment on GPU0 after Brain fold2 (PID 81447) exits. OMP/MKL=4.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
E=${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/external_m2_20261108; PY=${PYTHON:-python}; S=${WORKSPACE:-.}/3D_OrganoidSeg/scripts/train_external_m2_20261108.py
mkdir -p $E/logs
run() { G=$1; D=$2; echo "$(hostname) $(date) START $D GPU$G"
  CUDA_VISIBLE_DEVICES=$G $PY $S --model vitaformer --dataset $D --out $E/$D/m2_B > $E/logs/${D}_m2_B.full.log 2>&1; rc=$?
  echo "$(date) EXIT rc=$rc $D"; }
run 1 organoidnet &
( while kill -0 81447 2>/dev/null; do sleep 30; done; run 0 orgasegment ) &
wait; echo "$(date) ALL DONE"
