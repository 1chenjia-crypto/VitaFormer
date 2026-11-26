#!/bin/bash
# OrganoID human + mouse M2 protocol B, concurrently on GPU0 (V100-2). Completion is verified from result.json.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
E=${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/external_m2_20261108
cd ${WORKSPACE:-.}/3D_OrganoidSeg
for D in organoidid_human organoidid_mouse; do
  ( echo "$(hostname) $(date) START $D GPU0" >> $E/logs/organoidid_queue.full.log
    CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} scripts/train_organoidid_m2_20261108.py \
      --model vitaformer --dataset $D --out $E/$D/m2_B > $E/logs/${D}_m2_B.full.log 2>&1
    rc=$?; echo "$(date) EXIT rc=$rc $D" >> $E/logs/organoidid_queue.full.log ) &
done
wait; echo "$(date) ALL DONE" >> $E/logs/organoidid_queue.full.log
