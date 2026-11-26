#!/bin/bash
# GPU0 is exclusive-process: human waits for the mouse run's result.json, then runs on GPU0.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
E=${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/external_m2_20261108
until [ -f $E/organoidid_mouse/m2_B/result.json ]; do sleep 60; done; sleep 30
cd ${WORKSPACE:-.}/3D_OrganoidSeg
echo "$(hostname) $(date) START organoidid_human GPU0 (attempt 2)" >> $E/logs/organoidid_queue.full.log
CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} scripts/train_organoidid_m2_20261108.py \
  --model vitaformer --dataset organoidid_human --out $E/organoidid_human/m2_B > $E/logs/organoidid_human_m2_B.full.log 2>&1
rc=$?; echo "$(date) EXIT rc=$rc organoidid_human" >> $E/logs/organoidid_queue.full.log
