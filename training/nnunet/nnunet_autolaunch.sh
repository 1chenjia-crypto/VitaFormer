#!/usr/bin/env bash
# Detached (nohup) launcher for the 3 approved nnU-Net seeds; survives terminal/network loss.
#   seed 42   -> GPU 3, after P0-M inference (PIDs 2877, 2878) has exited
#   seed 3407 -> GPU 5, same condition
#   seed 2026 -> GPU 7, after all SegFormer-B2 training processes have exited
# Before each launch the GPU must have >= 40 GB free (memory used by any process <= 40 GB),
# so other users' jobs are not squeezed. Re-checked every 5 minutes until satisfied. Full logs per seed.
cd ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030
PY=${PYTHON:-python}
LOG=nnunet/logs
free_ok() { local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"); [ "$used" -le 40000 ]; }
launch() {  # gpu seed
  until free_ok "$1"; do echo "$(date) GPU $1 busy, waiting" >> $LOG/autolaunch.log; sleep 300; done
  echo "$(date) launching seed $2 on GPU $1 (used MB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1))" >> $LOG/autolaunch.log
  CUDA_VISIBLE_DEVICES=$1 nohup $PY nnunet_train_seed.py --seed $2 > $LOG/train_seed$2.full.log 2>&1 &
  echo "$(date) seed $2 pid $!" >> $LOG/autolaunch.log
}
while kill -0 2877 2>/dev/null || kill -0 2878 2>/dev/null; do sleep 120; done
echo "$(date) P0-M inference finished" >> $LOG/autolaunch.log
launch 3 42
launch 5 3407
while pgrep -f train_segformer_b2.py > /dev/null; do sleep 300; done
echo "$(date) SegFormer-B2 finished" >> $LOG/autolaunch.log
launch 7 2026
wait
