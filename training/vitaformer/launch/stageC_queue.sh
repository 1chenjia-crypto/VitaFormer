#!/usr/bin/env bash
# Stage C queue (Amendment 3). Waits for truly free GPUs among 1/3/5/7 (GPU 1 added 2026-11-05, Amendment 3a); never preempts.
E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104
PY=${PYTHON:-python}
TR=${WORKSPACE:-.}/3D_OrganoidSeg/vco/organoid_seg/train_v3_multiscale.py
JOBS=("m2 3407" "m2 2026" "m1 42" "m1 3407" "m1 2026")
CLAIM=$E/logs/stageC_claimed_gpus; : > $CLAIM
gpu_free() { # $1 index
  local u; u=$(nvidia-smi -i $1 --query-gpu=uuid,memory.used --format=csv,noheader,nounits)
  local uuid=${u%%,*}; local mem=${u##*, }
  [ "$mem" -lt 1000 ] || return 1
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid" && return 1
  grep -qx "$1" $CLAIM && return 1
  return 0
}
run_job() { # arch seed gpu
  local A=$1 S=$2 G=$3 N=stageC_$1_seed$2; local R=$E/$N
  echo "$(date) START $N on GPU $G"
  ( cd $E && sha256sum -c stageB_frozen_hashes.txt ) || { echo "$(date) HASH MISMATCH $N"; sed -i "/^$G$/d" $CLAIM; return; }
  CUDA_VISIBLE_DEVICES=$G $PY $TR --exp_name $N --out_dir $R \
    --lr 1e-4 --lr_encoder 1e-5 --max_steps 40000 --val_every 2000 \
    --freeze_blocks 4 --focal_loss --gamma 2.0 --dead_weight_min 10 \
    --use_dead_center_crop --dead_pixel_dropout_prob 0.0 \
    --sparse_dead_zoom_prob 0.8 --sparse_dead_threshold 5000 \
    --sparse_dead_zoom_factor 2 --oversample_5fu_factor 3 \
    --use_condition_split --seed $S --split_seed 42 --gpu 0 \
    --no_bam --no_mpm --arch $A > $E/logs/$N.train.full.log 2>&1
  local rc=$?; echo "$(date) TRAIN EXIT $rc $N"
  if [ $rc -eq 0 ] && grep -q TRAINING_COMPLETE $E/logs/$N.train.full.log; then
    CUDA_VISIBLE_DEVICES=$G $PY $E/scripts/stageB_evaluate.py --run_dir $R --arch $A --out ${R}_eval > $E/logs/$N.eval.full.log 2>&1
    echo "$(date) EVAL EXIT $? $N (final-print crash is cosmetic; check outputs)"
  fi
  sed -i "/^$G$/d" $CLAIM
  echo "$(date) DONE $N"
}
for J in "${JOBS[@]}"; do
  set -- $J; N=stageC_$1_seed$2
  if [ -e $E/$N ]; then echo "$(date) SKIP $N: out_dir exists (never reused)"; continue; fi
  while true; do
    G=""
    for g in 1 3 5 7; do if gpu_free $g; then sleep 180; gpu_free $g && { G=$g; break; }; fi; done
    [ -n "$G" ] && break; sleep 300
  done
  echo $G >> $CLAIM
  run_job $1 $2 $G &
  sleep 600   # let memory appear before the next free-GPU check
done
wait; echo "$(date) STAGE C QUEUE FINISHED"
