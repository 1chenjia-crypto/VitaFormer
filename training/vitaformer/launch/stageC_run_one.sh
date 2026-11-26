#!/usr/bin/env bash
# Stage C single run (Amendment 3b): same command as stageC_queue.sh run_job; args: arch seed gpu. Refuses existing out_dir.
E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104
PY=${PYTHON:-python}
TR=${WORKSPACE:-.}/3D_OrganoidSeg/vco/organoid_seg/train_v3_multiscale.py
A=$1 S=$2 G=$3 N=stageC_$1_seed$2; R=$E/$N
echo "$(date) START $N on $(hostname) GPU $G ($(nvidia-smi -i $G --query-gpu=name --format=csv,noheader))"
( cd $E && sha256sum -c stageB_frozen_hashes.txt ) || { echo "$(date) HASH MISMATCH $N"; exit 1; }
CUDA_VISIBLE_DEVICES=$G $PY $TR --exp_name $N --out_dir $R \
  --lr 1e-4 --lr_encoder 1e-5 --max_steps 40000 --val_every 2000 \
  --freeze_blocks 4 --focal_loss --gamma 2.0 --dead_weight_min 10 \
  --use_dead_center_crop --dead_pixel_dropout_prob 0.0 \
  --sparse_dead_zoom_prob 0.8 --sparse_dead_threshold 5000 \
  --sparse_dead_zoom_factor 2 --oversample_5fu_factor 3 \
  --use_condition_split --seed $S --split_seed 42 --gpu 0 \
  --no_bam --no_mpm --arch $A > $E/logs/$N.train.full.log 2>&1
rc=$?; echo "$(date) TRAIN EXIT $rc $N"
if [ $rc -eq 0 ] && grep -q TRAINING_COMPLETE $E/logs/$N.train.full.log; then
  CUDA_VISIBLE_DEVICES=$G $PY $E/scripts/stageB_evaluate.py --run_dir $R --arch $A --out ${R}_eval > $E/logs/$N.eval.full.log 2>&1
  echo "$(date) EVAL EXIT $? $N (final-print crash is cosmetic; check outputs)"
fi
echo "$(date) DONE $N"
