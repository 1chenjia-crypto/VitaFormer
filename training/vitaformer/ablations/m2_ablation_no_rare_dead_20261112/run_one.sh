#!/usr/bin/env bash
# args: seed gpu
E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/m2_ablation_no_rare_dead_20261112
A=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104
PY=${PYTHON:-python}
TR=${WORKSPACE:-.}/3D_OrganoidSeg/vco/organoid_seg/train_v3_multiscale.py
S=$1 G=$2 N=m2_noRD_seed$S; R=$E/$N
[ -e $R ] && { echo "$(date) REFUSE: $R exists"; exit 1; }
echo "$(date) START $N on $(hostname) GPU $G ($(nvidia-smi -i $G --query-gpu=name --format=csv,noheader))"
( cd $A && sha256sum -c stageB_frozen_hashes.txt ) || { echo "$(date) HASH MISMATCH"; exit 1; }
CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 $PY $TR --exp_name $N --out_dir $R \
  --lr 1e-4 --lr_encoder 1e-5 --max_steps 40000 --val_every 2000 \
  --freeze_blocks 4 --focal_loss --gamma 2.0 --dead_weight_min 10 \
  --use_dead_center_crop --dead_pixel_dropout_prob 0.0 \
  --sparse_dead_zoom_prob 0.8 --sparse_dead_threshold 5000 \
  --sparse_dead_zoom_factor 2 --oversample_5fu_factor 3 \
  --use_condition_split --seed $S --split_seed 42 --gpu 0 \
  --no_bam --no_mpm --arch m2 --no_rare_dead > $E/logs/$N.train.full.log 2>&1
echo "$(date) TRAIN EXIT $? $N"; grep -q TRAINING_COMPLETE $E/logs/$N.train.full.log && echo "$(date) DONE $N"
