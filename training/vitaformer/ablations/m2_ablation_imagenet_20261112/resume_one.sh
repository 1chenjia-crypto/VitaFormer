#!/usr/bin/env bash
# Amendment 1 (2026-11-13, author approved): resume after NFS outage. args: ABL(noRD|imagenet) seed gpu
# Same command as run_one.sh + --resume_full <run>/checkpoint_resume.pth. Pre-outage checkpoint copied first. New log file.
ABL=$1 S=$2 G=$3
case $ABL in noRD) E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/m2_ablation_no_rare_dead_20261112; X=--no_rare_dead;;
            imagenet) E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/m2_ablation_imagenet_20261112; X=--use_imagenet;; *) exit 2;; esac
A=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104
PY=${PYTHON:-python}; TR=${WORKSPACE:-.}/3D_OrganoidSeg/vco/organoid_seg/train_v3_multiscale.py
N=m2_${ABL}_seed$S; R=$E/$N
[ -f $R/final.pth ] && { echo "$(date) REFUSE: $R already has final.pth"; exit 1; }
[ -f $R/checkpoint_resume_preoutage.pth ] || cp -p $R/checkpoint_resume.pth $R/checkpoint_resume_preoutage.pth
echo "$(date) RESUME $N on $(hostname) GPU $G from $(sha256sum $R/checkpoint_resume_preoutage.pth | cut -c1-16)"
( cd $A && sha256sum -c stageB_frozen_hashes.txt ) || { echo "$(date) HASH MISMATCH"; exit 1; }
CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 $PY $TR --exp_name $N --out_dir $R \
  --lr 1e-4 --lr_encoder 1e-5 --max_steps 40000 --val_every 2000 \
  --freeze_blocks 4 --focal_loss --gamma 2.0 --dead_weight_min 10 \
  --use_dead_center_crop --dead_pixel_dropout_prob 0.0 \
  --sparse_dead_zoom_prob 0.8 --sparse_dead_threshold 5000 \
  --sparse_dead_zoom_factor 2 --oversample_5fu_factor 3 \
  --use_condition_split --seed $S --split_seed 42 --gpu 0 \
  --no_bam --no_mpm --arch m2 $X --resume_full $R/checkpoint_resume_preoutage.pth > $E/logs/$N.resume1.full.log 2>&1
echo "$(date) TRAIN EXIT $? $N"; grep -q TRAINING_COMPLETE $E/logs/$N.resume1.full.log && echo "$(date) DONE $N (resumed)"
