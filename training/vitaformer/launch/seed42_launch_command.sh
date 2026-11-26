#!/usr/bin/env bash
# Stage B, M2 seed 42 (approved 2026-11-04). Identical to vco/run_combined_no_bam_no_mpm_20261026.sh except --arch m2 and paths.
CUDA_VISIBLE_DEVICES=3 ${PYTHON:-python} ${WORKSPACE:-.}/3D_OrganoidSeg/vco/organoid_seg/train_v3_multiscale.py \
  --exp_name stageB_m2_seed42 --out_dir ${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104/stageB_m2_seed42 \
  --lr 1e-4 --lr_encoder 1e-5 --max_steps 40000 --val_every 2000 \
  --freeze_blocks 4 --focal_loss --gamma 2.0 --dead_weight_min 10 \
  --use_dead_center_crop --dead_pixel_dropout_prob 0.0 \
  --sparse_dead_zoom_prob 0.8 --sparse_dead_threshold 5000 \
  --sparse_dead_zoom_factor 2 --oversample_5fu_factor 3 \
  --use_condition_split --seed 42 --split_seed 42 --gpu 0 \
  --no_bam --no_mpm --arch m2
