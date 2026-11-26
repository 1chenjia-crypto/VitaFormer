#!/usr/bin/env bash
set -euo pipefail

cd ${WORKSPACE:-.}/OrganoidSegBaseline

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

${PYTHON:-python} scripts/train_unetpp_baseline.py \
  --voc-root ${WORKSPACE:-.}/OrganoidSegBaseline/VOC2012 \
  --train-split ${WORKSPACE:-.}/OrganoidSegBaseline/VOC2012/ImageSets/Segmentation/train_cond.txt \
  --val-split ${WORKSPACE:-.}/OrganoidSegBaseline/VOC2012/ImageSets/Segmentation/val_cond.txt \
  --test-split ${WORKSPACE:-.}/OrganoidSegBaseline/VOC2012/ImageSets/Segmentation/test_cond.txt \
  --work-dir ${WORKSPACE:-.}/OrganoidSegBaseline/runs/unetpp_cond_40k_torch \
  --max-iters 40000 \
  --batch-size 4 \
  --crop-size 512 \
  --base-channels 64 \
  --lr 2e-4 \
  --val-interval 1000 \
  --save-interval 1000 \
  --num-workers 4 \
  --resume
