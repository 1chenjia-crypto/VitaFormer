#!/usr/bin/env bash
# Waits for classic-baseline tmux sessions to end, then:
#   ImageNet seed 2026 -> GPU of cb_unet_s42 (5);  nnU-Net p512 seed 42 -> GPU of cb_unet_s3407 (4), needs >=40 GB free.
L=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/m2_ablation_imagenet_20261112/logs/queue_after_unet.log
log(){ echo "$(date '+%F %T') $*" >> $L; }
log "queue start"
while tmux has-session -t cb_unet_s42 2>/dev/null; do sleep 120; done
log "cb_unet_s42 ended -> ImageNet s2026 on GPU 5"
env -u TMUX tmux new-session -d -s m2imnet_s2026 "bash ${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/m2_ablation_imagenet_20261112/run_one.sh 2026 5 2>&1 | tee -a ${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/m2_ablation_imagenet_20261112/logs/runner_seed2026.log"
while tmux has-session -t cb_unet_s3407 2>/dev/null; do sleep 120; done
until [ $(nvidia-smi -i 4 --query-gpu=memory.free --format=csv,noheader,nounits) -ge 40000 ]; do log "GPU 4 <40GB free, waiting"; sleep 300; done
log "cb_unet_s3407 ended -> nnU-Net p512 s42 on GPU 4"
env -u TMUX tmux new-session -d -s nnunet_p512_s42 "cd ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/nnunet_ablation_p512_20261112 && CUDA_VISIBLE_DEVICES=4 ${PYTHON:-python} ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/nnunet_ablation_p512_20261112/train_p512_seed.py --seed 42 > ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/nnunet_ablation_p512_20261112/logs/train_p512_seed42.full.log 2>&1; echo EXIT \$? >> ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/nnunet_ablation_p512_20261112/logs/train_p512_seed42.full.log"
log "queue done"
