#!/usr/bin/env bash
# One OrganoidViT seed on ONE GPU: train (config unchanged) -> best-val checkpoint -> export test masks -> shared evaluator.
# Usage: run_seed.sh SEED GPU_INDEX
set -euo pipefail
SEED=$1; GPU=$2
B=${WORKSPACE:-.}/OrganoidSegBaseline; D=$B/orgvit_3seed_20261111; V=$B/venv-orgvit-py36/bin/python
CFG=local_configs/2DSegFormer/organoid_voc_b0_1446_40k_cond.py
RUN=$B/runs/orgvit_b0_1446_40k_cond_seed$SEED
[ -e $RUN ] && { echo "refusing: $RUN exists"; exit 2; }
n=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" -eq 0 ] || { echo "refusing: GPU $GPU busy ($n processes)"; exit 3; }
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 PYTHONPATH=$B/Orgvit
mkdir -p $RUN
{ echo "seed=$SEED host=$(hostname) gpu_index=$GPU gpu=$(nvidia-smi -i $GPU --query-gpu=name --format=csv,noheader) precision=fp32 start=$(date '+%F %T')"; } | tee -a $D/SEED_RECORD.md
cd $B/Orgvit
$V $D/check_config.py >> $D/SEED_RECORD.md
$V tools/train.py $CFG --work-dir $RUN --seed $SEED > $RUN/train.full.log 2>&1
read BEST_ITER BEST_VAL < <($V $D/select_best.py $RUN)
echo "seed=$SEED best_val_iter=$BEST_ITER val_mIoU=$BEST_VAL train_end=$(date '+%F %T')" | tee -a $D/SEED_RECORD.md
$V $D/export_masks.py --cfg $CFG --ckpt $RUN/iter_${BEST_ITER}.pth --out $D/masks_seed$SEED > $RUN/export.full.log 2>&1
${PYTHON:-python} ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/eval_masks_shared.py \
  --masks $D/masks_seed$SEED --out $D/shared_eval/organoidvit_seed$SEED.json > $D/shared_eval_seed$SEED.full.log 2>&1
echo "seed=$SEED DONE $(date '+%F %T') $(tail -1 $D/shared_eval_seed$SEED.full.log)" | tee -a $D/SEED_RECORD.md
