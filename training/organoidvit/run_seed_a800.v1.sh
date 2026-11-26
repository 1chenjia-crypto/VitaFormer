#!/usr/bin/env bash
# Amendment 1: one OrganoidViT seed on ONE A800 GPU (shared), cu110 build, TF32 off. Recipe unchanged.
# Usage: run_seed_a800.sh SEED GPU_INDEX
set -euo pipefail
SEED=$1; GPU=$2
B=${WORKSPACE:-.}/OrganoidSegBaseline; D=$B/orgvit_3seed_20261111; V=$B/venv-orgvit-py36-cu110/bin/python
L="$V $D/launch_fp32.py"
CFG=local_configs/2DSegFormer/organoid_voc_b0_1446_40k_cond.py
RUN=$B/runs/orgvit_b0_1446_40k_cond_seed${SEED}_a800
[ "$(hostname)" = "${EXPECTED_HOST:-gpu-host}" ] || { echo "refusing: not on gpu-host"; exit 4; }
[ -e $RUN ] && { echo "refusing: $RUN exists"; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 PYTHONPATH=$B/Orgvit
FREE=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits)
NP=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$FREE" -ge 30000 ] || { echo "refusing: GPU $GPU only ${FREE} MiB free"; exit 3; }
mkdir -p $RUN
echo "seed=$SEED host=$(hostname) gpu_index=$GPU gpu=$(nvidia-smi -i $GPU --query-gpu=name --format=csv,noheader) build=torch1.7.1+cu110 precision=fp32(TF32 off) shared_with=${NP}_other_procs free_MiB=$FREE start=$(date '+%F %T')" | tee -a $D/SEED_RECORD.md
cd $B/Orgvit
$V $D/check_config.py >> $D/SEED_RECORD.md
$L tools/train.py $CFG --work-dir $RUN --seed $SEED > $RUN/train.full.log 2>&1
read BEST_ITER BEST_VAL < <($V $D/select_best.py $RUN)
echo "seed=$SEED best_val_iter=$BEST_ITER val_mIoU=$BEST_VAL train_end=$(date '+%F %T')" | tee -a $D/SEED_RECORD.md
$L $D/export_masks.py --cfg $CFG --ckpt $RUN/iter_${BEST_ITER}.pth --out $D/masks_seed${SEED}_a800 > $RUN/export.full.log 2>&1
${PYTHON:-python} ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/eval_masks_shared.py \
  --masks $D/masks_seed${SEED}_a800 --out $D/shared_eval/organoidvit_seed$SEED.json > $D/shared_eval_seed$SEED.full.log 2>&1
echo "seed=$SEED DONE $(date '+%F %T') $(tail -1 $D/shared_eval_seed$SEED.full.log)" | tee -a $D/SEED_RECORD.md
