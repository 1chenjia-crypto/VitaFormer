#!/usr/bin/env bash
# One classic baseline run: original nohup command, only --seed and --work-dir changed. The original command already
# carries --resume, so relaunching the same command resumes from latest.pth (PLAN: interruption rule).
# Usage: run_one.sh MODEL(unet|unetpp|deeplabv3) SEED GPU
set -euo pipefail
M=$1; S=$2; G=$3
B=${WORKSPACE:-.}/OrganoidSegBaseline; C=$B/classic_baselines_3seed_20261111; P=${PYTHON:-python}
WD=$B/runs/${M}_cond_40k_torch_seed$S
mkdir -p $WD
FREE=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits); NP=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | wc -l)
echo "$(date '+%F %T') START model=$M seed=$S host=$(hostname) gpu_index=$G gpu=$(nvidia-smi -i $G --query-gpu=name --format=csv,noheader) shared_with=${NP}_other_procs free_MiB=$FREE resume_from=$( [ -e $WD/latest.pth ] && echo latest.pth || echo none)" >> $C/RUN_RECORD.md
cd $B
CMD=$(sed -n '/venv_OrganoidSegBaseline\/bin\/python/,$p' $B/run_${M}_cond_40k_torch_nohup.sh | tr -d '\\' | tr -s ' \n' ' ')
CMD=$(echo "$CMD" | sed -E "s|--work-dir [^ ]+|--work-dir $WD|; s|>.*$||")
CMD="$CMD --seed $S"
echo "$(date '+%F %T') CMD $CMD" >> $WD/train.full.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 bash -c "$CMD" >> $WD/train.full.log 2>&1
$P $C/pkl_to_png.py $WD/test_best_predictions.pkl $C/masks_${M}_seed$S >> $WD/train.full.log 2>&1
$P ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/eval_masks_shared.py --masks $C/masks_${M}_seed$S --out $C/shared_eval/${M}_seed$S.json > $C/logs/shared_eval_${M}_seed$S.log 2>&1
BEST=$(python3 -c "import torch" 2>/dev/null; grep -oE "BEST iter=[0-9]+ mIoU=[0-9.]+" $WD/train.full.log | tail -1)
echo "$(date '+%F %T') DONE model=$M seed=$S $BEST test=$(tail -1 $C/logs/shared_eval_${M}_seed$S.log)" >> $C/RUN_RECORD.md
