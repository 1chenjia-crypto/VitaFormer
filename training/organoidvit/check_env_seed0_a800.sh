#!/usr/bin/env bash
# Environment check: seed-0 iter_32000 through (1) original tools/test.py and (2) mask export + shared evaluator.
set -euo pipefail
GPU=$1
B=${WORKSPACE:-.}/OrganoidSegBaseline; D=$B/orgvit_3seed_20261111; V=$B/venv-orgvit-py36-cu110/bin/python
L="$V $D/launch_fp32.py"
RUN0=$B/runs/orgvit_b0_1446_40k_cond; CFG=local_configs/2DSegFormer/organoid_voc_b0_1446_40k_cond.py
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 PYTHONPATH=$B/Orgvit
mkdir -p $D/env_check_a800
cd $B/Orgvit
$L tools/test.py $CFG $RUN0/iter_32000.pth --eval mIoU mDice > $D/env_check_a800/tools_test_seed0.full.log 2>&1
$L $D/export_masks.py --cfg $CFG --ckpt $RUN0/iter_32000.pth --out $D/env_check_a800/masks_seed0 > $D/env_check_a800/export_seed0.full.log 2>&1
${PYTHON:-python} ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/eval_masks_shared.py \
  --masks $D/env_check_a800/masks_seed0 --out $D/env_check_a800/shared_eval_seed0.json > $D/env_check_a800/shared_eval_seed0.full.log 2>&1
${PYTHON:-python} $D/compare_seed0.py env_check_a800 > $D/env_check_a800/compare_seed0.txt 2>&1
cat $D/env_check_a800/compare_seed0.txt
