#!/usr/bin/env bash
# Waits for Stage B M2 seed 42 training to finish, then runs the pre-specified validation evaluation on the same GPU.
E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104
R=$E/stageB_m2_seed42
until grep -q -E "TRAINING_COMPLETE|Traceback" $R/train.log; do sleep 60; done
if grep -q Traceback $R/train.log; then echo "$(date) training FAILED; no evaluation" ; exit 1; fi
echo "$(date) training complete; evaluating"
CUDA_VISIBLE_DEVICES=3 ${PYTHON:-python} $E/scripts/stageB_evaluate.py \
  --run_dir $R --arch m2 --out $E/stageB_m2_seed42_eval
echo "$(date) evaluation exit $?"
