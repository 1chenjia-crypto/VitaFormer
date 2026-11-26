#!/bin/bash
# Stage F runner: per GPU, M2 seed then M0 seed (test inference), then shared-evaluator JSONs.
E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104; PY=${PYTHON:-python}
EV=${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030/eval_masks_shared.py
one() { G=$1; S=$2
  for M in M2 M0; do
    CUDA_VISIBLE_DEVICES=$G $PY $E/scripts/stageF_test_inference.py $M $S > $E/logs/stageF_${M}_seed${S}.full.log 2>&1 || { echo "$(date) $M s$S FAILED"; return 1; }
    $PY $EV --masks $E/stageF_m2_test/test_masks/${M}_seed${S} --out $E/stageF_m2_test/shared_eval/${M}_final_seed${S}.json >> $E/logs/stageF_${M}_seed${S}.full.log 2>&1
    echo "$(date) $M s$S done"
  done; }
one 0 42 & one 3 3407 & one 4 2026 & wait
echo "$(date) all done"
