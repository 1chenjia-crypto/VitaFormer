#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
cd ${WORKSPACE:-.}/OrganoidSegBaseline/Orgvit
mkdir -p ${WORKSPACE:-.}/OrganoidSegBaseline/runs/orgvit_b0_1446_40k_cond
/tmp/venv-OrganoidSegbaseline/bin/python tools/train.py   local_configs/2DSegFormer/organoid_voc_b0_1446_40k_cond.py   --work-dir ${WORKSPACE:-.}/OrganoidSegBaseline/runs/orgvit_b0_1446_40k_cond   --seed 0
