#!/bin/bash
# Stage G runner (AMENDMENT_1 §7): waits for Stage F to finish (no GPU contention), then per phase on GPUs 0/3/4:
# phase 1 = M2 + nnU-Net (primary), phase 2 = Full + Full-noRD, phase 3 = M0. Each phase waits for all three GPUs.
E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104
PY=${PYTHON:-python}; PN=${PYTHON:-python}
L=$E/logs; GPUS=(0 3 4); SEEDS=(42 3407 2026)
export nnUNet_compile=f
until grep -qE 'all done|FAILED' $L/stageF_run.full.log; do sleep 60; done
echo "$(date) Stage F finished: $(tail -n 1 $L/stageF_run.full.log)"
vf() { CUDA_VISIBLE_DEVICES=$1 $PY $E/scripts/stageG_export_vf.py $2 $3 > $L/stageG_$2_seed$3.full.log 2>&1; echo "$(date) $2 s$3 exit $?"; }
nn() { CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PN $E/scripts/stageG_export_nnunet.py $2 > $L/stageG_nnUNet_seed$2.full.log 2>&1; echo "$(date) nnUNet s$2 exit $?"; }
for i in 0 1 2; do ( vf ${GPUS[$i]} M2 ${SEEDS[$i]}; nn ${GPUS[$i]} ${SEEDS[$i]} ) & done; wait
echo "$(date) phase 1 done"
for i in 0 1 2; do ( vf ${GPUS[$i]} Full ${SEEDS[$i]}; vf ${GPUS[$i]} FullnoRD ${SEEDS[$i]} ) & done; wait
echo "$(date) phase 2 done"
for i in 0 1 2; do vf ${GPUS[$i]} M0 ${SEEDS[$i]} & done; wait
echo "$(date) phase 3 done; all done"
