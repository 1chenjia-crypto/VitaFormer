#!/usr/bin/env bash
# Amendment 5 runner: waits for a truly free GPU among 1/3/5/7 (no other process, <1 GB, two checks 2 min apart).
E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104; PY=${PYTHON:-python}
free() { local u; u=$(nvidia-smi -i $1 --query-gpu=uuid,memory.used --format=csv,noheader,nounits); [ "${u##*, }" -lt 1000 ] || return 1
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "${u%%,*}"; }
while true; do for g in 3 1 5 7; do if free $g; then sleep 120; free $g && G=$g && break 2; fi; done; sleep 300; done
echo "$(date) start on GPU $G"; cd /tmp
(cd $E && sha256sum -c stageE_frozen_hashes.txt) || { echo HASH MISMATCH; exit 1; }
CUDA_VISIBLE_DEVICES=$G $PY $E/scripts/stageE_full_inference.py; rc=$?; echo "$(date) inference exit $rc"
[ $rc -eq 0 ] && $PY $E/scripts/stageE_analysis.py --ckpt final --out $E/stageE_full_40k/analysis_final > $E/logs/stageE_analysis.full.log 2>&1; echo "$(date) analysis exit $?"
echo "$(date) STAGE E DONE"
