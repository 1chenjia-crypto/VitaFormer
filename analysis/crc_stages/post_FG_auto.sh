#!/bin/bash
# Detached post-processing: Stage F statistics after F inference; Stage G analyses after phase 1 (primary) and after all.
E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104; PY=${PYTHON:-python}; L=$E/logs
G=$E/stageG_dead_pr_val
until grep -qE 'all done|FAILED' $L/stageF_run.full.log; do sleep 60; done
echo "$(date) Stage F inference finished; running comparison"
$PY $E/scripts/stageF_comparison.py > $L/stageF_comparison.full.log 2>&1; echo "$(date) Stage F comparison exit $?"
until grep -q 'phase 1 done' $L/stageG_run.full.log 2>/dev/null; do sleep 60; done
$PY $E/scripts/stageG_analysis.py $G/analysis_primary M2 nnUNet > $L/stageG_analysis_primary.full.log 2>&1; echo "$(date) G primary analysis exit $?"
until grep -q 'all done' $L/stageG_run.full.log 2>/dev/null; do sleep 60; done
$PY $E/scripts/stageG_analysis.py $G/analysis_all M2 nnUNet Full FullnoRD M0 > $L/stageG_analysis_all.full.log 2>&1; echo "$(date) G full analysis exit $?"
$PY $E/scripts/stageG_objects.py $G/objects_delta0.csv M2:0.0 nnUNet:0.0 Full:0.0 FullnoRD:0.0 M0:0.0 > $L/stageG_objects_delta0.full.log 2>&1; echo "$(date) G objects exit $?"
echo "$(date) post all done"
