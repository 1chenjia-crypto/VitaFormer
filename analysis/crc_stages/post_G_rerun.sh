#!/bin/bash
# Re-run of the Stage G "all arms" steps. post_FG_auto.sh fired too early: its grep 'all done' matched Stage F's
# line ("Stage F finished: ... all done") in stageG_run.full.log, so analysis_all / objects failed (missing Full stats).
# This waits for the exact phase-3 marker and writes to new names (failed outputs kept untouched).
E=${WORKSPACE:-.}/3D_OrganoidSeg/vco/runs/arch_exploration_20261104; PY=${PYTHON:-python}; L=$E/logs
G=$E/stageG_dead_pr_val
until grep -q 'phase 3 done' $L/stageG_run.full.log 2>/dev/null; do sleep 60; done
$PY $E/scripts/stageG_analysis.py $G/analysis_all_run2 M2 nnUNet Full FullnoRD M0 > $L/stageG_analysis_all_run2.full.log 2>&1; echo "$(date) G full analysis exit $?"
$PY $E/scripts/stageG_objects.py $G/objects_delta0_run2.csv M2:0.0 nnUNet:0.0 Full:0.0 FullnoRD:0.0 M0:0.0 > $L/stageG_objects_delta0_run2.full.log 2>&1; echo "$(date) G objects exit $?"
echo "$(date) post rerun done"
