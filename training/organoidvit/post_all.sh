#!/usr/bin/env bash
# Runs once, after all 3 A800 seeds are DONE: summary -> area error v2 -> Fig S5 v2 -> Fig 2 v2.
set -uo pipefail
D=${WORKSPACE:-.}/OrganoidSegBaseline/orgvit_3seed_20261111; P=${PYTHON:-python}
FS=${WORKSPACE:-.}/3D_OrganoidSeg/paper_materials_20261110/figure_scripts
[ -e $D/POST_DONE ] && exit 0
echo "post_all start $(date '+%F %T')"
$P $D/orgvit_summary.py > $D/logs/post_summary.full.log 2>&1; echo "summary exit $?"
mkdir -p ${WORKSPACE:-.}/3D_OrganoidSeg/crc_area_error_posthoc_v2_orgvit3seed
OMP_NUM_THREADS=4 $P ${WORKSPACE:-.}/3D_OrganoidSeg/scripts/crc_area_error_posthoc_v2_orgvit3seed.py \
  > ${WORKSPACE:-.}/3D_OrganoidSeg/crc_area_error_posthoc_v2_orgvit3seed/run.full.log 2>&1; echo "area v2 exit $?"
(cd $FS && $P figS5_crc_area_error_v2.py > $D/logs/post_figS5.full.log 2>&1); echo "figS5 v2 exit $?"
(cd $FS && $P fig2_draft_v2.py > $D/logs/post_fig2.full.log 2>&1); echo "fig2 v2 exit $?"
date '+%F %T' > $D/POST_DONE
echo "post_all end $(date '+%F %T')"
