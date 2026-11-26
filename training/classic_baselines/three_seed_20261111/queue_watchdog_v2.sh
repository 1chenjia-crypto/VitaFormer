#!/usr/bin/env bash
# Waits for the OrganoidViT 3-seed job to finish, then runs the 6 classic-baseline trainings (one per GPU, >=20 GB free,
# max 6 at once), monitors them (crash/stall >60 min -> relaunch same command = resume from latest.pth, max 4 per run),
# and when all 6 are DONE runs all_crc_models_summary.py once.
C=${WORKSPACE:-.}/OrganoidSegBaseline/classic_baselines_3seed_20261111; B=${WORKSPACE:-.}/OrganoidSegBaseline
OV=$B/orgvit_3seed_20261111; W=$C/logs/queue_watchdog.log
JOBS="unet:42 unet:3407 unetpp:42 unetpp:3407 deeplabv3:42 deeplabv3:3407"
declare -A TRIES LAUNCHED GPUOF
log(){ echo "$(date '+%F %T') $*" | tee -a $W; }
log "queue v2 start (fix: v1 tmux launches were refused as nested sessions, no training ran)"
until [ -e $OV/POST_DONE ] || ! tmux ls 2>/dev/null | grep -q '^orgvit_s'; do sleep 300; done
log "OrganoidViT finished ($( [ -e $OV/POST_DONE ] && echo POST_DONE || echo 'sessions gone' )); starting classic baselines"
while true; do
  ndone=0
  for J in $JOBS; do
    M=${J%%:*}; S=${J##*:}; K="$M:$S"; SES=cb_${M}_s$S; WD=$B/runs/${M}_cond_40k_torch_seed$S; L=$WD/train.full.log
    if grep -q "DONE model=$M seed=$S " $C/RUN_RECORD.md; then ndone=$((ndone+1)); continue; fi
    alive=0; tmux has-session -t $SES 2>/dev/null && alive=1
    if [ $alive = 1 ]; then
      if [ ! -e $WD/test_best_predictions.pkl ] && [ -f $L ] && [ $(( $(date +%s) - $(stat -c %Y $L) )) -gt 3600 ]; then
        log "$K stalled (>60 min no log) -> kill, will resume"; tmux kill-session -t $SES
        for pid in $(pgrep -u "$USER" -f "${M}_cond_40k_torch_seed$S"); do [ $pid != $$ ] && kill $pid 2>/dev/null; done; sleep 20
      else continue; fi
    fi
    if [ -e $WD/test_best_predictions.pkl ] && [ "${LAUNCHED[$K]:-0}" = 1 ]; then
      grep -q "$K POSTTRAIN_FAIL" $W || log "$K POSTTRAIN_FAIL: training finished but mask/eval step failed; not retraining"; continue
    fi
    if [ "${LAUNCHED[$K]:-0}" = 1 ]; then
      TRIES[$K]=$(( ${TRIES[$K]:-0} + 1 ))
      [ ${TRIES[$K]} -gt 4 ] && { grep -q "$K MAXTRIES" $W || log "$K MAXTRIES reached, not relaunching"; continue; }
      log "$K not running and not DONE -> relaunch (resume from latest.pth if present), try ${TRIES[$K]}: last log $(tail -1 $L 2>/dev/null | cut -c1-120)"
      echo "$(date '+%F %T') RECOVER model=$M seed=$S try=${TRIES[$K]} (resume from $( [ -e $WD/latest.pth ] && echo latest.pth || echo scratch))" >> $C/RUN_RECORD.md
    fi
    nrun=$(tmux ls 2>/dev/null | grep -cE '^cb_[a-z0-9]+_s[0-9]+:')
    [ $nrun -ge 6 ] && continue
    busy=$(for k in "${!GPUOF[@]}"; do tmux has-session -t "cb_${k%%:*}_s${k##*:}" 2>/dev/null && echo ${GPUOF[$k]}; done)
    G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=20000' | sort -t, -k2 -nr | cut -d, -f1 | while read g; do echo "$busy" | grep -qx "$g" || { echo $g; break; }; done)
    [ -z "$G" ] && { log "$K waiting: no free GPU (>=20 GB, not used by our runs)"; continue; }
    GPUOF[$K]=$G; LAUNCHED[$K]=1
    env -u TMUX tmux new-session -d -s $SES "$C/run_one.sh $M $S $G > $C/logs/run_${M}_seed${S}.full.log 2>&1"
    log "$K launched on GPU $G"; sleep 30
  done
  if [ $ndone -eq 6 ]; then
    log "all 6 DONE -> summary"; ${PYTHON:-python} $C/all_crc_models_summary.py >> $W 2>&1
    date '+%F %T' > $C/ALL_DONE; log "queue exit"; exit 0
  fi
  sleep 300
done
