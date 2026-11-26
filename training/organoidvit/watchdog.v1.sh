#!/usr/bin/env bash
# Watchdog for the 3 A800 OrganoidViT seeds (Amendment 1: a crashed/stalled seed restarts FROM SCRATCH, recorded).
# Checks every 5 min. Max 2 restarts per seed. When all 3 are DONE, runs post_all.sh once and exits.
D=${WORKSPACE:-.}/OrganoidSegBaseline/orgvit_3seed_20261111; R=${WORKSPACE:-.}/OrganoidSegBaseline/runs
W=$D/logs/watchdog.log
declare -A TRIES=([42]=0 [3407]=0 [2026]=0)
log(){ echo "$(date '+%F %T') $*" | tee -a $W; }
log "watchdog start"
while true; do
  ndone=0
  for S in 42 3407 2026; do
    if grep -q "^seed=$S DONE" $D/SEED_RECORD.md; then ndone=$((ndone+1)); continue; fi
    RUN=$R/orgvit_b0_1446_40k_cond_seed${S}_a800; L=$RUN/train.full.log
    alive=0; tmux has-session -t orgvit_s$S 2>/dev/null && alive=1
    training=1; grep -q "^seed=$S best_val_iter=" $D/SEED_RECORD.md && training=0
    stalled=0
    if [ $alive = 1 ] && [ $training = 1 ] && [ -f $L ]; then
      age=$(( $(date +%s) - $(stat -c %Y $L) )); [ $age -gt 3600 ] && stalled=1
    fi
    if [ $alive = 0 ] && [ $training = 0 ]; then
      grep -q "seed $S POSTTRAIN_FAIL" $W || log "seed $S POSTTRAIN_FAIL: training finished but export/eval did not complete; NOT retraining (needs manual check of $RUN/export.full.log)"
      continue
    fi
    if [ $alive = 0 ] || [ $stalled = 1 ]; then
      reason="session ended without DONE"; [ $stalled = 1 ] && reason="train log not updated for >60 min"
      log "seed $S FAILED ($reason). last log line: $(tail -1 $L 2>/dev/null | cut -c1-160)"
      grep -E "Traceback|out of memory|Error" $L 2>/dev/null | tail -3 >> $W
      tmux kill-session -t orgvit_s$S 2>/dev/null
      for pid in $(pgrep -u "$USER" -f "orgvit_b0_1446_40k_cond_seed${S}_a800" ); do [ $pid != $$ ] && kill $pid 2>/dev/null; done
      sleep 20
      if [ ${TRIES[$S]} -ge 2 ]; then log "seed $S: max restarts reached, NOT restarting"; continue; fi
      TRIES[$S]=$((TRIES[$S]+1)); ts=$(date +%H%M)
      [ -e $RUN ] && mv $RUN $R/_failed_a800_seed${S}_try${TRIES[$S]}_$ts
      [ -e $D/masks_seed${S}_a800 ] && mv $D/masks_seed${S}_a800 $D/_failed_masks_seed${S}_try${TRIES[$S]}_$ts
      used=$(for s2 in 42 3407 2026; do grep -oP "^seed=$s2 host=gpu-host gpu_index=\K[0-9]+" $D/SEED_RECORD.md | tail -1; done)
      G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=30000' | sort -t, -k2 -nr | cut -d, -f1 | while read g; do echo "$used" | grep -qx "$g" || { echo $g; break; }; done)
      [ -z "$G" ] && G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=30000' | sort -t, -k2 -nr | head -1 | cut -d, -f1)
      if [ -z "$G" ]; then log "seed $S: no GPU with >=30 GB free; will retry next round"; TRIES[$S]=$((TRIES[$S]-1)); continue; fi
      echo "$(date '+%F %T') RESTART seed=$S from scratch (try ${TRIES[$S]}, reason: $reason) on GPU $G" >> $D/SEED_RECORD.md
      tmux new-session -d -s orgvit_s$S "$D/run_seed_a800.sh $S $G > $D/logs/run_seed${S}_a800_try${TRIES[$S]}.full.log 2>&1"
      log "seed $S restarted on GPU $G (try ${TRIES[$S]})"
    fi
  done
  if [ $ndone -eq 3 ]; then log "all 3 seeds DONE -> post_all"; $D/post_all.sh >> $W 2>&1; log "watchdog exit"; exit 0; fi
  sleep 300
done
