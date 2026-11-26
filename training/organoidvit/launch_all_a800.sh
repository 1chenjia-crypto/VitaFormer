#!/usr/bin/env bash
# Gate on the A800 env check, then start seeds 42/3407/2026 in separate tmux sessions, one per GPU (>=30 GB free).
set -euo pipefail
D=${WORKSPACE:-.}/OrganoidSegBaseline/orgvit_3seed_20261111
C=$D/env_check_a800/compare_seed0.txt
grep -q "AREA_CHECK PASS" $C || { echo "GATE FAIL: area check"; cat $C; exit 1; }
M=$(grep -oP "tools/test.py summary match: \('\K[0-9.]+" $C)
S=$(python3 -c "import json;print(json.load(open('$D/env_check_a800/shared_eval_seed0.json'))['mIoU'])")
python3 -c "import sys; m=float('$M'); s=float('$S'); sys.exit(0 if abs(m-75.30)<=0.05 and abs(s-75.30)<=0.05 else 1)" || { echo "GATE FAIL: mIoU tools=$M shared=$S"; exit 1; }
echo "$(date '+%F %T') A800 env check PASS: tools/test.py $M, shared $S; $(grep 'images with any' $C)" | tee -a $D/SEED_RECORD.md
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2>=30000' | sort -t, -k2 -nr | head -3 | cut -d, -f1)
[ ${#GPUS[@]} -ge 3 ] || { echo "only ${#GPUS[@]} GPUs with >=30 GB free: ${GPUS[*]}"; exit 2; }
i=0
for SEED in 42 3407 2026; do
  G=${GPUS[$i]}; i=$((i+1))
  tmux new-session -d -s orgvit_s$SEED "$D/run_seed_a800.sh $SEED $G > $D/logs/run_seed${SEED}_a800.full.log 2>&1"
  sleep 20
done
sleep 60
tail -6 $D/SEED_RECORD.md
for SEED in 42 3407 2026; do echo "== $SEED"; grep -E "launch_fp32|Iter \[" ${WORKSPACE:-.}/OrganoidSegBaseline/runs/orgvit_b0_1446_40k_cond_seed${SEED}_a800/train.full.log | tail -2 | cut -c1-200; done
