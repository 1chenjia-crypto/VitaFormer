#!/usr/bin/env bash
# Detached post-training for nnU-Net run 2 (P0B_PROTOCOL.md §7): per seed, after its training process has exited and
# checkpoint_final.pth exists, predict imagesTs on the same GPU (no TTA = primary, then TTA = secondary), then run the
# shared evaluator. Never overwrites: skips a prediction dir that already exists. Full logs per step.
#   usage: nnunet_post_train.sh GPU SEED TRAIN_PID
cd ${WORKSPACE:-.}/3D_OrganoidSeg/baselines_fair_20261030
GPU=$1; SEED=$2; TPID=$3
N=nnunet; L=$N/logs; R=$N/nnUNet_results_run2_seed$SEED
export nnUNet_raw=$PWD/$N/nnUNet_raw nnUNet_preprocessed=$PWD/$N/nnUNet_preprocessed nnUNet_results=$PWD/$R
export nnUNet_compile=f OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
BIN=${WORKSPACE:-.}/env_totalseg2/bin
while kill -0 $TPID 2>/dev/null; do sleep 120; done
CK=$(ls $R/Dataset801_CRCOrganoid/*/fold_0/checkpoint_final.pth 2>/dev/null)
if [ -z "$CK" ]; then echo "$(date) seed $SEED: training exited WITHOUT checkpoint_final.pth; nothing predicted" >> $L/post_train.log; exit 1; fi
echo "$(date) seed $SEED: training finished ($CK)" >> $L/post_train.log
for mode in noTTA TTA; do
  OUT=$N/pred_run2_seed${SEED}_$mode
  if [ -e $OUT ]; then echo "$(date) seed $SEED $mode: $OUT exists, skipped" >> $L/post_train.log; continue; fi
  FLAG=""; [ $mode = noTTA ] && FLAG="--disable_tta"
  CUDA_VISIBLE_DEVICES=$GPU $BIN/nnUNetv2_predict -i $nnUNet_raw/Dataset801_CRCOrganoid/imagesTs -o $OUT \
    -d 801 -c 2d -f 0 -chk checkpoint_final.pth $FLAG > $L/predict_run2_seed${SEED}_$mode.full.log 2>&1
  echo "$(date) seed $SEED $mode: predict exit $? ($(ls $OUT/*.png 2>/dev/null | wc -l) png)" >> $L/post_train.log
  ${PYTHON:-python} eval_masks_shared.py --masks $OUT \
    --out shared_eval/nnunet_run2_seed${SEED}_$mode.json > shared_eval/nnunet_run2_seed${SEED}_$mode.full.log 2>&1
  echo "$(date) seed $SEED $mode: eval $(tail -1 shared_eval/nnunet_run2_seed${SEED}_$mode.full.log)" >> $L/post_train.log
done
