# M2 ablation: w/o Rare-Dead (2026-11-12)

Approved by author 2026-11-12 ("M2 去掉 Rare-Dead 的 3 种子消融，肯定需要做").
This is an ablation only. M2 stays the frozen main model (MODEL_FREEZE_M2_20261108); nothing here replaces it.

## Design
- Model: M2 (`train_v3_multiscale.py --arch m2 --no_bam --no_mpm`), exactly the Stage B/C command, plus `--no_rare_dead`.
- `--no_rare_dead` (code unchanged, same flag as the earlier base-architecture ablation) turns off: focal loss, class weights,
  Dead-centre crop, sparse-Dead zoom, 5-FU oversampling, oversample, Dead-pixel dropout, Dead copy-paste.
- Seeds 42 / 3407 / 2026, split_seed 42, 40k steps, same LR/freeze settings.
- Code hash check against stageB_frozen_hashes.txt before each run; refuse to start if out_dir exists.

## Evaluation (fixed before results)
- Checkpoint: final.pth at step 40000 (selection-free, as Stage D/F and the frozen M2).
- Primary: validation set, seed-matched contrast M2 − noRD, mean over 3 seeds with bootstrap CI (same machinery as stageD_analysis.py).
- Endpoints: mIoU; Dead IoU, Dead precision, Dead recall; Alive IoU.
- Test set: reported descriptively (already used for M2 in Stage F).
- All results reported whatever they show.

## Hardware
- P100 16 GB is enough (peak ~5 GB, ~0.76 s/step, ~9–10 h per run; measured 2026-11-12 on GPU-P100-2).

## Amendment 1 (2026-11-13, author approved): resume after NFS outage
- NFS storage outage (dmesg "nfs4_reclaim_open_state: Lock reclaim failed"): A800 runs hung from ~02:38 and exited 07:07 with
  OSError [Errno 5] Input/output error while reading a training JPEG (file intact and readable afterwards).
- Affected: imagenet s42 (stopped 38700; resume from 38000), imagenet s2026 (39000; 38000), imagenet s3407 (32000; 32000),
  noRD s3407 (39600; 38000). Not affected: noRD s42 (P100, done 00:46), noRD s2026 (done 02:31).
- Resume = same command + `--resume_full checkpoint_resume_preoutage.pth` (model, optimizer, scheduler, step, history restored;
  data-loader RNG not restored → batch order after the resume point differs from an uninterrupted run).
  Pre-outage checkpoints kept as checkpoint_resume_preoutage.pth; script ../m2_ablation_imagenet_20261112/resume_one.sh;
  logs logs/*.resume1.full.log. A800 GPUs 0/1/3/4, 2026-11-13 08:50. Evaluation rule unchanged (final.pth @40000).
- Methods will state which runs were resumed.
