# nnU-Net ablation: CE loss only (exploratory, 2026-11-13)

Author approved 2026-11-13 ("补一个种子的nnU-Net只用CE跑一下"). EXPLORATORY: not in main text; at most one sentence in the Discussion.
Written before training; no result of this variant exists.

## Question
Without any rare-class strategy, nnU-Net reaches Dead recall ~74 (test) while M2 without Rare-Dead (unweighted CE) drops to ~57.
Hypothesis (untested): nnU-Net's Dice term (batch Dice + CE) acts as built-in class-imbalance handling.

## Design (one factor)
- Run-2 recipe unchanged (nnUNetPlans, patch 1024×1280, batch 2, 1000 epochs, deep supervision, checkpoint_final, fold 0,
  same seeding launcher, torch.compile off) except trainer = official `nnUNetTrainerCELoss` (nnunetv2 2.8.1):
  unweighted RobustCrossEntropyLoss, deep-supervision weights as default. Dice term removed. No code change.
- Seed 42 only. Reference = run-2 seed 42 (val), with the run-2 3-seed spread for context.

## Evaluation (fixed before results)
- Primary: validation (164), nnU-Net's own fold_0/validation masks (checkpoint_final), shared evaluator / ablation_analysis-style
  confusion metrics. Endpoints: Dead recall (primary), Dead IoU, Dead precision, mIoU.
- Reading rule (1 seed, so only large effects interpretable): Dead recall drop ≥ 10 points vs run-2 s42 → "Dice term contributes
  substantially to nnU-Net's rare-class recall"; |Δ| < 5 → "not explained by the Dice term"; else inconclusive.
- Test: noTTA predictions + shared evaluator, descriptive.

## Amendment 3 (2026-11-13 11:08): resume after NFS outage #3 (10:52)
All running nnU-Net runs died (FileNotFoundError in data workers; dmesg "Lock reclaim failed"). Same approved resume rule
(continue_training from checkpoint_latest, every 50 epochs). To stop repeated NFS failures, preprocessed data are now read from a
local RAM-disk copy (/dev/shm/<user>_nnUNet_preprocessed; all 2585 files sha256-identical to the original). Script resume_local.py.
Pre-resume checkpoints copied (checkpoint_latest_seed*_pre_outage2_*.pth). Evaluation rules unchanged.

## Amendment 4 (2026-11-14): validation-only rerun
In-training validation failed for all runs: resume_local.py (and train_p512_seed.py / train_celoss_seed.py) had no
`if __name__ == "__main__"` guard, so nnU-Net's spawned export workers re-executed the script (RuntimeError, "background workers
are no longer alive"); no validation masks were written. Training itself completed (epoch 1000, checkpoint_final.pth present).
Spawned children re-saved checkpoint_final at ~19:06–21:11 after loading it and training 0 epochs (weights expected unchanged;
no pre-copy to verify byte-identity). Fix: official `nnUNetv2_train 801 2d 0 -tr <trainer> -p <plans> --val` (has the guard),
checkpoint_final, same validation procedure as planned. Evaluation rules unchanged.
