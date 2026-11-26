# nnU-Net ablation: patch 512×512 (exploratory, 2026-11-12)

Status: plan only; third in the approved order (after the two M2 ablations). EXPLORATORY: not in the main text;
at most one sentence in the Discussion. Does not change any existing nnU-Net result.

## Question
nnU-Net beats M2 on CRC (test mIoU 85.05 vs 83.34). nnU-Net trains on near-whole high-res images
(patch 1024×1280, median image 2044×2048); M2/SegFormer see 224 windows / 512 crops.
Is nnU-Net's advantage mainly the large field of view, or its training recipe?

## Design (one factor)
- Run-2 recipe unchanged (nnUNetTrainer, 1000 epochs, checkpoint_final, fold 0 = 481 train / 164 val, torch.compile off,
  threads = 1, same seeding launcher) except plans `nnUNetPlans_p512`:
  2d patch_size [1024,1280] → [512,512]; batch_size 2 → 10 so pixels per iteration stay equal (2.62M).
  Network (9 stages), normalisation, preprocessed data (data_identifier nnUNetPlans_2d) unchanged.
  Inference = nnU-Net sliding window with the 512 patch (tile step 0.5, gaussian), as for the original.
- Seed 42 only. Reference = run-2 seed 42 (and the 3-seed run-2 spread, 84.28–85.68 test, ~1.4 pt).

## Evaluation (fixed before results)
- Primary: validation (164 images), nnU-Net's own fold_0/validation masks with checkpoint_final
  (same source as the nnU-Net val numbers in Stage D), shared evaluator. Endpoints mIoU, Alive/Dead IoU, Dead P/R.
- Test: noTTA predictions + shared evaluator (as nnunet_post_train.sh), descriptive.
- Reading rule: drop ≥ 1.5 mIoU (beyond seed spread) → "field of view contributes substantially";
  |Δ| < 1.0 → "advantage not explained by field of view; likely training recipe"; in between → inconclusive (1 seed).

## Hardware
A800 (≥ 40 GB free GPU, as run 2). Run 2 took ~27 s/epoch (~7.5 h). Not planned for P100.

## Files
make_plans_p512.py (creates nnUNetPlans_p512.json, refuses overwrite), train_p512_seed.py (launcher).

## Amendment 2 (2026-11-13 09:35, author approved): add seeds 3407 and 2026 now
- Reason: free A800 GPUs; avoids a later 1-seed "inconclusive" outcome. Decided BEFORE any evaluation of seed 42
  (seed 42 still training, epoch ~425; its fold_0/validation has not been produced or evaluated).
- Same launcher train_p512_seed.py, seeds 3407 and 2026, fresh runs (A800 GPUs 0 and 1).
- Reference becomes run-2 seeds 42/3407/2026 (seed-matched). Primary contrast: mean over 3 seeds of
  (run2 − p512) val mIoU, with the same reading rule (≥1.5 field of view contributes substantially; <1.0 not explained by
  field of view; else inconclusive). Still exploratory; Discussion only.

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
