# Classic CRC baselines (U-Net, U-Net++, DeepLabv3): three seeds — plan (post hoc supplement)

**Written:** 2026-11-11 19:20 CST, before any new training. Approved by the author on 2026-11-11.
**Status:** post hoc supplement. These are the three CRC baselines used in the OrganoidViT source paper
(Fig. 5 of that paper: relative area error of U-Net, U-Net++, DeepLabv3 and OrganoidViT on the CRC test set).
They are reported so that the source paper's comparison is reproduced in the same framework as our other models.
They are CRC-only and enter no functional or external analysis.

## Existing runs and a correction
The single-seed runs (`runs/{unet,unetpp,deeplabv3}_cond_40k_torch`) were trained with **seed 2026** (`run_meta.json`, the
script default), not "s42" as labelled in `crc_all_baselines_4class_20261109`. They are kept unchanged and used as seed 2026.

## New training (only the seed changes)
- **Seeds:** 42 and 3407 for each model, giving 6 runs. With the existing 2026 runs, each model has seeds 42, 3407 and 2026, the same as all other models.
- **Command:** identical to `run_{model}_cond_40k_torch_nohup.sh`, except `--seed S --work-dir runs/{model}_cond_40k_torch_seed{S}`.
  - Settings: 40k iterations, batch 4, crop 512, lr 2e-4, weight decay 1e-4, base channels 64, 4 workers, AMP on (the script default, as in the original runs), from scratch (no pretrained weights), class-weighted loss as in the script.
  - Environment: `venv_OrganoidSegBaseline` (torch 2.6.0+cu124), unchanged.
- **Checkpoint rule (unchanged):** validation every 1,000 iterations; the best validation mIoU checkpoint (`best_miou.pth`) is used for test. The script's final test writes `test_best_predictions.pkl`.
- **Hardware:** gpu-host, shared GPUs (the author approved sharing for OrganoidViT; the same applies here). One run per GPU, on GPUs with ≥ 20 GB free. These models need far less memory than OrganoidViT. At most 6 runs at once. GPU index and the number of other processes are recorded in `RUN_RECORD.md`.
- **Start:** automatically after the three OrganoidViT seeds finish (`orgvit_3seed_20261111/POST_DONE`), so the two jobs do not compete for the overloaded CPU.
- **Interruption (same rule as OrganoidViT Amendment 2):** a crashed or stalled run (log not updated for > 60 min) resumes from its latest checkpoint with the script's own `--resume` (`latest.pth`, saved every 1,000 iterations). Every resume is recorded and will be stated in Methods.

## Evaluation (shared evaluator for all 3 seeds)
- **Seed 2026:** the existing `test_best_predictions.pkl` (best-validation checkpoint) is converted to PNG masks. No new inference.
  - **Check:** shared-evaluator mIoU must equal the original script's mIoU (44.75 / 63.24 / 75.53) to < 0.01 pp.
- **Seeds 42 and 3407:** the same conversion of the pkl written by the training script.
- **Metrics:** `3D_OrganoidSeg/baselines_fair_20261030/eval_masks_shared.py` gives mIoU, per-class IoU / recall / precision and per-image confusion counts.

## Reporting (as decided by the author)
- These three models are **not** added to the area-error figure.
- They appear in a **table and a bar chart** with all CRC models (VitaFormer, nnU-Net, SegFormer-B2, OrganoidViT, U-Net, U-Net++, DeepLabv3):
  - test mIoU and per-class IoU (mean ± SD over 3 seeds, seeds shown as dots);
  - Dead recall / precision;
  - median per-image absolute relative area error for Alive / Dead / Bubble. This is the source paper's endpoint. It is computed from the same confusion counts (pred = tp + fp, GT = tp + fn), on images whose GT contains the class.
- **Descriptive only:** no new bootstrap contrasts. Weak results (for example the earlier late-training collapse of U-Net) are reported as they are, with training curves in the Supplement.

## Not done
No change to the scripts, hyperparameters, checkpoint rule or data split. No pretrained variants. No TTA.
