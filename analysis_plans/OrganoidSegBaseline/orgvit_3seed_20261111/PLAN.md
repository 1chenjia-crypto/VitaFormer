# OrganoidViT reproduction: three training seeds on CRC (post hoc supplement) — plan

**Written:** 2026-11-11 10:30 CST, before any new training. The author approved option A on 2026-11-11.
**Status:** post hoc supplement. Added after all functional results were known, so it is labelled post hoc.
OrganoidViT is evaluated on CRC only. It enters no functional analysis (Figures 3–5 are unchanged).

## Correction found while preparing this plan
The existing reproduction (`runs/orgvit_b0_1446_40k_cond`) was trained with **seed 0**: its training log says
`Set random seed to 0, deterministic: False`. It was not seed 42. Every file and document that labels it "s42" is wrong and will be corrected
to "seed 0". The seed-0 result (test mIoU 75.30) is kept as a historical record only. It is not reported in the main text and not specifically in the Supplement.

## Training (identical to the seed-0 reproduction; only the seed changes)
- **Seeds:** 42, 3407, 2026, the same as VitaFormer, SegFormer-B2 and nnU-Net.
- **Config:** `Orgvit/local_configs/2DSegFormer/organoid_voc_b0_1446_40k_cond.py`, unchanged (SHA256 recorded in `config_sha256.txt`).
  - Model: myt_b0 + UHead, mit_b0 ImageNet weights.
  - Loss: cross-entropy.
  - Optimiser: Adam, lr 5e-6, poly schedule.
  - Iterations: 40k.
  - Batch: 4 on **one GPU**.
  - Augmentation: resize 1446 × [0.5, 2], crop 512, flip, rotate 45°, photometric distortion.
  - Precision: fp32, as in the seed-0 run.
- **Command per seed:** `tools/train.py <cfg> --work-dir runs/orgvit_b0_1446_40k_cond_seed{S} --seed {S}`. No `--deterministic`, matching seed 0.
- **GPUs:**
  - One seed per GPU. No multi-GPU or distributed training, so the effective batch stays 4.
  - Only GPUs that are idle are used. On GPU-V100-2, GPU1 belongs to another user and is not used.
  - The GPU model and precision of each seed are recorded in `SEED_RECORD.md`.
  - Seed 0 ran on a Tesla V100S-PCIE-32GB. Seeds on a different GPU model will be noted in Methods.
- **Data:** `train_cond` / `val_cond` / `test_cond` (481 / 164 / 167). Verified on 2026-11-11 to be identical, image by image, to the split of all other models. Test-split SHA256: `483b0a28…`.
- **Environment:** the original `/tmp/venv-OrganoidSegbaseline` was wiped. It is rebuilt as `venv-orgvit-py36`: Python 3.6.8, torch 1.7.1+cu101, mmcv-full 1.2.7, timm 0.3.2, the repository's own mmseg. Package versions are in `env_freeze.txt`.

## Checkpoint rule (unchanged from the reproduction)
- Validation mIoU on `val_cond` every 2,000 iterations (the mmseg evaluation hook). The checkpoint with the highest validation mIoU is selected.
- The test set plays no part in selection.
- Methods will state that OrganoidViT uses the best validation checkpoint, whereas VitaFormer and nnU-Net use the final checkpoint.

## Environment check (before any new seed is evaluated)
- **Check:** with the new environment, the seed-0 `iter_32000.pth` is run once through the original pipeline (`tools/test.py`, whole image at 1446, mmseg metrics).
- **Pass criterion:** test mIoU 75.30 ± 0.05 pp.
- **Also checked:** the exported seed-0 masks are run through the shared evaluator. Pred and GT areas must match the existing CSV (`test_best_iter_32000_relative_area_errors.csv`), with ≤ 0.1% of pixels differing per image.
- **If the check fails:** stop and report; new seeds are not evaluated.

## Test evaluation of new seeds (shared evaluator only)
- **Masks:** for the selected checkpoint, label masks for the 167 test images are exported as PNG files. They are produced by `mmseg.apis.inference_segmentor` with the config's test pipeline (whole image, resize 1446, logits resized to the original size), which is the same computation as `tools/test.py`.
- **Metrics:** `3D_OrganoidSeg/baselines_fair_20261030/eval_masks_shared.py` computes mIoU, per-class IoU / recall / precision and per-image confusion counts.
- **Reported per seed:** mIoU, per-class IoU, selected iteration, and the validation mIoU at that iteration.
- **3-seed summary:** mean ± SD. This replaces "75.30" in the main text.

## Downstream updates (after all 3 seeds)
1. **Area-error analysis:** `crc_area_error_posthoc_20261111` is re-run with OrganoidViT changed from the seed-0 CSV to the 3 shared-evaluator JSONs.
   - Output goes to a new directory; the existing results are kept.
   - OrganoidViT remains a reference: it is shown with 3 seeds and IQR, with no new bootstrap contrasts. Adding contrasts would be a change to the plan and needs the author's approval.
2. **Figures:** Figure 2A (mIoU per seed) and the area-error figure are updated.
3. **Tables:** Table S1 and the early-baseline table gain OrganoidViT's three seeds.
4. **Relabelling:** files currently labelled "OrganoidViT (old, s42)" are relabelled seed 0.

## Not done
No change to the config, the learning rate, the batch or the checkpoint rule. No TTA. No functional or external evaluation of OrganoidViT.

---
## Amendment 1 — 2026-11-11 13:25 CST (author approved; written before any A800 run)
**Reason:** at 12:16 both V100S nodes (GPU-V100-1/-2) became unreachable (ping 100% loss). Seeds 42 and 3407 stalled at iter 2,200 and 1,300 and are treated as failed (directories kept, not used). The P100 nodes (16 GB) cannot hold the 18.8 GB training peak without changing the recipe.

**Change (hardware/software build only; the recipe is unchanged):**
- All three seeds (42, 3407, 2026) are trained **from scratch** on **gpu-host** (NVIDIA A800-SXM4-80GB). No resume from the stalled runs.
- **Build:** torch **1.7.1+cu110** (the same torch version, built for CUDA 11.0, because cu101 has no sm_80 kernels) and mmcv-full 1.2.7 (cu110 build). Separate env `venv-orgvit-py36-cu110`. The cu101 env is kept.
- **Precision:** TF32 is **disabled** (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`) through a launcher wrapper, so computation is true fp32 as on V100S. The flag state is written to each run's log.
- **GPU sharing (author approved):** no A800 GPU is idle. Each seed runs on its own GPU, shared with other users' processes, chosen among GPUs with ≥ 40 GB free memory. GPU index, other-process count and free memory at launch are recorded in `SEED_RECORD.md`.
- **New work directories:** `runs/orgvit_b0_1446_40k_cond_seed{S}_a800`, so that the stalled V100 runs (if their node comes back) cannot write into them.
- **Environment check on A800 (gate):** the seed-0 `iter_32000.pth` must reproduce test mIoU 75.30 ± 0.05 through `tools/test.py`. Shared-evaluator per-image areas are compared with the seed-0 CSV, and the differing fraction is reported. A small difference is expected from the different GPU / CUDA build; the gate is the mIoU tolerance plus ≤ 0.1% of pixels changed per image.
- **If a run crashes** (for example, out of memory caused by other users): restart that seed from scratch and record it.
- **Methods:** OrganoidViT seeds trained on A800 (torch 1.7.1+cu110, TF32 disabled). Note that VitaFormer M2 seeds were also trained on mixed hardware (seed 3407 on A800, seeds 42 and 2026 on V100S).
- **Amendment 1b (2026-11-11 13:50, author approved):** the free-memory threshold for an A800 GPU is lowered from ≥ 40 GB to **≥ 30 GB** (training needs about 19 GB). The reason: at 13:48 only one GPU had ≥ 40 GB free. The three seeds run simultaneously, one per GPU. Operational only; the recipe is unchanged.
- **Amendment 2 (2026-11-11 14:28, author approved):** if a seed crashes or stalls (training log not updated for more than 60 min), the watchdog **resumes it from its latest checkpoint** (`--resume-from latest.pth`, saved every 2,000 iterations), in the same work directory. It does not restart from scratch, which it does only if no checkpoint exists yet.
  - A resumed run's data-order and augmentation random stream after the resume point differs from an uninterrupted run. Every resume (seed, iteration, time) is recorded in `SEED_RECORD.md` and will be stated in Methods.
  - `select_best.py` now de-duplicates validation records by iteration. On an uninterrupted run its result is identical (verified on seed 0: iter 32,000, 0.7352).
  - The model-selection rule (best validation mIoU, evaluated every 2k iterations) is unchanged.
  - Runner: `run_seed_a800_v2.sh` (the running seeds keep using the unchanged `run_seed_a800.sh`).
  - Resume function test (scratch copy of a 2,000-iteration checkpoint, A800): resumed at iteration 2,000 and continued to 2,100 with the scheduled learning rate. The test copy was deleted.
