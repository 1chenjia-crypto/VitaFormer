# P1-S — SegFormer-B2 + the Rare-Dead bundle (protocol fixed 2026-10-31, before any P1-S training)

Approved by the author on 2026-10-31. The trigger was pre-specified in `P0B_PROTOCOL.md` §3: run P1-S if the SegFormer-B2 vs VitaFormer gap is < 3 pp. It is met: mIoU 82.92 vs 82.70, 3-seed means.

**Question.** Is the Dead-class behaviour (VitaFormer: higher and more stable Dead recall, lower precision) a property of the architecture, or of the Rare-Dead training strategy? Together with the existing VitaFormer Rare-Dead ablation (3 seeds, `--no_rare_dead`), this gives a 2 × 2 design: architecture × Rare-Dead.

## 1. What changes relative to the P0-B SegFormer-B2 baseline
Only the Rare-Dead bundle. It is the exact set of 6 interventions that VitaFormer's `--no_rare_dead` switch removes (`vco/organoid_seg/train_v2_stratified.py`), with VitaFormer's run values (`ablation_no_bam_no_mpm_seed42/train.log`).

| # | Intervention | VitaFormer value | In P1-S |
|---|---|---|---|
| 1 | Focal loss | γ = 2 | Focal γ = 2. It replaces the baseline's weighted CE. |
| 2 | Class weights | `compute_class_weights`, dead_weight_min 10 → [0.5, 0.5, 10.0, 0.9007] | The same weights, reproduced with VitaFormer's own function on VitaFormer's own train ordering. |
| 3 | 5-FU Dead oversampling | ×3 (7 images → +14 copies) | Same: train list has 495 entries. |
| 4 | Sparse-Dead zoom crop | p 0.8 if 0 < Dead px < 5,000; window crop/2, ×2 upsample | Same rule at crop 512 (window 256). Applies to 1 training image. |
| 5 | Dead-centre crop | p 0.6, jitter crop/4, ≥ 20 Dead px, 20 retries | Same, crop 512. |
| 6 | Minority-centred crop | p 0.4 | Same. |

- **Crop priority** is as in `dataset_v2.PatchOrganoidDataset`: zoom → Dead-centre → minority → random.
  - The zoom crop is taken at native resolution.
  - The other crops are taken after the baseline's scale jitter (0.5–2.0) and flip.
- **Not transferred**, because they are not part of the Rare-Dead bundle:
  - VitaFormer's base loss term (0.6 · Tversky);
  - VitaFormer's base augmentations (rot90, vertical flip, brightness/contrast/gamma, elastic);
  - VitaFormer's crop size 224, batch 16, optimiser and schedule.
- **Unchanged from the baseline** (imported from `train_segformer_b2.py`, which is not modified):
  - model and weights loading with hard checks;
  - AdamW 6e-5 (head ×10), warmup 1,500 iterations then poly decay, batch 8, 40,000 iterations;
  - **fp16 AMP without gradient clipping** (the frozen baseline setting; see §3);
  - validation every 1,000 iterations, sliding window 512/256;
  - best-validation-mIoU checkpoint;
  - single test pass with masks saved.
- **Logging additions only:** the GradScaler scale is logged every 100 iterations, and `final.pth` is saved. These make a possible NaN divergence diagnosable.
- **Smoke test** (CPU, 400 samples, scratch directory):
  - crop types: random 218 / minority 133 / Dead-centre 49;
  - mean Dead pixel fraction 0.0032 (VitaFormer's first batch: 0.0042);
  - the zoom branch fires in 39 of 50 draws on the sparse image.

## 2. Seeds, GPUs, evaluation
- Seeds 42, 3407, 2026. All three are run and reported.
- **GPUs:** queue launcher `p1s_queue.sh`.
  - GPUs 2 and 4 (another user's reserved memory) are never used.
  - A GPU counts as idle only if all 10 samples over 5 minutes show memory ≤ 3,000 MB and utilisation ≤ 5%.
  - The running nnU-Net jobs (GPUs 3, 5, 7) and other users' jobs are never shared.
  - At most 2 seeds are launched on one idle GPU, as in the baseline.
- **Evaluation:**
  - `eval_masks_shared.py` on `test_masks/`;
  - the same 3-seed paired hierarchical bootstrap as all other methods;
  - endpoints: mIoU, per-class IoU, Dead recall and precision, cross-seed SD.

## 3. Numerical stability
P1-S keeps exactly the baseline's numerics (fp16 AMP, no clipping), so that the P1-S vs baseline contrast changes only Rare-Dead. If P1-S diverges, that is reported as is, with its onset iteration. No setting is changed mid-experiment. Any stability fix is a separate, pre-specified experiment that applies equally to the baseline and P1-S: see `STABILITY_REVIEW_PLAN.md`.
- 2026-10-31 15:34: `p1s_post_eval.sh` (nohup) evaluates each P1-S seed with the shared evaluator as soon as its test_summary.json exists.
