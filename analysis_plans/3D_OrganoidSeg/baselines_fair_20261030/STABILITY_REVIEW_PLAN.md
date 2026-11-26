# SegFormer-B2 numerical-stability review (plan 2026-10-31; approved by the author 2026-11-02)

Written at the author's request. It is a separate experiment from the frozen P0-B and P1-S protocols. Whether to run it is decided after P1-S finishes.

## 1. Facts so far (nothing re-run)
- **Baseline seed 42:** validation mIoU 79.23 at iteration 19,000, 59.96 at 20,000, then 24.87 (all background) from 21,000 to 40,000. The first NaN training loss is logged at iteration 20,200 (last finite value at 20,100: 0.0136), and the loss is NaN in every log line after that.
- **Baseline seed 3407:** validation 80.95 at iteration 28,000, then 24.87 from 29,000. The first NaN loss is logged at iteration 28,400 (last finite value at 28,300: 0.0101).
- **Baseline seed 2026:** no NaN.
- **Primary results** (frozen protocol, best-validation checkpoint, all taken before divergence): iterations 18k, 26k and 32k; test mIoU 83.60, 82.32 and 82.83.
- **Kept untouched:** logs (`segformer_b2_runs/logs/seed*.full.log`), `val_history.json`, `best_val_miou.pth`, `test_summary.json` and `test_masks/`.
- **What cannot be recovered:** the baseline saved no final weights and did not log the GradScaler scale. So it is unknown whether the weights themselves became non-finite, or only the fp16 forward pass overflowed. P1-S logs both.
- **Numerical regime of VitaFormer** (`train_v2_stratified.py`): fp32 (no autocast) with `clip_grad_norm_(1.0)`. The baselines therefore currently run under a less stable regime than VitaFormer.

## 2. Why "NaN only under-estimates SegFormer" is not assumed
Divergence removes the low-learning-rate second half of training for 2 seeds. That could have lowered their best validation checkpoint. But checkpoint selection on validation, and seed-level variance, mean the direction of the effect on test is not known until it is measured.

## 3. Proposed experiment (to be fixed before any run; settings chosen a priori, not tuned)
- **Arms:** SegFormer-B2 baseline and P1-S, each with **all 3 seeds** (6 runs), whether or not a given seed diverged originally.
- **The only change in both arms:**
  - (a) autocast dtype bfloat16 instead of float16 (same exponent range as fp32; GradScaler not needed);
  - (b) `clip_grad_norm_(max_norm=1.0)`, which is VitaFormer's own value.

  Nothing else changes: optimiser, schedule, data, augmentation, validation, checkpoint rule.
- **Settings are fixed a priori,** not chosen by any comparison. Train and validation data are used only for the usual best-checkpoint selection and for a pre-declared acceptance check: no non-finite training loss. Test is evaluated once per run, after training.
- **Reporting:**
  - The original (frozen-protocol) results stay the primary results.
  - The stability arm is reported as a sensitivity analysis: "Does fixing the numerical instability change the baseline-vs-VitaFormer and P1-S conclusions?"
  - All 6 runs are reported.
- **Optional:** a VitaFormer reference already exists (fp32 + clipping); nothing is re-run for it.
- **Cost:** about 6 × 4.5 GPU-hours if run alone (seed 2026 alone took 4.4 h); about 9.5 h each when 2 share a GPU.

## 4. Approval and execution details (2026-11-02, fixed before any stability run)
- **Approved:** both arms × 3 seeds, bf16 + `clip_grad_norm_(1.0)`. All runs are kept. The re-evaluation covers Rare-Dead across architectures, and VitaFormer vs the stable SegFormer-B2.
- **Script:** `train_segformer_b2_stable.py` (`--rare-dead` selects P1-S). It imports the original scripts unchanged.
- **The autocast dtype change applies everywhere the originals use autocast,** i.e. the training forward pass and the sliding-window validation/test inference. The originals' inference also ran under fp16 autocast.
- **No GradScaler, no skip-on-NaN logic.** Additionally logged: pre-clip gradient norm (median/max per 100 iterations), cumulative non-finite loss count, max |BN running statistic|, and `final.pth`.
- **Outputs:** `segformer_b2_stable_runs/{baseline,p1s}_seed{N}/`, logs in `segformer_b2_stable_runs/logs/`.
- **Evaluation:** `eval_masks_shared.py` into `shared_eval/segformer_b2_stable_{baseline,p1s}_seed{N}.json`. Then the same contrasts as `unified_comparison_20261102`, written to a new output directory; the original results are untouched.
- **Primary vs sensitivity:** the frozen-protocol (fp16) results remain the primary record. The stable arm is the sensitivity analysis that answers whether the instability changed the conclusions. Both are reported in full.
- **Smoke test** (scratch directory, 200 iterations per arm): no non-finite loss, BN max ≤ 0.7, validation and test inference ran.
- **Step 1 audits (done before training):**
  - `stability_review_20261102/checkpoint_audit.csv`: every original test metric came from `best_val_miou.pth`, which is fully finite in all 6 runs. P1-S `final.pth`: 379/379 tensors NaN. The baseline saved no final weights.
  - `test_metric_provenance.csv`.
  - fp16-vs-fp32 test re-inference of the 6 best checkpoints (`stability_audit_fp16_inference_20261102.py`; precision audit only).
