# Stage 39: frozen SegFormer-B2 on DeepOrganoid (plan, 2026-11-09, written before any SegFormer DeepOrganoid inference; awaiting author approval)

## 0. Question
SegFormer-B2 is tied with M2 on CRC (82.92 vs 83.34, n.s.). Does its segmentation-derived readout on DeepOrganoid behave like
M2 (E3 stable across 3 training seeds) or like nnU-Net (E3 seed-unstable)? This is the most direct reviewer question for Fig 6
("same in-domain accuracy -> same functional readout?"). Inference + analysis only. No training, no change to any model.

Disclosure: SegFormer has never been run on DeepOrganoid (grep of analysis_ctg_alignment/ finds no SegFormer output).
M2 (Stage 36) and nnU-Net (Stages 35/37/38A) values are already known to the analyst.

## 1. Models (checkpoints frozen; SHA256 in frozen_hashes_pre_plan.txt)
- **Primary: SegFormer-B2 original** `baselines_fair_20261030/segformer_b2_runs/seed{42,3407,2026}/best_val_miou.pth`
  (the version in the paper's main CRC table: test 83.60 / 82.32 / 82.83, mean 82.92). Best-val checkpoints predate the
  fp16 NaN events (fp32 re-inference audit 11-02: >= 99.995% pixel agreement).
- Stable SegFormer-B2 (81.81): **not run** (amendment 11-09 after review; no further version search).
- Not included: P1-S (Rare-Dead SegFormer) - not a question the paper asks.

## 2. Input and inference
- Images: the same 3,456 DeepOrganoid IXM_BestFocus wells as Stages 31/35/36, via `extract_full_features.load_new()`
  (per-image p1-p99 stretch to uint8 x3, native size). Identical input to M2 and nnU-Net.
- Inference: SegFormer's own CRC test function `predict()` imported **unchanged** from `train_segformer_b2.py`
  (reflect pad to >= 512, ImageNet mean/std, 512/256 sliding window, fp16 autocast, mean softmax, argmax). No scale change,
  no TTA, no threshold. This is the native configuration, i.e. the analogue of nnU-Net V0 (no post-hoc config search here).
- Non-finite guard (pre-specified): count non-finite logits per image; if any, that image is re-run in fp32 and logged.
  The count is reported.
- Features: the Stage 36 feature extractor (`legacy.feats`) on the hard masks, unchanged. Masks + overlays saved.
- Script: copy of `scripts/run_m2_deeporganoid_p1p99_20261109.py` with only the model loader / predict call swapped.
- Output: this directory only (`segformer_seed{42,3407,2026}/`). Nothing overwritten.

## 3. QC gate (descriptive, as Stage 35)
Border / corner foreground metrics + contact sheets (CTG=0 wells, DMSO wells). Reported; does not change any setting.
A failing gate is reported as "QC failed" and the analysis still runs.

## 4. Endpoints (unchanged from Stage 35/37 and 38B)
- Stage 37 script copied with only the model map / paths / output dir changed: E1 (LOCO Ridge CTG, core), E1b, E1c, E2,
  E3 (within-drug line ranking), E3b; same cohorts, bootstrap seeds, permutation settings.
- Stage 38B script copied the same way: general-sensitivity baseline G, partial correlation, held-out increment.
- Reproduction checks: M2 s42 must reproduce E1 0.769 and E3 0.717 exactly; otherwise stop and report.

## 5. Decision rules (fixed now)
- Primary display: per-seed E1 and E3 (with CI) for M2 / nnU-Net V0 / SegFormer-B2 -> Fig 6 table.
- Contrast M2 s42 - SegFormer s_k (paired, same draws): "M2 advantage" requires CI excluding 0 on E1 or E3, same direction
  excluding Organoid52, and same direction vs >= 2 of 3 SegFormer seeds. Symmetric rule for "SegFormer advantage". Else "equivalent".
- Reproducibility statement: "SegFormer E3 stable" if all 3 seeds have E3 CI lower bound > 0 and seed range <= M2's range x 2
  (M2 range 0.705-0.745 = 0.04 -> threshold 0.08); "unstable" if >= 1 seed has CI including 0. Otherwise "intermediate".

## 6. Pre-stated interpretation (both directions will be written)
- SegFormer stable and equivalent to M2 -> Fig 6 reports plainly which models gave stable readouts (M2 and SegFormer) and
  which did not; readout stability is not specific to M2. No reframing to emphasise nnU-Net weakness.
- SegFormer unstable -> "SegFormer-B2, with CRC accuracy close to M2 (82.92 vs 83.34, n.s.), and nnU-Net, with higher CRC
  accuracy, gave less reproducible rankings". No claim that M2 is necessary or that architecture causes it.
- In all cases: E1 results reported side by side; no "SegFormer failed" wording.

## 7. Compute
3 x 3,456 images. One free GPU, one job per GPU, OMP_NUM_THREADS=4. Est. < 1 h per seed.
