# Stage G — Dead operating point vs discrimination (CRC VALIDATION only) — DRAFT, awaiting user approval
Drafted 2026-11-06, before any probability export or PR computation. No training, no GT change, no new architecture,
no test-set data read. M2 stays frozen; anything found here is a candidate inference-rule revision that needs separate approval.

## Question
Are the extra Dead detections from Rare-Dead (a) a Recall–Precision trade-off that a Dead threshold can control, or
(b) a sign of weak Dead/Background discrimination (lower precision at the same recall, lower AUPRC)?

## Already known (seen before this plan; stated so it cannot be re-discovered as "new")
- Test, 3 seeds, Full vs Full-noRD: Dead recall +11.6 [2.4, 23.9]; precision −2.9 n.s.; Dead IoU +2.5 n.s.; mIoU +0.66 n.s.
  Full-noRD Dead precision 57.1 vs nnU-Net 72.1.
- Validation (Stage D, final.pth): M2 Dead P/R 54.0/76.8, nnU-Net 65.0/65.6; far-background Dead FP M2 ≈ M0 ≈ 1.5× nnU-Net.
- LQA B2: excluding disputed regions widens the M0–nnU-Net Dead gap.

## Step 1 — Rare-Dead ablation audit (files only)
- Tabulate every existing Rare-Dead on/off result (test 3-seed contrasts, any validation histories) with source paths.
- List same-architecture on/off checkpoint pairs and verify step = 40000 + sha256:
  Full (expFinal_fromscratch / full_model_seed3407 / full_model_seed2026) vs Full-noRD (ablation_no_rare_dead_seed*).
  Expected: no M0-noRD or M2-noRD exists → no same-architecture pair for M2.
- If all 6 Full / Full-noRD final.pth are step 40000, include them in Step 2 as the **direct Rare-Dead test**
  (same architecture, Rare-Dead the only difference). If any is not, report and drop that arm (no substitution).

## Step 2 — Probability export (validation, 164 images)
| Arm | Checkpoint | Inference |
|---|---|---|
| M2 ×3 (main) | final.pth (Stage D hashes) | infer_hard unchanged, but return the window-averaged softmax before argmax |
| M0 ×3 (secondary) | final.pth | same |
| Full ×3, Full-noRD ×3 (Rare-Dead test, if Step 1 passes) | final.pth | same (Full model class, as trained) |
| nnU-Net ×3 | run2 checkpoint_final | `nnUNetv2_predict --save_probabilities --disable_tta` (no-TTA = primary test protocol) on fold_0 val images |
- Gate: argmax of exported probabilities must equal the existing Stage D val masks (VF arms, ≥ 99.999%). nnU-Net: agreement
  with its fold_0/validation masks reported (those used mirroring, so not expected identical); stop if < 99%.
- Stored per image: p_dead (float16) + argmax over {bg, alive, bubble} (uint8). New directory only.

## Step 3 — Analysis (pre-specified)
**Decision rule D(t):** pixel = Dead if p_dead ≥ t, else argmax over the other three classes. Grid t = 0.05 … 0.95 by 0.01,
plus plain argmax as reference. One t shared across seeds.

**Per arm × seed × t (pooled pixels, valid GT):** Dead precision / recall / IoU, Alive IoU, Bubble IoU, 4-class mIoU,
far-background Dead FP px (A0 FARBG definition, unchanged).

**Threshold-free:**
- Pixel Dead AUPRC (Dead vs rest, p_dead; 1,001-bin histogram).
- Precision at matched recall: at nnU-Net's own argmax recall (seed-matched), and at fixed recall 60 / 65 / 70 / 75.
- Object level (8-connectivity): GT Dead object detected if ≥ 50% of its pixels are predicted Dead; predicted Dead component
  is a false object if < 10% of its pixels are GT Dead. Object recall / precision at argmax and at t*.

**Threshold selection (validation only):** t* = argmax of the 3-seed-mean mIoU; t_dead = argmax of 3-seed-mean Dead IoU
(secondary, reported, not adoptable unless it equals t* within 0.02).

**Statistics:** hierarchical paired bootstrap as in unified_comparison_20261102 (3 seed positions + physical val wells,
10,000 replicates, RNG 20261107). Contrasts:
1. M2 − nnU-Net: precision at nnU-Net recall; AUPRC.
2. M2 at t* − M2 at argmax: Dead IoU, mIoU, Dead precision, far-bg Dead FP.
3. Full − Full-noRD: AUPRC; precision at matched recall (the Rare-Dead discrimination test).
4. M2 − M0: AUPRC (does the CNN branch change discrimination).
5. Sensitivity: rerun 1–3 with LQA review_val_20261106 disputed boxes set to ignore, only if the user's review sheet is
   filled; boxes of all categories, regardless of which model they came from.

## Pre-specified interpretation
- **Operating point (a):** M2 precision at nnU-Net recall within 5 pp of nnU-Net AND AUPRC difference CI includes 0 or
  |Δ| < 0.03.
- **Discrimination (b):** precision gap at matched recall ≥ 10 pp with CI excluding 0, or AUPRC lower with CI excluding 0.
- In between → "mixed"; reported as such.
- **Rare-Dead itself:** if Full and Full-noRD have AUPRC within 0.03 (CI includes 0) → Rare-Dead only moves the
  operating point; if Full AUPRC is lower (CI excludes 0) → Rare-Dead degrades discrimination.
- **Threshold "solves it" (adoption candidate):** at t*, M2 Dead IoU ≥ +2.0 pp vs argmax, mIoU ≥ −0.1, CI of the Dead IoU
  gain excluding 0, 3/3 seeds positive.

## Step 4 — Decision proposal (no action without approval)
- Threshold solves it → propose t* as a declared post-hoc inference revision: apply once to test (separate stage, new dir),
  plus a DeepOrganoid check of how much functional (Alive) area changes. Frozen argmax results stay reported.
- Threshold does not solve it AND Full vs Full-noRD shows Rare-Dead lowers AUPRC → propose an M2 No-Rare controlled
  training (3 seeds, final.pth, otherwise identical recipe) as a plan only.
- Threshold does not solve it AND Rare-Dead does not lower AUPRC → no No-Rare training; report Dead/Background
  discrimination as a known limitation of the architecture.

## Limitations stated in advance
- Pixel AUPRC is dominated by large Dead objects; the object-level metrics compensate partly.
- Unlabelled real Dead counts as FP for every model; the LQA sensitivity analysis is the only correction.
- VF softmax (window-averaged) and nnU-Net softmax are calibrated differently; matched-recall and AUPRC comparisons do not
  depend on calibration, cross-model threshold values do and are not compared.
- Validation was used for Stage C/D checkpoint-rule decisions; it is not untouched.

## Cost and scheduling
Inference only: ~10–15 min per VF arm-seed, a few min per nnU-Net seed; starts after Stage F (test inference) finishes,
on GPUs free at that time. Outputs: stageG_dead_pr_val/ only.
