# Stage 35: frozen nnU-Net v2 vs frozen VitaFormer on DeepOrganoid (plan, 2026-11-02, written before any nnU-Net DeepOrganoid inference)

Approved by the author on 2026-11-02. Question: does VitaFormer have an independent downstream drug-screen advantage over the stronger CRC segmenter (nnU-Net v2)? Both models are frozen. No training, no tuning on DeepOrganoid. CTG and drug results are not used to select checkpoints, thresholds or preprocessing.

## 1. Models (fixed now)
- **VitaFormer:** reused unchanged from Stage 31 (3,456 images each; masks and features exist; no rerun).
  - Simplified seed 42 is the primary VitaFormer representation.
  - Full seed 42 is a comparator.
- **nnU-Net v2:** `checkpoint_final.pth` of run 2 (`baselines_fair_20261030/nnunet/nnUNet_results_run2_seed{N}`), no TTA. This is the CRC primary setting, fixed before seeing CRC test results.
  - **Primary: seed 42**, matched to the VitaFormer seed-42 checkpoints used in Stage 31.
  - **Secondary (pre-specified, all reported): seeds 3407 and 2026**, as checkpoint-seed robustness. nnU-Net has 3 frozen seeds on DeepOrganoid; VitaFormer has only seed 42 there, because a VitaFormer DeepOrganoid rerun is not authorised.

## 2. Input and preprocessing (identical audit standard to Stage 31)
- **Same 3,456 IXM_BestFocus wells** (`train_resolved.csv`), same `load_new()`: per-image 1st–99th percentile stretch of the 16-bit image to uint8, replicated to 3 channels, native scale (682 × 682).
- **Model-internal normalisation** is each model's own frozen training-time step:
  - VitaFormer: fixed ImageNet mean/std, 224/112 sliding window, fp16 autocast, mean softmax.
  - nnU-Net: its plans' per-image per-channel z-score; patch 1024 × 1280; the 682 × 682 image is padded by nnU-Net; Gaussian-weighted sliding window, tile step 0.5; no mirroring.
- **Features:** the same `feats(prob, gray)` and border metrics as Stage 31, computed from each model's softmax in memory. Only hard masks and overlays are saved.
- **QC gate before any CTG analysis (Stage 28/31 standard):**
  - per-line foreground distribution, all-background and near-empty counts, Dead-positive wells;
  - border-touching foreground;
  - contact sheets of 9 lines across the CTG range, reviewed visually.
  - The gate is descriptive: no CTG endpoint is used to judge it, and no setting is changed after it. A failing gate is reported as such, and the analysis still runs (marked "QC failed").

## 3. Endpoints: Stage 31/32/34 methods unchanged; nnU-Net added as a representation
| # | Endpoint | Method (source) | Paired comparison |
|---|---|---|---|
| E1 | **CTG association (primary):** nested LOCO ridge, CORE NEG-delta → relative CTG, Pearson | Stage 31 `run_loco`; Stage 32 hierarchical draws (cell line → condition), 1,000, seed 20261029, rows aligned by sample_id | VF-simplified − nnU-Net s42 (same draws) |
| E1b | Area-only LOCO; RICH NEG-delta (secondary) | same | same |
| E1c | 200 within-line label permutations, nnU-Net s42 core | Stage 31 | — |
| E2 | **Dose–response consistency:** median dose-level CTG–morphology ρ; concordant fraction among strong-CTG trajectories | Stage 31 `dose_trajectories` on E1 OOF predictions; cell-line bootstrap 1,000 | paired Δ with shared cell-line draws |
| E3 | **Patient-line ranking:** within-drug mean ρ of model-free area AUC (area relative to the line's NEG median) vs CTG AUC | Stage 34 `ranking_block`, drug bootstrap 2,000, permutation 10,000, seed 20261030 | VF-simplified − each nnU-Net seed (same drug draws) |
| E3b | Pooled potency Spearman (92 line × drug pairs) | Stage 34 | paired Δ, shared cell-line draws 2,000 |

- **Cohorts:** all nine lines (main) and **excluding Organoid52** (sensitivity, item 5). For E3, also excluding LPA (Stage 34 post-hoc flag, reported as such).
- **Representations in every table:**
  - VF-simplified s42 (reference for the paired Δ), VF-Full s42;
  - nnU-Net s42 / s3407 / s2026;
  - Stage 32 thresholds as context (local contrast, Otsu).

## 4. Decision rules (fixed now)
- **"VitaFormer has an independent downstream advantage over nnU-Net"** only if all three hold:
  - the paired Δ (VF-simplified − nnU-Net s42) CI excludes 0 in favour of VitaFormer on E1 (primary) or E3;
  - the same direction holds excluding Organoid52;
  - the direction holds against at least 2 of the 3 nnU-Net seeds.
- **"nnU-Net is better downstream":** the symmetric rule.
- **"Equivalent / no evidence of difference":** any other outcome. The paper then describes the workflow as segmenter-agnostic, with VitaFormer as one implementation.

## 5. Recorded domain differences (reported, not tuned)
- Normalisation: fixed ImageNet statistics vs per-image z-score.
- Window/patch: 224 vs 1024 × 1280; the image is smaller than nnU-Net's patch, so it is padded.
- Pretraining: BiomedCLIP vs none.
- CRC training images are colour RGB (2048 px), while DeepOrganoid images are grey replicated to 3 channels (682 px). Both models face the same shift; neither was adapted.
- External alive/dead semantics stay unvalidated for both models. Dead-positive wells are counted and reported only.
