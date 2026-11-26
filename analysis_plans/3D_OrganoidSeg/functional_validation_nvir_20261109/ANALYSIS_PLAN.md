# NViR independent ATP validation (plan, 2026-11-09, written before any model inference on NViR; awaiting author approval)

## Amendment 1 (11-09, after review, before any inference)
- Positioning: **extension validation across tissue type and imaging conditions** (liver spheroids, 4 microscopes), not a second
  tumour drug-response validation; it cannot validate drug-sensitivity ranking transfer or the Dead class. A negative result is
  reported as a boundary of applicability.
- Wording: "frozen models with target-domain scale selection" (network weights unchanged, no ATP used, but target-domain
  reference masks are used to pick the input scale) - **never "zero-shot"**.
- Reference masks: produced by the authors' U-Net (trained on ~1,000 manual labels, reported val IoU > 0.9; per the paper,
  to be confirmed in the PDF) -> "author-provided reference masks", not manual ground truth. Scale selection may bias models
  toward that algorithm's boundaries; disclosed. Transparency: N1 for all 5 candidate scales is reported in a supplementary
  table, the pre-selected scale remains the primary result.
- Fairness: M2, nnU-Net and SegFormer-B2 (all existing weights, 3 seeds each, no retraining) get the identical scale rule
  (seed 42, train_val only, same candidate grid, same criterion).
- N2: our features (Set A geometry; Set A+B geometry+intensity) are **not** a reproduction of the paper's CVE (different
  features, model class only shared). Test rows are never used for feature choice or tuning (sklearn defaults; Ridge alpha by
  CV on train_val). Paper values are inserted into results only after the author verifies them in the PDF.
- Scope: no other external dataset is extended; no further model versions.

## 0. Question
Do frozen segmentation-derived readouts (spheroid area, shape) track an independent destructive viability assay (CellTiter-Glo 3D
ATP) in a different organoid type (human liver spheroids), lab, and 4 microscopes? Inference + analysis only; no model is
trained or changed. Comparison with the source paper is descriptive (its own split and metric), per the project rule.

## 1. Data (Zenodo 15374443; read-only at external_datasets/NViR_zenodo15374443/)
- Source paper: Dubinsky et al., "Non-invasive quantification of viability in spheroids using deep learning" (NViR),
  Front. Bioeng. Biotechnol. 2026, doi 10.3389/fbioe.2026.1797474; code github.com/DanielDubinsky/atp_paper.
  ⚠️ Paper read via web extraction only (tail partly summarised); the author must confirm Table 1/2 values in the PDF.
- `data_labeled.csv`: 2,359 wells, one centred bright-field field per well (10x), ATP (`value`, raw), compound, concentration,
  treatment, plate, day, `split`, `domain`. All 56 (location, microscope, study, plate, day) groups contain medium controls (checked).
- Split (author's, used as is): train_val 777 = domain A (Echo Rebel) studies before 2022-07-28; test 1,582 =
  A-test 245, B Echo Revolve 712, C Cytation 5 245, D Thorlabs 380. (CSV domain labels A_echo_rebel / B_echo_revolve_v0 /
  A_cytation_v0 / A_thorlabs mapped to paper A/B/C/D by counts and microscope names.)
- Images used: `images_labeled_cropped_0.2/` (author's crop: square of side 0.4 x min(H,W) centred on the spheroid bbox;
  960 or 1216 px, grayscale). `images_labeled_masks/`: binary spheroid masks from the authors' U-Net (reported val IoU > 0.9)
  -> called **reference masks**, not ground truth.
- Not used: unlabeled set, chromalive, DILI/proctor IC50 benchmark, pathologist scores (exploratory at most).

## 2. Label (author's code, atp/preprocess.py `normalize_by_compound`)
normalized ATP = value / median(value of compound == "medium") within (location, microscope, study, plate, day).
All test rows kept, including controls (as in scripts/evaluate.py).

## 3. Models and representations
- **M2** seeds 42 / 3407 / 2026 (frozen, final.pth; primary).
- nnU-Net run-2 seeds 42 / 3407 / 2026 (frozen; native V0 inference as Stage 35).
- SegFormer-B2 original seeds 42 / 3407 / 2026 (same checkpoints as Stage 39).
- Otsu: global Otsu on the crop + largest connected component (no tuning).
- Reference masks: authors' U-Net masks cropped identically (upper reference for mask-geometry readouts).
- Foreground = all non-background classes except Bubble (Alive + Dead). Dead fraction reported separately (exploratory).

## 4. Input scale (the only free choice; fixed by a rule that never sees ATP)
Large single objects fail at native resolution (Brain lesson). Candidate input sides S in {256, 384, 512, 768, native}
(crop resized with area interpolation; masks resized back with nearest). For each model family, using **seed 42 only and
train_val rows only**, choose the S with the highest median foreground Dice vs the reference masks; ties -> smaller S.
That S is then fixed for all seeds and all test rows. Otsu runs at native crop. Chosen S and the Dice table are reported.
ATP is not loaded until S is fixed for every model (enforced: the scale script does not read the `value` column).

## 5. Features (fixed)
- Set A (geometry, primary): foreground area (in original-pixel units), equivalent diameter, circularity 4piA/P^2, solidity,
  eccentricity, number of connected components (>= 50 px at original scale).
- Set B (adds intensity, secondary): mean intensity inside foreground, mean intensity in a ring of width 0.1·eq. diameter
  outside it, their ratio (analogue of the paper's "relative brightness").
- Control normalisation: each feature / median of the medium wells in the same 5-key group (mirrors the label).

## 6. Endpoints
- **N1 (primary, no ATP fitting):** Spearman rho(normalized foreground area, normalized ATP) on test rows, per domain
  (A-test, B, C, D) and pooled. 95% CI by cluster bootstrap over plate groups (2,000 draws, seed 20261109).
- **N1b:** dose-direction concordance. For each (study, compound, day) with >= 4 concentrations and |rho(conc, ATP)| >= 0.5:
  sign(rho(conc, area)) == sign(rho(conc, ATP)); % concordant.
- **N2 (supervised, comparable to paper):** GradientBoostingRegressor (sklearn defaults, random_state 0; the paper's baseline
  model class) and Ridge (alpha by 5-fold CV on train_val; secondary), fit on train_val Set A (Set A+B secondary),
  predict test; R^2 and MAE per domain and pooled, computed as in the authors' evaluate.py.
  Descriptive comparison to paper Table 1/2 (R^2 A/B/C/D: CVE .16/.56/.32/.68; Pathology+CVE .56/.70/.50/.62;
  NViR .83/.84/.70/.80). Not paired (author predictions unavailable); NViR is an end-to-end CNN trained on ATP.
- **N3 (reproducibility):** seed spread of N1 and N2 per model family; foreground Dice vs reference masks per domain.

## 7. Decision rules (fixed now)
- "Area readout tracks ATP in independent liver-spheroid data": pooled test N1 CI excludes 0, same sign in >= 3 of 4 domains,
  in >= 2 of 3 M2 seeds. Otherwise reported as not established.
- M2 vs Otsu and M2 vs reference masks: paired bootstrap delta-|rho| (same draws); claim only if CI excludes 0.
- M2 vs nnU-Net / SegFormer: Stage 37 rule (CI excludes 0, same direction vs >= 2 of 3 comparator seeds), else "equivalent".
- N2 wording: "between CVE and NViR", "comparable to CVE", etc. by position of the 3-seed mean; never "outperforms NViR"
  unless every domain's R^2 exceeds NViR's mean + SD, which would still be descriptive.
- Dead class: liver spheroids are dense and dark; Dead activation is reported but not interpreted as cell death.

## 8. What the analyst has already seen
CSV columns and counts; 2 cropped images and 1 reference mask (visual check of format); paper Tables 1-2 and code.
No model inference on NViR, no image-ATP analysis.

## 9. Outputs and compute
New directory only (`functional_validation_nvir_20261109/`); data directory read-only. Scale step: 777 images x 5 scales x
3 families (seed 42). Main: 2,359 images x 3 seeds x 3 families. One GPU, OMP_NUM_THREADS=4. Est. a few hours total.

## Amendment 2 (11-09 18:50, implementation details fixed before any NViR inference; no ATP read)
- Input: authors' 8-bit crops used as is (no intensity stretch), grayscale replicated to 3 channels.
- Resize: PIL BOX to S x S; masks back with NEAREST. Reference masks cropped with the authors' crop_spheroid.py geometry
  (verified: 2,359/2,359 sizes equal the image crops; visual contour check on 4 random images OK). Note: the reference masks
  exclude the loose fluffy periphery of some spheroids (boundary convention of the authors' U-Net).
- Otsu: foreground = darker side, holes filled, largest component. Feature normalisation 0/0 or x/0 -> 1.0.
- N2 Ridge = RidgeCV(alphas = 10^-3..10^3, 13 values, cv = 5) on standardised features.
- Supplementary scale table needs seed-42 test-row inference at all 5 scales (extra compute, run last).
- Scripts: scripts/nvir_{infer,select_scale,analysis}_20261109.py (hashes in frozen_hashes_pre_run.txt); queue run_queue.sh.
