# OrganoID GemcitabineScreen — external Dead-class validation against propidium iodide (PI)
# ANALYSIS_PLAN (draft 2026-11-09, awaiting author approval; written before any M2 result on this dataset)

Frozen main model: VitaFormer M2 (`model_freeze_m2_20261108/MODEL_FREEZE_M2_20261108.md`). Nothing here changes M2.
Results are reported whatever they show. Comparison only with the source paper (OrganoID), not with OrganoidViT.

## 0. Question
Does M2's **Dead** class (the CRC-trained 4-class head, never trained on this data) mark organoid death that an
independent dead-cell stain (PI) also reports? This is the first external check of the Dead class. All earlier external
sets only tested binary foreground.

## 1. Source and what was verified (2026-11-09, author code + released files; no M2 run yet)
- Matthews et al., OrganoID (PLoS Comput Biol 2022), Figure 3. Data: `external_datasets/OrganoID_Data/GemcitabineScreen/`
  (`BF/` 42 bright-field stacks, `PI/` 42 PI stacks, `OrganoIDProcessed/` author OrganoID instance labels and tracking).
  Code: `external_datasets/OrganoID/Publication/Figure3/` (`ProcessGemcitabineExperiment.py`, `3b-c/`, `3d/DoseResponseCurves.py`,
  `3e/`), together with the author output CSVs `PIMeasurements.csv`, `OrganoidMeasurements.csv` and `MTSAssay.csv`.
- Design, taken from the author code:
  - Each stack is 19 frames at 4 h intervals (0–72 h), 1022×1024, uint16. Wells are XY007–XY054.
  - `dosage = [inf, 0, None, 1000, 300, 100, 30, 10, 3][(XY-1)//6]` gives 7 groups (0, 3, 10, 30, 100, 300, 1000 nM
    gemcitabine) × 6 replicate wells = 42 wells. The groups XY001–006 (inf) and XY013–018 (None) are not in the data.
  - MTS viability: 7 doses × 3 separate wells (not the imaged wells).
- Author readouts:
  - Whole-image PI = Σ(PI/1000) per frame.
  - Per-organoid masked PI and area come from the OrganoID labels.
  - Dose response is read at **t = 52 h**, with a 4-parameter log-logistic fit (`fsigmoid`, dose 0 excluded). EC50 is
    reported for each readout. Effect-size heatmaps are at t = 72 h.
- **Already seen before this plan (author CSVs only, not M2):** I ran the authors' fit code on their CSVs. The resulting
  EC50 values are MTS 26.2 nM, PI 16.9 nM, OrganoID area 12.7 nM and tracked PI/area 24.5 nM.
  - The values in the paper figure must be confirmed by the author (user) before RESULTS are written.
  - Mean PI at t52 rises from 1.12e6 (0 nM) to about 1.5–1.7e6 (≥30 nM).
  - Well-level Spearman(PI, OrganoID area) at t52 = −0.57.
  - **So area alone already tracks PI.** This sets the bar the Dead class has to clear (§4, incremental test).

## 2. Models and inference (zero-shot only; protocol A)
- There is no Dead ground truth to train on, so protocol B (fine-tuning) cannot test the Dead class. Only zero-shot is used.
- Models: CRC M2 `final.pth`, seed 42 (primary) and seeds 3407 / 2026 (robustness); SHAs as in the freeze record.
  4 classes, frozen `infer_hard`: ImageNet normalisation, 224/112 sliding window, fp16 forward pass, softmax average,
  argmax. Dead probabilities are also saved.
- Input: each BF frame, uint16 → 8-bit by a per-frame p1–p99 linear stretch (same rule as the DeepOrganoid Stage 31/36
  pipeline), grey replicated to RGB.
- **Scale, fixed by a PI-blind and dose-blind rule before any PI analysis:**
  - Candidates: rescale factors {1.0, 0.75, 0.5, 0.25}.
  - Run M2 seed 42 on the **t = 0 frame of all 42 wells only**. This is pre-treatment, so dose has no effect yet.
  - Choose the factor with the highest mean foreground Dice (Alive ∪ Dead vs author OrganoID label > 0).
  - Ties within 0.005 go to the larger factor.
  - The predicted masks are resized back to native resolution with nearest-neighbour interpolation.
  - That one factor is then used for all frames and seeds. The selection table is saved before any PI file is opened by the analysis.
- If the best t = 0 Dice is < 0.5, M2 does not find the organoids in this domain. Then the Dead analysis is still run and
  reported, but it is labelled "segmentation failed — Dead endpoints not interpretable", and no validation claim is made.

## 3. Readouts (per well × frame; 42 × 19 = 798)
From M2 (argmax unless stated):
- **DF = Dead fraction** = Dead px / (Alive + Dead px). Wells with 0 foreground pixels get DF = NA, and their number is reported.
- DP = mean softmax P(Dead) over the M2 foreground (threshold-free, secondary).
- FG area and the area fold change vs t = 0 of the same well (baseline readout, same model without using the Dead class).

From PI (the independent readout):
- **PI_img** = the author whole-image Σ(PI/1000).
  - Sanity check: it is recomputed from the TIFs and must match `PIMeasurements.csv` exactly for all 798 rows.
  - If it does not match, stop and fix the reader.
- PI_pos pixels = PI > T. T is the 99th percentile of PI pixel values in the dose-0 wells at t = 0, fixed before any M2
  comparison.

Comparators, so that we can see what the Dead class adds:
- (a) M2 FG area fold change.
- (b) author OrganoID area fold change (from the author labels).

## 4. Endpoints
- **Primary:** the well-level Spearman ρ between DF and PI_img at t = 52 h (the author time point), n = 42 wells, seed 42.
- Key secondary (all must be stated together):
  - S1 Incremental: partial Spearman ρ(DF, PI_img | M2 FG area fold change) at t52. Does Dead add information beyond shrinkage?
  - S2 Pixel level: AUROC of M2 P(Dead) for PI_pos pixels inside the M2 foreground, pooled over all 42 wells at t52.
    The AUROC is also computed per well, and the median is reported. Dead-class precision and recall against PI_pos at argmax are reported too.
  - S3 Dose: Spearman of DF with dose (log10(dose+1)) at t52. The author 4-parameter fit is applied to DF and FG area at t52,
    and the EC50 values are compared descriptively with the author MTS / PI / area EC50 (§1).
  - S4 Time course: within each well, Spearman of DF(t) vs PI_img(t) over 19 frames. The median is given for treated (≥30 nM)
    and untreated-like (0–10 nM) wells.
- Descriptive:
  - DP versions of the primary endpoint and S1.
  - Seeds 3407 / 2026.
  - t = 72 h.
  - The same analyses with comparators (a) and (b) in place of DF, so the area-only performance is shown next to Dead.
  - QC: M2 vs OrganoID foreground Dice per frame, and the number of empty-foreground frames.

## 5. Statistics
- Well-level CIs: bootstrap of wells stratified by dose group (10,000 replicates, RNG 20261109), 95% percentile interval.
- Pixel AUROC CI: well-cluster bootstrap over wells (same settings).
- Permutation p for the primary endpoint: DF permuted across wells, 10,000 permutations.
- n = 42 wells from one PDAC line and one plate. Dose is close to binary (MTS: ≤10 nM ≈ alive, ≥30 nM ≈ 0.23), so ρ is
  driven mostly by the dose step. This is stated as a limitation, and S1 / S4 address it.
- **Allowed claim "M2's Dead class tracks an independent dead-cell stain on an external dataset" only if all three hold:**
  - the primary CI lower bound > 0;
  - the S1 CI lower bound > 0;
  - the S2 pooled AUROC CI lower bound > 0.5.
- If only the primary holds, the wording is "the Dead fraction follows dose/PI, but no evidence beyond organoid shrinkage".
- If the primary fails, the wording is "Dead class not externally validated".
- Never claim Dead calibration or quantitative death measurement (CRC Dead precision is about 58%).

## 6. Outputs and rules
- New directories only: `gemcitabine_pi/{scale_select, m2_seed42, m2_seed3407, m2_seed2026, analysis}`.
- Logs are kept unfiltered. RESULTS.md records every run, crash and deviation.
- GPU: only truly free GPUs (V100-2 GPU0 is exclusive; A800 3/5/7 only).
- No change of the input stretch, scale rule, threshold T, time point or endpoints after any M2-vs-PI result.
- Scripts and this plan are hashed into `frozen_hashes.txt` before inference.

## Approval (2026-11-09)
- Author approved this plan 2026-11-09 and asked to start. Scripts: `scripts/gem_pi_infer_20261109.py` (scale_select / full),
  `scripts/gem_pi_analyze_20261109.py`. Hashes are in `frozen_hashes.txt`.
- Implementation detail, fixed now: the pixel AUROC is computed from 1,000-bin histograms of P(Dead), with ties inside a bin
  counted as 1/2.
