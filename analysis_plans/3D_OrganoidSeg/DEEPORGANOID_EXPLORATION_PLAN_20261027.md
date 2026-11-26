# Initial DeepOrganoid exploration plan for VitaFormer

Plan date: 2026-10-27

## 1. Purpose

Use the original frozen Full VitaFormer model, including MPM and BAM, to test whether segmentation-derived bright-field morphology captures drug-treatment phenotypes and aligns with an independent CellTiter-Glo functional readout in DeepOrganoid.

This phase is exploratory external functional validation. DeepOrganoid has no manual pixel-level segmentation masks, so it cannot establish external segmentation IoU or external alive/dead pixel accuracy.

## 2. Primary research questions

1. Does the frozen Full Model produce usable, stable foreground masks across nine DeepOrganoid cell lines?
2. Do segmentation-derived morphology features associate with independent `Lum_FA` CTG inhibition measurements?
3. Are morphology–assay associations reproducible across held-out cell lines rather than driven by random image-level splitting?
4. Do increasing drug concentrations produce coherent dose–morphology trajectories?
5. After controlling for cell line, concentration, and overall assay effect, do different compounds retain distinguishable morphology fingerprints?
6. Only after the above: does MPM improve biological/morphology alignment relative to No MPM, and does BAM improve target-mask boundary plausibility or QC?

## 3. Frozen starting point

- Primary model: Full Model checkpoint selected by validation mIoU, initially seed 42 for continuity with existing full inference.
- No target labels are used for segmentation training or checkpoint selection.
- Primary image type: IXM_BestFocus.
- Existing full inference for 3,456 images should be audited and reused before any rerun.
- The original model remains read-only.

Because the dataset has already been explored across all nine cell lines, the analysis must be described transparently as exploratory. Strict leave-one-cell-line-out or grouped resampling reduces optimism but does not recreate a never-inspected external cohort.

## 4. Phase A — data and output audit

### A1. Metadata audit

- Verify one row per cell line × plate/well condition.
- Standardize compound names, function groups, concentrations, and controls.
- Identify DMSO/NEG and EMPTY/MEDIA rows.
- Derive stable identifiers for cell line, plate/date, row/column, drug, concentration, and technical replicate.
- Document whether the two nominal replicates are true technical or biological replicates.

### A2. Existing-inference audit

For every image record:

- foreground fraction
- alive/dead/bubble fraction
- mean maximum softmax probability
- prediction entropy
- all-background flag
- no-Dead flag
- excessive foreground flag
- excessive bubble flag
- border-touching fraction
- connected-component count and size distribution

Stratify QC by cell line, compound, concentration, image type, and `Lum_FA` range. Produce representative overlays for normal and failure cases.

Decision gate: continue with phenotype analysis only for features/configurations that pass prespecified QC. Never select a model solely because it maximizes CTG correlation while producing visibly drifting masks.

## 5. Phase B — morphology feature extraction

Use identical definitions for all models/configurations.

### B1. Primary foreground features

- foreground area ratio
- connected-component count
- median and quantile component area
- small- and large-component fractions
- fragmentation index
- foreground solidity/circularity where stable
- border-contact fraction

### B2. Image-conditioned morphology

Measure within predicted foreground:

- intensity mean and dispersion
- pellucidity
- entropy or texture proxy
- gradient magnitude
- edge density

### B3. Alive/dead auxiliary features

- alive/dead soft probability ratio
- hard Dead fraction
- Dead-positive component count
- uncertainty around alive/dead boundaries

These are secondary because 3,242/3,456 original predictions contained no hard Dead pixels. Do not make the external Dead-mask story the primary endpoint.

### B4. Control normalization

For each cell line and matched assay context, calculate DMSO/NEG-referenced feature effects. Keep both raw and normalized values. Record fallback behavior when matched controls are unavailable.

## 6. Phase C — prespecified functional-assay alignment

### Primary endpoint

Association between `Lum_FA` and a prespecified foreground-morphology feature set under leave-one-cell-line-out evaluation.

Recommended primary configuration based on existing evidence:

- Full Model
- IXM_BestFocus
- fixed scale/fusion configuration selected without using the held-out cell line
- ridge regression or another simple regularized model
- no drug name, cell-line identity, plate position, or raw `Lum_FA` leakage as model inputs

Report:

- pooled Pearson and Spearman
- R2 and MAE
- per-cell-line metrics
- macro-average across held-out cell lines
- grouped bootstrap confidence intervals resampling cell lines and experimental conditions
- permutation/shuffled-label negative control

### Required comparators

- area only
- raw-image/basic intensity features without segmentation
- foreground morphology without MPM-derived features
- full morphology feature set
- simple DMSO-normalized area response

The result should show whether segmentation contributes information beyond image intensity and overall area.

## 7. Phase D — dose and compound phenotype analysis

### D1. Dose-response consistency

Within each cell line × compound:

- rank concentrations
- test monotonic trends of `Lum_FA` and morphology features
- estimate Spearman dose–feature relationships
- summarize concordant and discordant conditions
- use cluster-aware confidence intervals across cell lines

### D2. Drug-specific morphology fingerprints

The claim “different drugs induce different death phenotypes” requires more than representative images.

Construct a phenotype vector from area, component-size distribution, fragmentation, intensity, entropy, gradient, and pellucidity. Then test whether compound identity explains residual morphology after accounting for:

- cell line
- concentration
- `Lum_FA` or effect-strength bin
- plate/date when available

Possible analyses:

- mixed-effects models per feature
- PERMANOVA on residualized phenotype vectors
- leave-one-cell-line-out compound classification
- incremental performance over a baseline containing only concentration and `Lum_FA`

Only claim drug-specific phenotypes if morphology adds reproducible information after these controls. Otherwise describe selected patterns as exploratory examples.

## 8. Phase E — figures and tables

Recommended main/supplementary outputs:

1. Dataset and workflow diagram: bright-field image → frozen Full Model → mask/QC → morphology → CTG alignment.
2. QC figure across nine cell lines with representative success and failure overlays.
3. Predicted versus observed `Lum_FA` under leave-one-cell-line-out evaluation.
4. Per-cell-line forest plot with confidence intervals.
5. Area-only versus full-morphology comparison.
6. Dose–morphology curves for three to four representative compounds.
7. Matched low/middle/high `Lum_FA` images with masks and morphology radar/heatmap.
8. Residualized compound morphology map or clustering, only if supported after adjustment.
9. Supplementary full QC table and all compound/cell-line results.

## 9. Claims allowed and prohibited

### Supported if analysis succeeds

- The frozen model generated segmentation-derived morphology features on an independent organoid drug-screening dataset.
- Foreground morphology showed measurable alignment with an orthogonal CTG assay.
- Morphology carried information beyond area alone, if confirmed by the comparator analysis.
- Selected drug treatments showed reproducible dose-associated morphology shifts.

### Not supported without manual target masks

- Improved external segmentation IoU.
- Accurate external alive/dead pixel labels.
- CTG as a pixel-level ground truth.
- More accurate masks merely because CTG correlation is higher.
- Clinical-response prediction or a validated viability biomarker.

Use “functional alignment,” “assay-aligned phenotype,” and “exploratory external validation,” not “external segmentation accuracy.”

## 10. Architecture comparison after Full-Model exploration

Do not use DeepOrganoid correlation alone to select the final architecture. After the Full-Model story is established, compare Full, No MPM, No BAM, and No BAM + No MPM on a fixed multi-objective table:

| Dimension | Full | No MPM | No BAM | No BAM + No MPM |
|---|---:|---:|---:|---:|
| held-out primary mIoU/Dead IoU | known | known | known | known |
| seed stability | known | known | known | known |
| GT morphology agreement | pending | pending | optional | pending |
| boundary endpoints | pending | optional | pending | pending |
| DeepOrganoid QC/drift | primary first | later | later if useful | later |
| CTG alignment under fixed analysis | primary first | later | later if useful | later |
| inference time/memory | pending | pending | pending | pending |
| implementation complexity | highest | lower | lower | lowest |

Decision rules:

- Retain MPM if it improves GT morphology faithfulness, decision faithfulness, or external assay alignment under a fixed analysis, even without mIoU gain.
- Retain BAM if it improves boundary-specific endpoints or robustly reduces biologically important boundary errors.
- If neither specialized endpoint improves, use No BAM + No MPM as the final segmentation core and compute morphology as an explicit post-processing explanation layer.

DeepOrganoid should be treated as supportive biological validation, not an unrestricted architecture-tuning set. Any model comparison performed after inspecting Full-Model results must be declared exploratory/sensitivity analysis.

## 11. Optional high-value annotation

If feasible, manually annotate a stratified 60–100-image DeepOrganoid subset covering:

- all nine cell lines
- DMSO controls
- low, middle, and high `Lum_FA`
- several compounds and concentrations
- normal, small-object, bubble-heavy, and failure-prone images

Even binary foreground masks would allow external segmentation QC. A smaller alive/dead subset would be highly valuable but should not be required for the first exploratory paper version.

## 12. Immediate next action

Audit the existing Full-Model outputs under:

`external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment/12_vitaformer_deeporganoid_seg_survival/original_vitaformer`

and the Stage 16 static analysis under:

`external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment/16_vitaformer_morphology_assay_alignment`

Create a concise inventory of reusable files, stale/inconsistent summaries, missing statistical controls, and figures that can be regenerated from existing tables before launching any new inference.

