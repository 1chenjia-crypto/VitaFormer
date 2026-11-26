# BrainOrganoid2024 strong-baseline comparison plan (2026-11-03)

## Status and execution gate

This document freezes the proposed analysis before any new training. No run may
start until the author approves this plan and the implementation audit confirms
that all three models can use the same folds and evaluation contract. New
outputs must be written to new directories; existing checkpoints, predictions,
logs, and reports must not be overwritten or deleted.

Allowed GPUs for later approved runs: 3, 5, and 7 only. GPUs 2 and 4 are
prohibited. Full logs must retain warnings and stderr.

## Why BrainOrganoid2024 is primary

DeepOrganoid has no pixel-level ground truth, so it cannot determine whether
VitaFormer segments more accurately than SegFormer-B2 or nnU-Net. It remains a
secondary functional-validation dataset. BrainOrganoid2024 contains 1,407
labelled images from 64 trackable organoids, two laboratories, four clones, and
ten time points, and supports organoid-ID-level, clone-stratified five-fold
comparison.

## Question and permitted claim

Primary question: under the same BrainOrganoid2024 organoid-ID folds, does the
final VitaFormer simplified model adapt more accurately or more consistently
than SegFormer-B2 and nnU-Net?

This experiment tests cross-dataset **adaptability** after labelled target-data
training. It does not by itself establish frozen zero-shot generalization. The
new three-model comparison will reproduce the source paper's split and
out-of-fold evaluation structure, but it is not a reproduction of the source
paper's model-training recipe.

## Audit of the existing VitaFormer result

The existing run at
`vco/runs/expFinal_brain_organoid2024_5fold/` is useful historical evidence but
is not the final head-to-head result because:

1. it initializes from `vco/runs/expFinal_fromscratch/best_miou.pth`, the
   VitaFormer Full checkpoint, not the final simplified checkpoint;
2. the reported Dice 0.9709 is the mean across folds of fold-level pooled-pixel
   (micro) Dice, whereas the source paper reports image-level Dice summaries;
3. checkpoint selection evaluated a fixed 80-image validation subset although
   the saved local folds contain 176-264 validation images.
4. its folds were regenerated with a local random procedure and do not match
   the source authors' released `splits.csv`. The local test-fold organoid
   counts are 16/12/12/12/12, whereas the official counts are
   13/13/13/13/12.

The historical run will not be overwritten or silently relabelled. The new
comparison will use the final simplified model and a shared evaluation
contract.

## Models

1. **VitaFormer simplified**: final architecture without BAM or MPM; initialize
   from the frozen CRC seed-42 simplified checkpoint
   `vco/runs/ablation_no_bam_no_mpm_seed42/best_miou.pth`.
2. **SegFormer-B2**: `nvidia/mit-b2`; use the numerically stable training path
   and initialize from its frozen CRC seed-42 checkpoint
   `baselines_fair_20261030/segformer_b2_stable_runs/baseline_seed42/best_val_miou.pth`.
3. **nnU-Net v2 2D**: use the frozen CRC seed-42 model as the source for
   transferable network weights from
   `baselines_fair_20261030/nnunet/nnUNet_results_run2_seed42/Dataset801_CRCOrganoid/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth`.

Audited SHA256 values:

- VitaFormer simplified: `aa4ba757ebcc93b0e686360364368b3124e46a647f573fdcaa0d5453ffec0368`
- SegFormer-B2 stable: `ba47f5e0de1da43fe8a8601fd010472c6ef6473930b36aa514729dd328aa60e1`
- nnU-Net v2 2D: `8e3917463defb91f0733ba56a427e8511c4f64b0d8cef0eaf0e3e7778ef7091e`

The primary comparison uses one fixed CRC source seed (42) and five target-data
folds. It does not run three random seeds inside every fold. Variation across
folds is reported as fold variability, not random-seed variability.

## Frozen folds

All models must use the source authors' released split file:

`<WORKSPACE>/OrganoidSegBaseline/results/brain_organoid2024_metric_alignment_audit/source_cache/robust_monitoring_organoid_growth-main/model_training/splits.csv`

Official split SHA256:
`493c7635286b1150c627c604d76238d897c9bc1e2ca5c4e6df14308d06382e22`.

Audited official train/validation/test counts are:

- split 0: 880/241/286 images; 40/11/13 organoids;
- split 1: 880/241/286 images; 40/11/13 organoids;
- split 2: 880/242/285 images; 40/11/13 organoids;
- split 3: 879/242/286 images; 40/11/13 organoids;
- split 4: 902/241/264 images; 41/11/12 organoids.

The official test folds cover all 64 organoid IDs exactly once and generate
1,407 out-of-fold predictions. No image or organoid may cross partitions within
a fold. A copied split file and its SHA256 must be stored in the new run
directory, and split equality must be checked automatically before training.

## Task definition and preprocessing

- Binary segmentation: background = 0, organoid = 1.
- Reinitialize every model's final segmentation layer(s) to two output classes.
  Transfer all shape-compatible non-output weights from the CRC checkpoint; do
  not give any model a retained four-class head advantage.
- Use the source images and author-provided masks at their native spatial
  resolution for final evaluation.
- Training crops, normalization, and sliding-window inference may follow each
  architecture's already established pipeline, but they must be fixed before
  test evaluation and recorded in the manifest.
- No test-label-driven scale selection, threshold selection, connected-component
  filtering, or morphology cleanup is permitted.
- Foreground threshold is 0.5 unless a different value is selected solely on
  the validation partitions according to a rule frozen before training. The
  preferred primary analysis fixes 0.5 for all models.
- No TTA in the primary comparison. A TTA result may only be a labelled
  sensitivity analysis applied to all three models.

## Training and checkpoint-selection contract

- Use the official fold membership and the complete official validation
  partition in each split for all models. Do not regenerate an inner
  train/validation split.
- Initialize each architecture from its own frozen CRC seed-42 model so the
  comparison addresses transfer from the same source task. Output heads may be
  changed only as required for binary segmentation.
- Reuse each architecture's frozen optimizer/numerical-stability settings; do
  not tune learning rates or augmentation using BrainOrganoid test results.
- VitaFormer and SegFormer use a maximum of 40,000 optimizer steps, evaluated
  every 4,000 steps. The selected checkpoint maximizes validation image-level
  mean Dice (not test Dice).
- nnU-Net uses its native `nnUNetTrainer` schedule, frozen before training: 1,000
  epochs, 250 training iterations and 50 validation iterations per epoch, SGD
  with initial learning rate 0.01 and polynomial decay, and the native best-EMA
  validation checkpoint. This is deliberately not forced into the 40,000-step
  transformer schedule. Training compute will be reported. The official nnU-Net
  pretrained-weight loader must transfer every shape-compatible non-segmentation
  layer and skip all `seg_layers`; no schedule choice may be changed after seeing
  test performance.
- Save the best and last checkpoint, complete log, environment metadata, source
  checkpoint hash, fold hash, and command line for every run.

## Outcomes

### Primary outcome

Out-of-fold image-level Dice. Pool the 1,407 out-of-fold predictions and report
the arithmetic mean, standard deviation, median, and interquartile range across
images for each model.

### Secondary outcomes

- image-level IoU, precision, and recall;
- pooled-pixel (micro) Dice and IoU, clearly labelled as micro metrics;
- image-level Dice stratified by Lab A and Lab B;
- image-level Dice stratified by clone, day, and embedding state;
- fold-level mean and standard deviation;
- failure rate: fraction of images with Dice < 0.50 and Dice < 0.90;
- trainable parameters, peak GPU memory, training time, and inference time,
  reported descriptively.

### Source-paper reference

Evaluate the source authors' released SegFormer out-of-fold predictions with
the same image-level evaluator and include them as a separate reference row:
`Original SegFormer MiT-B0 (author OOF)`. This reference used MiT-B0,
256 x 192 downscaling, ImageNet initialization, 1,000 iterations, batch size 2,
AdamW at 1e-4, and a 1:10 cross-entropy/Dice loss. It must not be labelled as
the same protocol as the new CRC-initialized SegFormer-B2 experiment.

## Statistical analysis

- Pair models on the same out-of-fold images.
- For each model contrast, report the mean paired difference in image-level Dice
  and a two-sided 95% confidence interval.
- Use 10,000 paired cluster-bootstrap replicates, resampling the 64 organoid IDs
  with replacement and retaining all images belonging to each sampled organoid.
- Primary contrasts:
  1. VitaFormer simplified minus nnU-Net;
  2. VitaFormer simplified minus SegFormer-B2.
- Adjust the two primary p-values with Holm's method if p-values are reported.
- Lab-specific contrasts are secondary and are not used to redefine the primary
  conclusion.

Interpretation is based on effect sizes and confidence intervals:

- CI entirely above 0: evidence that VitaFormer is better on this endpoint;
- CI includes 0: no demonstrated difference;
- CI entirely below 0: comparator is better.

## Required outputs

Write all new artifacts under a new directory, proposed:

`baselines_fair_20261030/brain_organoid2024_strong_baselines_20261103/`

Required files:

- `ANALYSIS_PLAN.md` (a copy of this frozen plan);
- `manifest.json` with paths, hashes, environment, folds, and commands;
- one subdirectory per model and fold;
- `out_of_fold_metrics.csv`, one row per image and model;
- `fold_summary.csv`;
- `contrasts.csv`;
- `RESULTS.md` including all successes, failures, and deviations;
- representative examples selected by a rule defined without inspecting model
  identity (for example fixed Dice quantiles).

## Stopping and decision rules

1. Run an implementation audit and, if needed, a one-batch CPU/GPU loading
   check that does not train or select parameters.
2. Stop and revise this plan before training if CRC checkpoint transfer is not
   technically equivalent across models or if the saved folds cannot be
   reproduced exactly.
3. After author approval, run the five-fold comparison without interim
   test-result-driven changes.
4. If VitaFormer is better or statistically indistinguishable while showing
   lower variability or another prespecified advantage, proceed to manuscript
   writing without architecture replacement.
5. If VitaFormer is clearly worse, review error patterns before considering any
   architecture change. Do not start module swapping automatically.

## DeepOrganoid follow-up (secondary, separate plan)

DeepOrganoid should subsequently compare frozen VitaFormer, SegFormer-B2, and
nnU-Net using the same image preprocessing and the already frozen functional
endpoints (including CTG LOCO Pearson). Because no pixel masks exist, this will
be labelled functional robustness, not segmentation-accuracy validation. It
must have its own frozen analysis plan before any new inference or analysis.
