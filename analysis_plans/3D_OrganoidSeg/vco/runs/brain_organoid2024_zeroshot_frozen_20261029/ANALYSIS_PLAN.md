# BrainOrganoid2024 zero-shot with the frozen models — analysis plan

Written 2026-10-29 before any new result was seen. Authorized by the author ("先跑脑类器官零样本").
Inference only. No training, no fine-tuning, no threshold or scale chosen on these labels.

## Known before this run
- July 2026 zero-shot of the Full model (`vco/runs/expFinal_fromscratch/brain_organoid2024_zeroshot`, rule P(alive)+P(dead) ≥ 0.01, native resolution): global Dice 0.122, precision 0.41, recall 0.07. The contact sheet shows the cause: one brain organoid spans ~300–500 px, so a 224 px window sees mostly uniform dark interior, which the CRC model labels as background.
- Author-provided per-image Dice on the same 1,407 images (`df_pred.csv`): CellProfiler 0.648 ± 0.365, OrganoSeg 0.762 ± 0.349, MOrgAna 0.65–0.82, SegFormer (trained on their folds) 0.960 ± 0.040.
- Our fine-tuned Full model (5-fold, folds differ from the authors'): 0.970 ± 0.029.

## Models
- `simplified` (frozen final; `ablation_no_bam_no_mpm_seed42/best_miou.pth`, sha aa4ba757…) — primary.
- `full` (`expFinal_fromscratch/best_miou.pth`, sha 1ed9e6c7…) — comparator and reproducibility check against the July run.

## Protocol
- Input: image as stored (LabA RGB jpg, LabB grayscale tif → replicated to 3 channels); no intensity stretch.
- Inference: frozen protocol (224 window, stride 112, ImageNet mean/std, reflect pad, softmax average).
- Foreground = argmax ∈ {alive, dead}. Evaluated at native resolution against the binary GT.
- Scales (pre-declared, all reported): input resized by s ∈ {1, 0.5, 0.25, 0.125} (area interpolation); class probabilities bilinearly upsampled to native size before argmax.
  - s = 1 is the primary zero-shot result.
  - s < 1 is a sensitivity analysis of object scale. No single scale is reported as "the" result. Any scale that looks best is labelled oracle-selected.
- Secondary post-processing (pre-declared): fill holes + keep largest component, because each image contains one organoid.
- Reproducibility: Full at s = 1 with rule P(alive)+P(dead) ≥ 0.01 must reproduce the July global Dice 0.122 (±0.002).

## Statistics
- Image-level Dice mean ± SD (paper metric), overall and by lab; global (micro) Dice.
- Paired comparisons with an organoid-level cluster bootstrap (org_id, 64 clusters, 2,000 replicates, seed 20261029): simplified − Full at each scale; each scale − s = 1; simplified vs CellProfiler / OrganoSeg (paired by image).
- By day and by embedding state, descriptively.

## Wording rules
- If the s = 1 Dice is poor, report it as a zero-shot failure caused by object scale. Fine-tuning remains the supported route (0.970).
- Rescaling results do not count as zero-shot performance unless a scale rule that does not use GT is specified in advance. None is specified here.
