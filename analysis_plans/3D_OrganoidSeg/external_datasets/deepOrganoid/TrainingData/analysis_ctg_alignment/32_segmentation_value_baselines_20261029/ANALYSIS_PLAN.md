# Stage 32 plan — does VitaFormer segmentation add downstream value? (written 2026-10-29 before any Stage 32 result)

Question (reviewer): would a simple threshold segmentation + area give the same CTG/drug-response results?

## Image representations (all from the same p1–p99 uint8 images used in Stage 31; no CTG used to set any parameter)
- R0 `intensity_only`: whole-well (inner region) intensity mean/std/p10/p50/p90 + entropy + gradient — no segmentation.
- R1 `threshold_global_otsu`: within the well mask, pixels darker than the Otsu threshold of the well intensities; min component 10 px; holes filled.
- R2 `threshold_local_contrast`: dark-contrast map = Gaussian(σ=25) background − image; Otsu on the in-well contrast map; min component 10 px; holes filled (OrBITS-style background-corrected threshold).
- R3 `vitaformer_simplified` (Stage 31), R4 `vitaformer_full` (Stage 31), reported; R3 primary.
Well mask (fixed geometry for all): pixels > Otsu of the full image, largest component, eroded 15 px. Parameters above are fixed now and are not tuned.
Morphology features use the identical `feats()` function (one-hot masks for R1/R2).

## Endpoints (identical protocol to Stage 31; main cohort all nine lines; Organoid52-excluded sensitivity)
E1 LOCO ridge CTG alignment, NEG-delta, specs: area only; core 9 features. Hierarchical bootstrap CI (shared draws across representations) and paired bootstrap of the Pearson difference R3 − R1/R2/R0.
E2 Dose-response AUC agreement (92 line × drug): model-free area-relative-to-NEG AUC vs CTG AUC, Spearman + cell-line bootstrap; within-line drug rank; within-drug line rank.
E3 Dose trajectories concordance (Stage 24 rule).

## Feature increment within VitaFormer simplified (E1 protocol, NEG-delta)
M1 area; M2 area + size distribution (component count, mean/max area, equivalent diameter median, small/large fraction); M3 all 21 rich features. Accept a null increment; no further feature-set search.
VitaFormer alive-only area vs alive+dead foreground area reported (expected identical; dead ≈ 0).

## Metadata baseline (does the image add information beyond known conditions?)
LOCO ridge on drug one-hot + log10 concentration + drug × log10 concentration (+ control indicator); vs the same + area (R3 NEG-delta); vs area alone. Held-out cell-line identity is never an input; NEG normalization uses only that line's own control wells (available at test time in a real screen).
