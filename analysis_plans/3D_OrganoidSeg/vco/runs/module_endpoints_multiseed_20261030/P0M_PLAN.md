# P0-M — run plan (2026-10-30; approved by the author; written before any full-run result)

The protocol is `audit_final_experiments_20261030/BAM_MPM_TARGETED_PROTOCOL.md`. This file only fixes execution details and records amendments made before the run.

## Inference (script `scripts/p0m_module_inference_20261030.py`)
- 12 runs: Full / No BAM / No MPM / Simplified × seeds 42/3407/2026, each `best_miou.pth` (read-only, SHA-256 in each `manifest.json`). Outputs: hard masks and uint8 4-class probabilities.
- 12 dependence runs, Full × 3 seeds × {mpm_const, mpm_swap, bam_const, bam_swap}: hard masks only.
- Inference loop: identical to Stage 30 `infer_hard`. Smoke check: the seed-42 Full and Simplified masks on 2 images are pixel-identical to the Stage 30 cache.
- GPUs 3 and 5, one process each. Both are shared with light jobs of another user; GPUs 2 and 4, which hold another user's reserved memory, are not used.

## Amendments to the protocol (decided before the full run)
1. **Definition of "constant".** MPM and BAM maps are computed per 224-px patch, so "the image's own mean" is implemented as each patch's own spatial mean, per channel.
2. **Definition of "swap".**
   - Each test image gets a fixed partner image: a derangement with seed 20261031, so no image is its own partner.
   - The partner is resized to the same size if needed (2048×2044 vs 2048×2048).
   - The map is computed from the partner patch at the same coordinates.
3. **Scope of the dependence test.** Restricted to the Full model (3 seeds), which contains both modules. The No-BAM and No-MPM single-module variants are not perturbed. Reason: compute. The main question, "does Full use the information from the module?", is answered by Full alone.
4. **Smoke outputs.** 2 images were written to a scratch directory, not to this directory.

## Analysis (separate script, after inference; no change of endpoints)
- **Endpoints and statistics:** those defined in the protocol, §1–§3.4: BF1 at 2 px (1/4/8 sensitivity), ASSD, HD95, Boundary IoU (d = 0.5% of the diagonal), area errors, Dead P/R/IoU, strata, morphology composite (IQR→SD rule), dependence flip rates and Δ metrics. Paired hierarchical bootstrap, seeds + wells, 10,000 replicates, seed 20261031.
- **Metric code:** reuses the Stage 30 `boundary_metrics_all` and `morphology_features` code unchanged.
- **Consistency check:** the seed-42 masks vs the Stage 30 cache, all 167 images; report the agreement fraction.
5. (Analysis script, before any comparison result.) The irregularity stratum's solidity used a convex hull of pixel centres, which gives values > 1 for small objects. It was fixed to use pixel corners (solidity ≤ 1). Found while unit-testing on 2 images; no comparison had been computed.
6. (2026-10-31, analysis crashed before any comparison was computed.) `pd.qcut` failed on the contrast stratum because of tied edges (p99−p1 = 252 or 255 for many images). Tertiles are now cut on ranks (`rank(method='first')`, ties broken by fixed row order), which gives equal-count tertiles for all six strata. The crashed run `analysis_v1/` (log `logs/analysis_v1.full.log`) is kept untouched; the analysis is rerun into `analysis_v2/` (log `logs/analysis_v2.full.log`).
7. (2026-10-31, before any comparison output of `analysis_v2`.) The tie check of amendment 6 showed that `rank(method='first')` assigns tied images to tertiles by file order: 82 of 167 images for contrast (16 distinct values; 82 images at p99−p1 = 252) and 21 for brightness. The other strata are unaffected (0–2 images). Tertiles are now cut on **average** ranks, `pd.cut(rank(method='average'), [0, n/3, 2n/3, n])`. Tied values therefore always share a tertile, and groups may be unequal: contrast 50 / 82 / 35. `analysis_v2` was stopped before writing any output (its log records this); the analysis is rerun as `analysis_v3/` (log `logs/analysis_v3.full.log`). The contrast stratum is near-saturated and is interpreted with caution.
