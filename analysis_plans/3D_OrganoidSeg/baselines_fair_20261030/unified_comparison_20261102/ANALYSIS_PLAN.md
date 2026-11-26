# Unified 3-seed comparison: analysis plan (2026-11-02, written before the bootstrap was run)

**Already seen before this plan:** per-seed and mean test metrics of every method (from `shared_eval/` and the earlier reports). The bootstrap intervals below have not been computed yet. Every contrast listed is reported, whatever its result.

## Inputs (no inference re-run; per-image tp/fp/fn over the 167 test images)
| Group | Source | Note |
|---|---|---|
| VF-Simplified (final model) | `vco/runs/ablation_no_bam_no_mpm_seed*/test_best_miou_uniform.json` | |
| VF-Full | `expFinal_fromscratch`, `full_model_seed3407`, `full_model_seed2026` | |
| VF-Full-noRD | `ablation_no_rare_dead_seed*` | Full architecture without the Rare-Dead bundle |
| SegFormer-B2 | `shared_eval/segformer_b2_seed*.json` | |
| P1-S (SegFormer-B2 + Rare-Dead) | `shared_eval/segformer_b2_raredead_seed*.json` | |
| nnU-Net (primary: no TTA) | `shared_eval/nnunet_run2_seed*_noTTA.json` | |
| nnU-Net TTA (secondary) | `shared_eval/nnunet_run2_seed*_TTA.json` | |

- The VitaFormer JSONs come from `evaluate_test.py`. The shared evaluator reproduces them to < 0.002 pp (P0-B consistency check).
- The stem order must be identical across all inputs, or the analysis stops.

## Statistics
- **Bootstrap:** the established hierarchical paired bootstrap (`analyze_multiseed_ablation.py`), 10,000 replicates, RNG seed 20261102.
  - Each replicate resamples 3 seed positions with replacement.
  - For each seed position it resamples the 154 physical wells with replacement.
  - Both arms are evaluated on the same images. Pixels are pooled over the resampled images.
- **Interaction:** for the difference-in-differences contrast, all four groups use the same draws.
- **Reported per contrast:** observed mean Δ, 95% percentile CI, per-seed Δ, and the number of seeds favouring each side.
- **Endpoints:** mIoU, mDice, Alive IoU, Dead IoU, Bubble IoU, Dead recall, Dead precision.
- **Cross-seed stability:** SD and range over the 3 seeds, per method and endpoint. Descriptive only. With 3 seeds, no test of variance differences is meaningful, and none is made.

## Contrasts (A − B), all reported
1. VF-Simplified − SegFormer-B2
2. VF-Simplified − nnU-Net
3. VF-Full − SegFormer-B2
4. VF-Full − nnU-Net
5. SegFormer-B2 − nnU-Net
6. Rare-Dead in SegFormer: P1-S − SegFormer-B2
7. Rare-Dead in VitaFormer: VF-Full − VF-Full-noRD
8. Interaction: (P1-S − SegFormer-B2) − (VF-Full − VF-Full-noRD)
9. VF-Simplified − P1-S (both trained with Rare-Dead)
10. Secondary: nnU-Net TTA − nnU-Net no TTA

## Known caveats (stated before results)
- **NaN divergence under fp16 AMP in 5 of 6 SegFormer runs:**
  - baseline seeds 42 and 3407;
  - all 3 P1-S seeds (iterations 39.4k, 29.1k and 33.7k).
  - In every run the best-validation checkpoint precedes the divergence.
  - Every float tensor in the P1-S `final.pth` files is NaN.
  - In all SegFormer runs, including the non-diverged ones, the running mean of the decode-head BatchNorm reaches 1.2–3.6 × 10⁴ (fp16 maximum: 65,504).
  - Hypothesis: fp16 activation overflow in the decode head. This is not tested.
- **Pairing:** "seed 42" of two different methods is not a shared random state. Seed pairing is the established convention; image and well pairing is real.
- **The Rare-Dead 2×2 is not a perfectly matched design.** Each architecture keeps its own base recipe:
  - VitaFormer: CE/Focal + Tversky, crop 224, fp32 + clipping;
  - SegFormer: CE/Focal, crop 512, fp16, no clipping.
