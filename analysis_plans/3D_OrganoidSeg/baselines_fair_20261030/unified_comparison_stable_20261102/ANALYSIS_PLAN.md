# Unified comparison with the stable SegFormer arms: plan (2026-11-02, written before any stability-run result)

- **Same script and statistics as `unified_comparison_20261102`:** hierarchical paired bootstrap (seed positions + 154 wells), 10,000 replicates, RNG 20261102, shared draws, the same 7 endpoints. Only the group and contrast lists change.
- **New groups:**
  - SegFormer-B2-stable: `shared_eval/segformer_b2_stable_baseline_seed*.json`
  - P1-S-stable: `shared_eval/segformer_b2_stable_p1s_seed*.json`

  The VitaFormer, original SegFormer and nnU-Net inputs are as before.

## Contrasts (all reported)
1. VF-Simplified − SegFormer-B2-stable
2. VF-Full − SegFormer-B2-stable
3. SegFormer-B2-stable − nnU-Net
4. Rare-Dead in SegFormer (stable): P1-S-stable − SegFormer-B2-stable
5. Rare-Dead in VitaFormer: VF-Full − VF-Full-noRD (unchanged, for the interaction)
6. Interaction (stable): (P1-S-stable − SegFormer-B2-stable) − (VF-Full − VF-Full-noRD)
7. VF-Simplified − P1-S-stable
8. Stability effect, baseline: SegFormer-B2-stable − SegFormer-B2 (fp16)
9. Stability effect, P1-S: P1-S-stable − P1-S (fp16)

## Reporting rule
- **Primary record:** the original fp16 results.
- **Sensitivity:** the stable results. If a conclusion from `FINAL_EXPERIMENT_REPORT_20261102.md` (Q1 or the VF-vs-SegFormer comparison) changes direction or CI status under the stable arms, the change is reported explicitly, and the conclusion is stated as **not robust to training numerics**.
- **If a stable run itself diverges** (any non-finite loss): it is reported as is, and no further fix is attempted.
