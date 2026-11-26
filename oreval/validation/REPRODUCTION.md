# Reproduction of the paper's analyses with oreval (2026-11-10)

`prepare_paper_inputs.py` takes the per-image readouts used in the paper and writes them to `work/`, which is local only:
- NViR: `fg_area` per run from the analysis feature table, plus the authors' metadata;
- DeepOrganoid: `foreground_area_ratio` per run from the Stage 39 feature table, plus metadata roles.

These were then run through the generic `oreval.run` with `configs/validation_nvir.json` and `configs/validation_deeporganoid.json`.

| Endpoint | Paper (stage files) | oreval | Match |
|---|---|---|---|
| NViR pooled area–ATP ρ [CI], M2 s42 / s3407 / s2026 (physical-plate clusters) | 0.851 [0.789, 0.888] / 0.877 [0.831, 0.902] / 0.872 [0.832, 0.894] | identical | ✅ all 9 runs, including CIs |
| NViR seed ranges (VitaFormer / SegFormer / nnU-Net) | 0.026 / 0.252 / 0.234 (v4 text corrected from a hand-rounded 0.233) | 0.026 / 0.252 / 0.234 | ✅ |
| NViR dose direction (of 28 series) | M2 27/26/26; SegFormer 25/26/25; nnU-Net 19/24/18 | identical with `aggregate: "none"` | ✅ |
| DeepOrganoid within-drug ranking (mean ρ [CI]) | M2 s42 0.717 [0.510, 0.862]; …; nnU-Net s3407 −0.382 [−0.562, −0.177] | identical for all 9 runs | ✅ |
| DeepOrganoid general-sensitivity baseline | 0.735 | 0.735 | ✅ |

**Note on dose direction.** With the default `aggregate: "median"` (correlations on per-concentration medians), NViR gives 29 eligible
series instead of 28, and the counts differ slightly. The paper used well-level correlations, so set `aggregate: "none"` to reproduce
it. Report which setting you use.

`work/` and `out_*/` contain third-party-derived values and are not for redistribution.
