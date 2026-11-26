# oreval — evaluating segmentation-derived functional readouts (v0.1.0)

Companion resource to *"Segmentation accuracy and functional readouts in bright-field organoid imaging: a comparative evaluation"*.
It lets you run **any** segmenter's masks through the same functional analyses used in the paper and report them with their evidence tier.

```
masks (any model)  ──oreval.masks──▶  readouts.csv  ─┐
experimental metadata (controls, groups, assay, drug, conc, line) ─┴─oreval.run (JSON config)──▶ tables
```

## Install
Python ≥ 3.10 with numpy, scipy, pandas, pillow (tested with Python 3.10/3.11; scipy ≥ 1.11, pandas ≥ 2.0). No GPU, no deep-learning
framework needed. Run commands from this folder.

## Use
1. Readouts from masks:
   `python -m oreval.masks --masks <mask_dir> --foreground-labels 1,2 --output runs/<model>_s<seed>.csv`
   (or supply your own table with an `image` column and numeric readouts).
2. Write a config (see `configs/toy_example.json`, `configs/validation_*.json`; all keys documented at the top of `oreval/run.py`).
3. `python -m oreval.run --config <config.json>`.

## Outputs and tiers
| File | Endpoint | Tier |
|---|---|---|
| `association.csv` | Spearman ρ(control-normalised readout, assay), pooled and per stratum, cluster-bootstrap CI | control-relative, no fitting |
| `association_paired.csv` | paired Δ\|ρ\| between runs on shared bootstrap draws | — |
| `dose_direction.csv` | sign agreement of ρ(conc, readout) and ρ(conc, assay) in series with a clear assay trend | control-relative, no fitting |
| `within_group_ranking.csv` | per drug: Spearman over lines of concentration-averaged responses; mean over drugs; drug bootstrap | control-relative, no fitting |
| `general_sensitivity_baseline.csv` | ranking from each line's mean assay response to the other drugs | assay-only reference |
| `seed_ranges_association.csv` | observed range (max − min) of ρ across seeds per model family | descriptive |
| `run_log.json` | config and the number of undefined normalisations per run | — |

Calibrated predictions (fitting readouts to assay labels) are not produced here; fit them with your own held-out design and label
them as calibrated.

## Design choices you must state (the paper shows they change results)
- **Controls** are defined from metadata only, never from assay values. The normalisation mode is `ratio` or `difference`, and
  undefined ratios become NaN by default; their count is logged.
- **Clusters**: rows that share a physical unit must share a cluster. For example, a well imaged on two microscopes in NViR is one
  cluster, otherwise CIs are too narrow.
- **Group bootstrap for rankings** resamples drugs. Drugs that share a mechanism are not independent, so this CI can be too narrow.
- **Input scale and training seed** belong to the evaluated pipeline. Choose the scale without the functional readout, and run several
  seeds.

## Verification
- **Toy example (synthetic, no third-party data).** `python examples/make_toy_example.py` builds `examples/toy/`, which is already
  shipped. Then `python examples/check_toy.py <new_dir>` re-runs it and compares with `examples/toy/expected_output` (PASS on
  2026-11-10). Model B has edge false positives, so its within-drug ranking inverts (−0.75 vs 1.0 for model A), as nnU-Net did on DeepOrganoid.
- **Reproduction of the paper's analyses** (`validation/REPRODUCTION.md`). Built from the paper's readout tables, oreval reproduces:
  - NViR area–ATP ρ and CIs for all 9 model × seed runs;
  - NViR dose-direction counts;
  - DeepOrganoid within-drug rankings and CIs for all 9 runs, and the general-sensitivity baseline (0.735).

## Data
Third-party data are not redistributed. `manifests/` lists the identifiers and splits used:
- `crc_split.csv`: 812 CRC image ids with train/val/test;
- `nvir_split_and_wells.csv`: 2,359 NViR images with split, domain and physical well id (2,114 unique wells);
- `deeporganoid_samples_and_cohorts.csv`: 3,456 DeepOrganoid wells with metadata roles and cohort flags (no assay values).

Where to obtain each dataset: `manifests/DATA_ACCESS.md`.
