# Stage 38A — nnU-Net input-configuration sensitivity on DeepOrganoid (DRAFT 2026-11-09, awaiting author approval)

**Status: post hoc.** It was written after Stages 35/37 were seen and is prompted by an external review. It is not part of any pre-registration.
All configurations run are reported. No configuration is chosen by its CTG result.

## Question
Is nnU-Net's DeepOrganoid failure (rim/corner false positives; within-drug line ranking ≈ 0) caused by an input-handling
choice rather than by the model? This decides whether Figure 5 may claim an M2 advantage.

## Facts verified from code (2026-11-09)
- Script `scripts/run_nnunet_deeporganoid_20261102.py` uses `nnUNetPredictor.predict_single_npy_array` with spacing (999, 1, 1).
  Settings: tile step 0.5, Gaussian weighting, **mirroring off** (the CRC primary setting), checkpoint_final, fold 0.
- Plans (`Dataset801_CRCOrganoid`, 2d):
  - spacing 1.0 × 1.0, so no resampling happens;
  - patch 1024 × 1280;
  - CRC median image 2044 × 2048;
  - per-image per-channel ZScore over the whole image (`use_mask_for_norm` False).
  - The z-score is therefore computed on the real 682 × 682 image, **before** padding.
  - The image is then constant-padded (value 0 = image mean after z-score) up to 1024 × 1280. About 65% of the network
    input is this padding. The prediction is cropped back to 682 × 682.
- CRC images are never padded (2044 × 2048 ≥ patch), so the network never saw a large constant border in training.
- Input image: identical to M2 (`load_new` p1–p99 stretch, grey ×3). Foreground = classes 1 + 2. No post-processing for either model.
- Organoid scale (CTG-blind check):
  - median component equivalent diameter is 17.0 px in CRC training GT and 13.4 px in DeepOrganoid M2 vehicle-well masks;
  - linear ratio 1.27, so the scale mismatch is small.

## Configurations (nnU-Net seed 42 screen; one factor changed vs V0 each)
| ID | Change | Rationale |
|---|---|---|
| V0 | none (Stage 35 masks reused; reproduction check: E3 = 0.068) | reference |
| V1 | **reflect-tile** the 682² image to ≥ 1024 × 1280 before prediction; keep the central 682² prediction | removes the constant-padding border the model never saw |
| V2 | upsample ×1.27 (bilinear) before prediction; downsample the probabilities back to 682² | matches organoid size to CRC (from the CTG-blind rule above) |
| V3 | mirroring TTA **on** (nnU-Net default) | the default inference setting that a reviewer would use |

## Endpoints and analysis (unchanged Stage 35/37 code and seeds)
- QC: median foreground per line; foreground-mask Dice vs M2; border-touch fraction; contact sheets.
- E1 core NEG-delta LOCO Pearson; E1b area_raw; E3 within-drug line ranking; E3b pooled potency.
- Paired Δ = M2 s42 − nnU-Net variant, with the same draws.

## Decision rule (fixed before running)
- **"Advantage robust to configuration"**: for every variant V1–V3, the E3 Δ CI excludes 0 in M2's favour.
- **"Configuration-dependent"**: in any variant the E3 Δ CI includes 0 or favours nnU-Net. That variant is then rerun with
  seeds 3407 and 2026 to confirm, and Figure 5 is rewritten as "input handling determines functional readout reliability".
  No claim of an inherent M2 advantage is made in that case.
- Because only one seed is screened, "not recovered" supports only limited configuration robustness, not "nnU-Net generally fails".

## Cost and rules
- GPU: about 3 × 30 min (V100-2 GPU0, exclusive), plus the analysis (about 50 min each).
- Output: `38_posthoc_robustness_20261109/A_nnunet_V{1,2,3}/` (new). Stage 35/37 are read-only.

## Approval and frozen details (2026-11-09, before any variant was run)
- The author approved A and B. All variants are kept and reported. No variant is selected or hidden by its CTG result.
  No architecture search or other experiments are added.
- Padding verified in the nnU-Net v2 source (`predict_from_raw_data.py` l.690): `pad_nd_image(..., 'constant', {'value': 0})`
  after normalisation; `slicer_revert_padding` crops back.
- Implementation (`scripts/run_nnunet_deeporganoid_variants_20261109.py`, copy of the Stage 35 script + `--variant`):
  - **V1**: numpy `reflect` pad of the uint8 image, centred: H 682 → 1024 (171/171), W 682 → 1280 (299/299).
    Disclosed side effect: nnU-Net's per-image z-score is then computed over the tiled image (reflected content
    included). The central 682² of the softmax is kept, argmax.
  - **V2**: PIL bilinear resize 682 → 866 (= round(682 × 1.27)); prediction; softmax bilinear-resized to 682² (torch, align_corners False); argmax.
  - **V3**: `use_mirroring=True`; all else = V0.
- The features, `feats(prob, gray)` and border metrics are unchanged. Masks and overlays are saved.
- Analysis `scripts/analyze_stage38A_nnunet_variants_20261109.py` (copy of Stage 37, models = M2 s42 REF, nnU-Net s42 V0/V1/V2/V3)
  → `A_analysis/`. Contact sheets show M2 | V0 | V1 | V2 | V3. Reproduction check: V0 must give E3 = 0.068 and E1 core = 0.777.
- Confirmation rule (fixed now): if any variant's E3 Δ CI includes 0 or favours nnU-Net, that variant is run for seeds
  3407 and 2026, and the analysis is rerun with `--confirm <variant>` → `A_analysis_confirm_<variant>/`. If 2 of the 3 seeds
  show the same, the "configuration-dependent" reading applies.
- GPU: V100-2 is currently occupied by another user. A waiter starts the runs on the first GPU with no compute process.
  A800 / V100-1 are not reachable from this session.
- Wording: CRC image codes are called "sample codes", with no inference about patient identity.

## Confirmation rule fired (2026-11-09 13:27, recorded before any confirmation seed was run)
- Screen (seed 42): V0 reproduced (E3 0.068, E1 core 0.777). E3 delta M2 − nnU-Net: V1 +0.223 [0.087, 0.367];
  **V2 −0.007 [−0.133, 0.087]**; **V3 +0.223 [−0.027, 0.440]**. V2 and V3 CIs include 0, so both are rerun with seeds 3407 and 2026
  (`run_A_confirm.sh`), then `--confirm V2` and `--confirm V3`.
- Hardware: the confirmation seeds run on GPU-P100-3 GPU1 (the V100 GPUs were not reachable or free). Check before running:
  first 50 wells of V2/V3 seed 42 re-run on the P100 vs the V100 masks gave pixel agreement ≥ 0.99996 (mean 0.99999) and
  max |Δ foreground ratio| 2e-5. Treated as negligible; disclosed.
