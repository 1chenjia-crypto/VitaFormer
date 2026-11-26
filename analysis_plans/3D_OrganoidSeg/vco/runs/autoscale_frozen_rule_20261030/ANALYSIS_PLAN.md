# Rule-frozen automatic scale adaptation — analysis plan

Written 2026-10-30, before any auto-scale result was computed. Requested by the author: one rule-frozen automatic scale experiment with strict independent validation, then the Stage 31–33 figures.
Inference only. No training. Model: frozen simplified (`ablation_no_bam_no_mpm_seed42/best_miou.pth`, sha aa4ba757…). The Full model gets the same frozen rule as a secondary analysis.

## Why independence needs care
On BrainOrganoid2024 we have already seen that 0.25× downscaling works (oracle). The rule designer (us) is therefore not blind to Brain. To keep the validation honest:
1. The rule is specified here from first principles and CRC **training** labels only.
2. It is checked (go/no-go) on a **development set built from CRC validation images only**: synthetic magnification.
3. It is frozen (a JSON with SHA-256) before any external image is processed with it.
4. It is then run **once** on Brain (semi-independent: the designer saw native and oracle results) and on four datasets whose scale behaviour was never examined (**fully independent**). Results are reported whatever they are.

## The rule (v1)
Input image I (H×W). Candidate scales G = {1, 1/2, 1/4, 1/8}. A scale s < 1 is skipped only if min(H·s, W·s) < 32 px (degenerate guard); images smaller than 224 px are reflect-padded, as in the frozen protocol.
For each candidate s:
1. I_s = PIL BOX resize of I by s; P_s = frozen sliding-window inference (224 / 112, ImageNet norm).
2. F_s = argmax(P_s) ∈ {alive, dead}; fill holes; keep components with area ≥ 100 px (in I_s pixels).
3. d_s = area-weighted median equivalent diameter √(4A/π) of the kept components, in I_s pixels (undefined if none).
4. c_s = |ln(d_s / D_ref)|.

Selection: s* = argmin c_s over defined scales; ties go to the larger s; if no scale is defined, s* = 1.
Output: argmax of P_{s*}, bilinearly upsampled to H×W; foreground = alive ∪ dead.

D_ref is the pooled area-weighted median equivalent diameter of filled GT organoid components (classes 1 ∪ 2, ≥ 100 px) over the 481 CRC train_cond images at native resolution. This is the object size the model was trained on. Computed 2026-10-30 from 16,584 components: **D_ref = 35.49 px** (count median 18.7; p10–p90 12.6–35.9). No parameter of the rule uses any external label.

## Development check (CRC validation only, 164 val_cond images)
- **Construction:** for each image and magnification m ∈ {1, 2, 4, 8}:
  - pick one GT organoid component (seeded, seed 20261030);
  - crop a (1024/m)² window centred on its centroid (shifted to stay inside the image);
  - upsample by m with bicubic interpolation, giving a 1024² canvas.
  - The correct scale is 1/m. GT is used only to build the set and score it.
- **Pre-specified go criteria (simplified model):**
  - (a) exact scale recovery ≥ 90% at m = 1;
  - (b) exact recovery ≥ 70% at each of m = 2, 4, 8;
  - (c) at m = 1, auto − native image Dice ≥ −0.01 (point estimate).
- **No-go handling:** if the rule fails, it may be revised **using the development set only**. Every revision is logged with its dev results (v2, v3, …). External data is not touched until a version passes.

## Independent validation (run once with the frozen rule)
| Set | Images | Independence |
|---|---|---|
| CRC test_cond (in-domain non-degradation) | 167 | in-domain; test labels used before for other purposes, never for this rule |
| BrainOrganoid2024 (all) | 1,407 | semi-independent (native/oracle seen); uses the cached probabilities from 2026-10-29 |
| OrganoID mouse (validation + testing) | 8 | independent |
| OrganoID PDAC (validation + testing) | 42 | independent (native Dice of the Full model, alive-only rule, was seen as an aggregate) |
| OrgaSegment (train + val + eval; never used to train our model) | 231 | independent |
| OrganoIDNet (Validation + Test; non-augmented) | 189 | independent |

The `training` folders of OrganoID are augmented 512 px crops and are excluded.

## Endpoints
- Per set: image Dice (mean ± SD) and global Dice for native, auto, and each fixed scale (each fixed scale is an oracle reference only); scale-choice distribution.
- Primary per set: auto − native image Dice, image-level bootstrap 95% CI (2,000, seed 20261030; Brain uses an org_id cluster bootstrap).
- Pre-declared hypotheses:
  - H1 (Brain): auto − native > 0, with CI excluding 0.
  - H2 (non-degradation): for every set where the native image Dice is ≥ 0.6, auto − native ≥ −0.02 (lower CI bound reported).
  - H3 (gap to oracle): auto vs the best fixed scale per set, reported descriptively.
- Only the simplified model's results count for H1/H2. Full is descriptive.

## What will be claimed
- If H1 and H2 hold: "a size-normalisation rule fixed in advance from training data alone restores zero-shot transfer to large organoids without hurting in-domain or other external sets". The rule then goes into the Windows tool.
- If either fails: report the failure; the tool keeps a manual "organoid size" setting and fine-tuning remains the recommended route.

## Amendments (before any auto-scale result)
- 2026-10-30, A1: the minimum-side condition was changed from 224 px to 32 px. With 224 px, the rule could never select 1/4 or 1/8 for 768–1,040 px images. That conflicts with the frozen inference protocol, which reflect-pads small inputs, and would make the rule useless for large-object images. Noticed from image dimensions only, before any rule output existed.
- 2026-10-30, A2 (process record, not a rule change): the first dev launch ran `rm -f dev/*`. That deleted only the outputs of the 3-image smoke test (`--limit 3`) written ~10 minutes earlier; no full result existed. Its smoke numbers (m=1/2/4/8 auto Dice .934/.903/.907/.874) are kept in the session record. That launch also filtered warnings from the log; it was stopped before writing any output. The rerun writes to `dev_simplified_v1_rerun/` (the script refuses an existing directory) with the full unfiltered log `logs/dev_simplified_v1_rerun.full.log`. Output-only additions to the dev script: `--out`, per-scale IoU, and a per-image best-scale oracle. The rule module `autoscale_rule_20261030.py`, the model and the data split are unchanged.
