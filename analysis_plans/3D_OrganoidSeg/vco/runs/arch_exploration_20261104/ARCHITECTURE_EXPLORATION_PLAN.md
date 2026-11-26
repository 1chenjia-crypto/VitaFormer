# ARCHITECTURE_EXPLORATION_PLAN — VitaFormer multi-scale branch (draft 2026-11-04, awaiting author approval)

Status: **Stage A0 + A executed. Amendments 1–2 adopted 2026-11-04. Stage B (M2 seed 42 only) approved by the author 2026-11-04.**
Stage B requires explicit author approval of this file. Stage C requires a second, separate approval.
The rules in §4 were written **before** any M1/M2 training result exists. The M0 validation reference values in §4.1
are filled in from A0 output (M0 masks only); filling them does not change any rule.

## 0. Fixed context
- Paper main model stays **Simplified VitaFormer (M0)**: `vco/runs/ablation_no_bam_no_mpm_seed{42,3407,2026}/best_miou.pth`
  (seed 42 SHA aa4ba757…). Not modified, not retrained. Even if M2 passes every stage, it does **not** automatically
  replace M0; the author decides afterwards (improvement size, external generalisation, re-validation cost).
- Hypothesis: M0's decoder has no encoder features above 14×14 (layers 4/8/12 are different depths, same resolution);
  adding true high-resolution features may improve boundaries, Dead and small targets.
- Not touched: BrainOrganoid2024 five-fold run (Codex), all frozen results, `model_v2.py`, `train_v2_stratified.py`.

## 1. Models
| ID | Definition | Code |
|---|---|---|
| M0 | Frozen Simplified VitaFormer (existing 3 seeds; no new training) | `model_v2.OrganoidSegV2(use_mpm=False, use_bam=False)` |
| M2 | M0 + from-scratch CNN branch (3 stride-2 stages, 32/64/128 ch → 112/56/28) concatenated as extra skips into dec3/dec2/dec1; decoder widths, ViT projections, head identical to M0 | `model_v3_multiscale.OrganoidSegMultiScale(zero_cnn_input=False)` |
| M1 | M2 with the CNN-branch input set to zeros: identical parameters/FLOPs/optimiser groups, no image information through the CNN skips (capacity control) | `…(zero_cnn_input=True)` |

Design note (why M1 is defined this way): M2 keeps M0's decoder topology, so a "decoder-only" variant with no
high-resolution input is M0 itself. The remaining confound for M2 vs M0 is the extra decoder capacity from wider skip
inputs (+0.67 M parameters); M1 controls exactly that. **M1 is trained only if M2 reaches GO**, because without an
M2 improvement there is nothing to attribute.

Locked: CNN branch design (no channel/depth/pretraining variants), no deep supervision, no BAM/MPM, no nnU-Net
weights, 4-class output, inference protocol (224/112 sliding window, ImageNet normalisation, softmax averaging).

## 2. Training (identical to M0 except `--arch`)
Script `vco/organoid_seg/train_v3_multiscale.py` = copy of `train_v2_stratified.py` with only a `--arch` flag and the
model constructor changed (`A_design_smoke/train_v3_vs_v2.diff`). Command = M0's launch script
(`vco/run_combined_no_bam_no_mpm_20261026.sh`) plus `--arch m2`:
`--lr 1e-4 --lr_encoder 1e-5 --max_steps 40000 --val_every 2000 --freeze_blocks 4 --focal_loss --gamma 2.0
--dead_weight_min 10 --use_dead_center_crop --dead_pixel_dropout_prob 0.0 --sparse_dead_zoom_prob 0.8
--sparse_dead_threshold 5000 --sparse_dead_zoom_factor 2 --oversample_5fu_factor 3 --use_condition_split
--seed 42 --split_seed 42 --no_bam --no_mpm`.
Same split (481/164/167), Rare-Dead, loss, optimiser, schedule, precision (fp32), checkpoint rule (`best_miou.pth` by
the script's in-training validation). Output: `vco/runs/arch_exploration_20261104/stageB_m2_seed42/` (new).

Known limitation, kept for parity: the in-training validation scores one random 224 crop per validation image
(`dataset_v2.PatchOrganoidDataset`, augment=False), so checkpoint selection is noisy. It is kept identical to M0.
The **decision metric** is a separate whole-image evaluation of the selected checkpoint on all 164 validation images.

## 3. Evaluation for the decision (validation only)
- Whole-image inference with the frozen protocol on the 164 validation images; pooled confusion counts; per-class
  IoU; mIoU over 4 classes (shared-evaluator convention).
- Endpoints: mIoU, Dead IoU, Dead recall, foreground Boundary F1 at 2 px (Stage-30 `boundary_metrics_all` code),
  full-image latency (same GPU, same session as an M0 timing).
- The 167-image locked test set is **not** evaluated in Stage B.
- CTG/DeepOrganoid/BrainOrganoid results are not used for any selection.

## 4. Stage B decision rule (M2, seed 42 only)
### 4.1 M0 validation reference (3 seeds, whole-image; from A0)
Source: `A0_gap_decomposition/a0_val_reference_per_seed.csv` (whole-image, 164 validation images, M0 masks only).
| Endpoint | min0 | mean0 | max0 | GO threshold |
|---|---:|---:|---:|---:|
| mIoU | 81.52 | 81.62 | 81.70 | ≥ 82.50 |
| Dead IoU | 43.88 | 44.31 | 44.55 | ≥ 46.55 (see note) |
| Dead recall | 74.18 | 75.20 | 75.73 | ≥ 74.18 (NO-GO if < 69.18) |
| Foreground BF1 2 px | 0.826 | 0.829 | 0.830 | > 0.830 |
Context (not a threshold): nnU-Net validation mIoU 83.33, Dead IoU 48.46, fg BF1 0.868.
Note from A0 (written before any M2 result): 68% of the validation Dead-IoU gap to nnU-Net comes from M0 Dead false
positives far from organoids, not from boundaries; criterion 2 may therefore be structurally hard for M2. Whether to
keep, or replace it by "Dead IoU ≥ min0" (non-inferiority), is for the author to decide **before** Stage B starts.
Decision recorded here: **Amendment 1 adopted by the author on 2026-11-04 (option A), before any M2 training** — see §4.3.

### 4.2 Outcomes (written before any M2 result)
Let max0/min0/mean0 be M0's 3-seed max/min/mean on validation.
- **GO** (ask author to approve Stage C) — all of:
  1. mIoU ≥ max0 + 0.8;
  2. Dead IoU ≥ max0 + 2.0;
  3. Dead recall ≥ min0;
  4. foreground BF1 (2 px) > max0;
  5. full-image latency ≤ 1.5 × M0.
- **NO-GO** (stop; archive as negative result) — any of:
  1. mIoU ≤ mean0;
  2. Dead recall < min0 − 5.0;
  3. non-finite loss or failed run (reported; a single relaunch only if the cause is infrastructure, e.g. GPU eviction).
- **INCONCLUSIVE** — everything else (e.g. mIoU above mean0 but inside the M0 seed range, or mIoU passes but Dead
  does not). Action: report to the author; options are (a) stop, or (b) train **one** more M2 seed (3407) and apply the
  same thresholds to the 2-seed M2 mean. No other change (architecture, hyper-parameters) is allowed.

A single validation improvement is never reported as "M2 is better"; only Stage C can support that.

### 4.3 Amendment 1 (2026-11-04, author decision "option A"; written after A0, before any M2 training)
Reason: A0 showed that 68% of M0's validation Dead-IoU gap to nnU-Net is Dead false positives far (>8 px) from any
GT organoid, while ≈99% of the Alive gap lies within 8 px of GT boundaries. A high-resolution-skip architecture targets
the latter; requiring a +2 pp Dead-IoU gain would test a mechanism M2 does not address. The original rule (GO
criterion 2: Dead IoU ≥ max0 + 2.0 = 46.55) is kept above as history and is reported alongside, but it no longer
decides GO.

**Replacement GO criterion 2 (Dead IoU non-inferiority):** M2 seed-42 Dead IoU ≥ min0 = **43.88**.
- Reference model: M0 (frozen Simplified), seeds 42/3407/2026, `best_miou.pth`.
- Evaluation unit: pooled pixel confusion counts over the 164 whole validation images (whole-image sliding-window
  inference, frozen protocol); Dead IoU = TP/(TP+FP+FN) for class 2.
- Decision type: point estimate of a single seed against M0's observed 3-seed range (no hypothesis test; with one
  seed a test would not be meaningful). The paired M2 − M0(seed 42) difference with a 95% physical-well cluster
  bootstrap CI (10,000 replicates) is **reported descriptively** for every endpoint and does not change the outcome.

Final Stage B GO rule (all required): mIoU ≥ 82.50; Dead IoU ≥ 43.88; Dead recall ≥ 74.18; foreground BF1 (2 px)
> 0.830; full-image latency ≤ 1.5 × M0 (same GPU, same session). NO-GO and INCONCLUSIVE rules of §4.2 unchanged
(NO-GO if mIoU ≤ 81.62, or Dead recall < 69.18, or failed run).

### 4.4 Stage B report contents (validation only; fixed before training)
1. mIoU and the four class IoUs; 2. Alive and Dead precision/recall; 3. foreground and Alive BF1 (2 px; 1/4/8 px
sensitivity); 4. HD95 / ASSD (images where both boundaries exist); 5. small-organoid recall / miss rate (A0 definition:
GT alive/dead components ≥20 px, size tertiles with edges 139/313 px from validation GT; missed <10% covered,
detected ≥50%); 6. parameters, GFLOPs, peak memory, latency; 7. paired M2 − M0(seed 42) differences with well-cluster
bootstrap CIs (descriptive); 8. success and failure examples, chosen by a fixed rule (the 3 images with the largest
and the 3 with the smallest per-image foreground-BF1 change vs M0 seed 42).
Mechanism diagnostics: (a) CNN-branch participation — re-infer M2 with the CNN input zeroed (inference only) and
report the foreground/class flip rate and the metric change; (b) A0 region-swap / error-location tables recomputed
for M2 vs M0 (boundary band B3/B8, small components, far-background Dead false positives).
The 167-image test set, BrainOrganoid and DeepOrganoid/CTG are not used in Stage B.

## 4b. Staged route after Stage B (Amendment 2, 2026-11-04; designs only, nothing implemented or trained)
- **M2 GO →** M2 seeds 3407 and 2026 under the identical protocol (needs approval); the GO rule is then applied to
  the 3-seed mean, and reproducibility across seeds is reported. Only then decide whether M3 is warranted.
- **M2 NO-GO →** archive. If the participation and region diagnostics show no boundary improvement from the CNN
  features, stop this route. Only if the CNN features demonstrably help but fusion appears limiting (e.g. boundary
  band improves while the 112→224 stage, which has no skip, dominates remaining errors) may an M3 proposal be
  submitted, with its own mechanism rationale and stopping rule.
- **M2 INCONCLUSIVE →** §4.2 options (stop, or one extra seed).
- **M3 (candidate, not implemented):** M2's encoder and CNN branch unchanged; decoder replaced by a U-Net-style
  decoder: 2×2 transposed-conv upsampling at each stage; same-scale fusion at 28/56/112 (as M2) **plus** a 224
  full-resolution skip from a 1-stage stem (3→16 ch, stride 1) — note: this stem is a new input path and is listed
  explicitly as part of M3; conv blocks per stage (2× conv-BN-GELU). No deep supervision. Compared M2 vs M3 only.
  Real tensor sizes (probe 2026-11-04): ViT l4/l8/l12 = 768×14×14 each; CNN = 32×112², 64×56², 128×28²;
  decoder outputs 256×28², 128×56², 64×112², 32×224².
- **M4 (candidate, not implemented):** M3 + deep supervision with auxiliary heads at 28/56/112; auxiliary logits are
  **upsampled to 224 and scored against the full-resolution label** (no label down-sampling, so sparse Dead pixels
  cannot vanish); weights 1 / 0.5 / 0.25 / 0.125 (main → coarsest), normalised to sum 1; same Rare-Dead loss.
  Inference uses the main head only. Before any M4 run: report how many Dead pixels/components would vanish under
  nearest-neighbour label down-sampling at each scale, to document why label down-sampling is not used.
  Compared M3 vs M4 only.
- Each step needs separate approval; never more than one new architecture in training at a time; single-seed
  development on validation first, multi-seed only after a pre-set GO. One final candidate is chosen on validation
  only, then frozen before the single locked-test and external evaluation.

## 5. Stage C (only after GO and a second author approval)
- M2 seeds 3407 and 2026; M1 seeds 42/3407/2026.
- Locked-test evaluation once, with the existing hierarchical paired bootstrap (seeds × 154 wells, 10,000 replicates):
  M2 − M0, M2 − M1, M1 − M0; and descriptively vs nnU-Net.
- DeepOrganoid non-inferiority check (frozen p1–p99 preprocessing, Stage-31 endpoints), descriptive; not used for
  selection. Its own ANALYSIS_PLAN before any inference.

## 6. Resources
- GPUs 3/5/7 only (never 2/4); wait when busy; `setsid nohup … < /dev/null`; wait on PIDs; full logs.
- Measured in the smoke test (A800, shared GPU, relative values): M0 → M2: trainable params 60.62 M → 61.29 M (+1.1%); GFLOPs/224 window 47.4 → 49.7 (+4.9%); train step (batch 16, fp32) 0.16 s → 0.15 s; peak train memory 4.2 → 5.8 GB; full-image inference 10.6 s → 10.5 s on a shared GPU (relative only; dedicated A800 earlier ≈0.5 s). All losses finite; gradients reach the CNN branch (`A_design_smoke/smoke_results.json`).
- Estimated cost: Stage B = 1 run × ≈4.5 h (M0 took 257 min per 40k steps on A800) + ≈30 min validation inference.
  INCONCLUSIVE extension = +1 run. Stage C = 5 runs (2 × M2, 3 × M1) ≈ 22 GPU-hours (3 GPUs in parallel ≈ 9 h wall)
  + test evaluation + DeepOrganoid inference (3,456 wells).

## 7. Reporting
All runs (including failures and NO-GO) are kept and reported. No result in this branch overwrites M0's results or the
paper's tables. If reported in the paper, it goes under "architecture exploration" unless the author decides otherwise.
