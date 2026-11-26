# VitaFormer final model freeze — M2 (2026-11-08)

**Author decision (2026-11-08):** VitaFormer **M2** (BiomedCLIP ViT-B/16 encoder + from-scratch high-resolution CNN branch,
No BAM, No MPM, Rare-Dead training, argmax inference) is the paper's final, frozen main model.
This record **supersedes** `3D_OrganoidSeg/ARCHITECTURE_FREEZE_20261029.md` (M0, seed42 best_miou) as the main-model
definition. The 10-29 file is kept unchanged; M0 remains in the paper as the architecture-ablation reference.

## 1. Frozen from now on (author rule)
- No new architecture, loss, threshold/post-processing, BAM/MPM or Rare-Dead ablation training.
- No relabeling / no removal of disputed-label images / no re-split of the CRC dataset.
- External and DeepOrganoid results are reported whatever they show and are **not** used to change M2.
- External datasets use the **source paper's split and metric definition** (author rule 2026-11-08).
- Training M2 on an external dataset under "protocol B" (source split, same recipe for all compared models) is an
  evaluation protocol, not a change of M2; it must be pre-registered in that dataset's ANALYSIS_PLAN.

## 2. Checkpoints (all `final.pth`, step 40000, verified 2026-11-08: 0 non-finite values)
| Seed | Path (relative to `vco/runs/arch_exploration_20261104/`) | SHA256 | Training hardware |
|---|---|---|---|
| 42 | `stageB_m2_seed42/final.pth` | `0f12be44859d06723ebfa35e8416df8cd4b02ff83f3ac153654d0f68df226101` | A800 (TF32 conv default) |
| 3407 | `stageC_m2_seed3407/final.pth` | `1e4bd6afd0a1925094103c991430246b9f1fef09ce5b480e2c47056a7128f585` | A800 |
| 2026 | `stageC_m2_seed2026/final.pth` | `c268464ee2ae3ee19e2112fc91ae86a95aa5d95086374df526fda9a7aa3225b8` | V100S (pure fp32), Amendment 3b |

Hashes are identical to `stageF_frozen_hashes.txt` (used for the Stage F test run). Checkpoint size 361,037,930 bytes each.
Pretrained encoder: `~/.cache/biomedclip/open_clip_pytorch_model.bin` SHA256 `52cc993c…576be`.
Full machine-readable manifest (code, configs, data splits, env): `freeze_manifest.json` (built by `make_freeze_manifest.py`),
pip environment: `env_pip_freeze.txt` (venv_organoid: Python 3.11.5, torch 2.6.0+cu124).

## 3. Model definition
`vco/organoid_seg/model_v3_multiscale.OrganoidSegMultiScale(weight_path=BiomedCLIP, num_classes=4, zero_cnn_input=False)`:
BiomedCLIP ViT-B/16 taps at layers 4/8/12 (14×14) → M0 decoder; plus a CNN branch (3 stride-2 stages, 32/64/128 ch,
112/56/28 px) concatenated as extra skips into dec3/dec2/dec1. No BAM, no MPM. 4 classes (background, Alive, Dead, Bubble).
Trainable parameters 61.29 M. The `morph_head` is not part of M2 and no morphology claim is made.

## 4. Training protocol (identical across seeds except `--seed`)
`train_v3_multiscale.py` (= `train_v2_stratified.py` + `--arch` flag only):
`--lr 1e-4 --lr_encoder 1e-5 --max_steps 40000 --val_every 2000 --freeze_blocks 4 --focal_loss --gamma 2.0
--dead_weight_min 10 --use_dead_center_crop --dead_pixel_dropout_prob 0.0 --sparse_dead_zoom_prob 0.8
--sparse_dead_threshold 5000 --sparse_dead_zoom_factor 2 --oversample_5fu_factor 3 --use_condition_split
--seed {42,3407,2026} --split_seed 42 --no_bam --no_mpm --arch m2`
(seed42: `stageB_m2_seed42/launch_command.sh`; 3407: `scripts/stageC_queue.sh` run_job; 2026: `scripts/stageC_run_one.sh`).
Rare-Dead = the six bundled interventions (focal γ=2, Dead weight ≥10, Dead-centre crop, sparse-Dead zoom, 5-Fu oversampling,
condition-stratified split). Split (condition-stratified, split_seed 42): train 481 / val 164 / test 167 images
(`OrganoidSegBaseline/VOC2012/ImageSets/Segmentation/{train,val,test}_cond.txt`, SHA256 in manifest).

**Checkpoint rule = fixed last step (final.pth, 40k).** History, stated as it happened: the original rule (best_miou on a single
random 224 crop per validation image) selected 6k–18k steps for Stage C and gave M2 < M0 (Stage C). Amendment 4
(2026-11-06, written after Stage C was seen) switched all arms to selection-free final.pth; this is a **post hoc** rule
change and must be reported as such.

## 5. Inference protocol (frozen)
`infer_hard` in `scripts/a0_m0_val_inference.py`: RGB, /255, ImageNet mean/std normalisation, reflect-pad, 224 window /
112 stride sliding window, fp16 autocast for the forward pass, softmax averaged over overlaps, **argmax** (no threshold,
no TTA, no post-processing). 3 seeds are reported as mean ± SD of per-seed results (no ensemble) unless a plan says otherwise.

## 6. CRC evidence at freeze (no new runs)
Validation (164 img, whole image, Stage D/G): M2 mIoU 82.62 ± 0.22 (82.4–82.9), M2 − M0 +0.83 [0.54, 1.12], 3/3 seeds;
Alive +1.7, BF1 +0.021; argmax is the validation optimum, cross-validated thresholding −0.16 mIoU (keep argmax).

Locked test (167 img / 154 wells, Stage F, `stageF_m2_test/RESULTS.md`; test set was **not blind**: M0 and nnU-Net had been
evaluated on it earlier):

| Seed | mIoU | Alive | Dead | Bubble | Dead R | Dead P |
|---|---|---|---|---|---|---|
| 42 | 83.54 | 86.10 | 49.91 | 98.24 | 78.7 | 57.7 |
| 3407 | 83.05 | 86.09 | 49.98 | 96.22 | 77.5 | 58.5 |
| 2026 | 83.42 | 86.12 | 49.47 | 98.15 | 77.2 | 58.0 |
| **Mean ± SD** | **83.34 ± 0.25** | 86.10 | 49.79 | 97.54 | 77.8 | 58.0 |

Paired hierarchical bootstrap: M2 − nnU-Net −1.71 [−2.90, −0.04] (nnU-Net significantly better; Dead P −14.1);
M2 − M0-final +0.56 [−0.11, 0.90] (3/3 positive, CI touches 0), Alive +1.84 [1.66, 2.11]; M2 − SegFormer-B2 +0.42 n.s.

## 7. Known limitations to carry into the paper
1. nnU-Net (from scratch) remains significantly better on CRC; M2 uses a pretrained encoder (BiomedCLIP).
2. Rare-Dead benefit (AUPRC +0.10, matched-recall precision +7–11 pp) was shown on the Full architecture only (Stage G);
   its effect in M2 is inferred, not directly tested (no M2-noRD by author rule).
3. M2 over-calls Dead at argmax (precision ≈ 58%); far-background Dead FP unchanged vs M0.
4. Checkpoint rule changed post hoc (§4); test not blind (§6).
5. Seed 2026 trained on V100 (fp32), others on A800 (TF32).
6. All DeepOrganoid / external results so far used M0 or Full → must be re-run with M2 (planned next).

## 8. Next steps (in order)
1. External generalisation with M2 first (source-paper splits): BrainOrganoid2024 (5-fold), OrgaSegment, OrganoIDNetData;
   then M0 and nnU-Net under the same protocol. Each dataset gets an ANALYSIS_PLAN before any result.
2. DeepOrganoid functional validation with M2 (Stage 31/34 protocol); M0 results kept unchanged for reference.
3. Paper writing (internal performance, architecture ablation, cross-domain, drug-function evidence chain).
