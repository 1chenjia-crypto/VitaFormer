# M2 ablation: ImageNet encoder init instead of BiomedCLIP (2026-11-12)

Status: plan written 2026-11-12, approved order = (1) M2 w/o Rare-Dead → (2) this → (3) nnU-Net patch-512 (exploratory).
Ablation only; M2 (MODEL_FREEZE_M2_20261108) stays the main model.

## Prior knowledge (stated before running, so the result cannot steer the rule)
Old Full architecture (BAM+MPM, best-val checkpoint, test set; statistical_analysis_20261026/REPORT.md):
ImageNet − BiomedCLIP mIoU +0.45 [−0.92, +1.98] (1+/2− seeds), Dead IoU +2.3 [−1.3, +6.9].
The mean was driven by seed 2026, whose BiomedCLIP run selected step 10k. Reading: no detectable difference.
Expectation for M2: likely no difference. Any outcome is reported.

## Design
- M2 command exactly as Stage B/C (`train_v3_multiscale.py --arch m2 --no_bam --no_mpm`, Rare-Dead ON) plus `--use_imagenet`.
- `--use_imagenet`: timm vit_base_patch16_224 loaded from torchvision ImageNet ViT-B/16
  (~/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth, 346 MB, 2021). Same architecture, same freeze_blocks 4,
  same LR (encoder 1e-5). Only the encoder initial weights change. Code unchanged (hash check vs stageB_frozen_hashes.txt).
- Seeds 42 / 3407 / 2026, split_seed 42, 40k steps.
- Before the first run: 60-step smoke test (log must print "encoder init : ImageNet" and "missing=0, unexpected=0").

## Evaluation (fixed before results)
- final.pth at step 40000 (selection-free, as the frozen M2).
- Primary: validation set, seed-matched M2 − ImageNet, mean of 3 seeds with bootstrap CI (stageD_analysis machinery).
- Endpoints: mIoU; Dead IoU / precision / recall; Alive IoU; fg BF1.
- Interpretation rule: "BiomedCLIP helps" only if the CI excludes 0 AND all 3 seeds agree in direction.
  Otherwise the paper says the encoder initialisation source has no detectable effect, and makes no BiomedCLIP-superiority claim.
- Test set: descriptive only.

## Hardware
P100 16 GB works (M2 peak ~5 GB, ~0.76 s/step → ~9–10 h/run). Preferred: A800 after the classic baselines finish.
2026-11-12 smoke test (60 steps, A800 GPU0): ImageNet loaded missing=0 unexpected=0, encoder init ImageNet, focal on, TRAINING_COMPLETE -> PASS

## Amendment 1 (2026-11-13, author approved): resume after NFS outage
- NFS storage outage (dmesg "nfs4_reclaim_open_state: Lock reclaim failed"): A800 runs hung from ~02:38 and exited 07:07 with
  OSError [Errno 5] Input/output error while reading a training JPEG (file intact and readable afterwards).
- Affected: imagenet s42 (stopped 38700; resume from 38000), imagenet s2026 (39000; 38000), imagenet s3407 (32000; 32000),
  noRD s3407 (39600; 38000). Not affected: noRD s42 (P100, done 00:46), noRD s2026 (done 02:31).
- Resume = same command + `--resume_full checkpoint_resume_preoutage.pth` (model, optimizer, scheduler, step, history restored;
  data-loader RNG not restored → batch order after the resume point differs from an uninterrupted run).
  Pre-outage checkpoints kept as checkpoint_resume_preoutage.pth; script ../m2_ablation_imagenet_20261112/resume_one.sh;
  logs logs/*.resume1.full.log. A800 GPUs 0/1/3/4, 2026-11-13 08:50. Evaluation rule unchanged (final.pth @40000).
- Methods will state which runs were resumed.
