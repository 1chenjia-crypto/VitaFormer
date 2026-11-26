# Stage F — frozen M2 on the CRC TEST set (167 images / 154 wells), 2026-11-06
Written before any M2 test inference. User-approved (2026-11-06): "跑M2测试集推理补齐表1". Inference only; no training.

## Frozen model (no further changes)
- M2 = OrganoidSegMultiScale (BiomedCLIP + high-res CNN branch), final.pth step 40000, seeds 42/3407/2026
  (paths + sha256 in stageF_frozen_hashes.txt). Checkpoint rule = final.pth (Stage D, Amendment 4); chosen on validation only.
- Inference = infer_hard from scripts/a0_m0_val_inference.py unchanged (224 window / 112 stride, reflect pad, ImageNet norm,
  softmax averaging, argmax, autocast). No TTA, no post-processing.
- Evaluation = baselines_fair_20261030/eval_masks_shared.py unchanged (original GT, pooled pixels, 4-class mIoU).
- Statistics = scripts/unified_comparison_20261102.py unchanged (hierarchical paired bootstrap, 3 seed positions × 154 wells,
  10,000 replicates, RNG seed 20261102); only GROUPS / CONTRASTS / OUT replaced.

## Known limitation (stated before results)
The CRC test set is NOT blind: test metrics of M0/Full/SegFormer/nnU-Net and test masks/label quality were seen earlier.
M2 test numbers are therefore a frozen-model evaluation, not a pristine held-out test. Test results will NOT be used to
choose a network, checkpoint, threshold or post-processing, and do not trigger further architecture work.

## Arms
- M2-final (new inference, 3 seeds) — main.
- M0-final (new inference, 3 seeds, final.pth) — matched-checkpoint-rule comparator for M2−M0 (M0's frozen test numbers use
  best_miou.pth; Stage D showed selection matters). The frozen M0 (best_miou) numbers are unchanged and remain the reference.
- Existing JSONs: VF-Simplified (M0 best_miou), SegFormer-B2, SegFormer-B2-stable, nnU-Net (no TTA, primary).

## Contrasts (A − B), all reported whatever the result
1. M2-final − nnU-Net
2. M2-final − M0-final (matched rule)  [primary architecture contrast]
3. M2-final − VF-Simplified (frozen M0, best_miou)
4. M2-final − SegFormer-B2
5. M2-final − SegFormer-B2-stable
6. M0-final − VF-Simplified (checkpoint-rule effect on test; descriptive)
Endpoints: mIoU, mDice, Alive IoU, Dead IoU, Bubble IoU, Dead recall, Dead precision.

## Consistency checks before inference
- M0 best_miou.pth re-predicts 2 test images vs existing P0-M masks (expect ≈100%).
- M2 final.pth re-predicts 2 validation images vs Stage D final_masks (expect ≈100%).
