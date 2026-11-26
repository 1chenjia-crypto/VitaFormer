# Label Quality Audit (LQA) — protocol, fixed 2026-11-06 BEFORE any A-extension / B result (author approved)
Original GT stays the primary benchmark. No GT is modified, no image removed, no prediction used as label.
All model-defined regions are called **potential annotation conflicts (PAC)**, never confirmed errors.
Formal sensitivity analysis = CRC VALIDATION only (164 images). Test = descriptive audit only (already produced,
selected with test predictions; never used for selection / thresholds / method design).

## Model families and predictions (existing frozen checkpoints only)
- M0 Simplified VitaFormer: best_miou.pth (A0 val masks), 3 seeds.  - nnU-Net v2 run2: checkpoint_final fold_0/validation, 3 seeds.
- SegFormer-B2 (original runs, segformer_b2_runs/seed*): best_val_miou.pth; val masks produced by
  `segformer_val/segformer_val_inference.py` using the script's own `predict` (512/256 sliding window, the protocol
  that produced its test masks); consistency check = 2 test images re-predicted must equal existing test_masks.
- M1/M2 are not part of the LQA (not benchmark models).

## Definitions (fixed; no tuning after results)
- Family-unanimous class at a pixel: all 3 seeds of the family predict the same class c.
- PAC(S) for a set of families S: every family in S is family-unanimous on the same class c, GT is valid (0..3) and GT != c.
- Location: B8 = within 8 px (EDT) of a GT label transition (as in A0); FARBG = >8 px from any GT non-background pixel; other = rest.
- Missed-object candidates: 8-connected components of PAC pixels with GT=background and consensus in {alive, dead},
  area >= 20 px; size bins by A0 validation edges 139 / 313 px.
- Aggregation per image and per well (parse_filename well_id).

## A. Audit (val with S = all 3 families; test descriptive with S = all 3 families, 9 predictions)
Per image / well / class: PAC pixel fraction by GT-class -> consensus-class, BG->Dead vs BG->Alive, location split,
missed-object candidate counts/areas/size bins, inter-family disagreement.

## B1. Leave-one-family-out sensitivity (val)
For family X, ignore PAC(other two families); evaluate each seed of X on original GT vs GT with ignore region.
Report original and masked metrics (mIoU, per-class IoU, Dead P/R), excluded pixel count/fraction, class mix of excluded pixels.
Ignore regions differ between families -> NOT a ranking.

## B2. Common-pixel paired sensitivity (val)
Pair (X, Y) evaluated on the SAME pixel set: ignore PAC({Z}), Z = third family:
M0 vs nnU-Net (Z = SegFormer); M0 vs SegFormer (Z = nnU-Net); nnU-Net vs SegFormer (Z = M0).
Seed-matched differences, original vs masked, 10,000 well-cluster paired bootstrap (as Stage C).
Report mask coverage and class distribution; results depend on the rule; NOT corrected-GT accuracy; NOT for model selection.

## C. Noise-robust training: written feasibility assessment only. No OOF training, no auto-relabelling, no M3/M4.
