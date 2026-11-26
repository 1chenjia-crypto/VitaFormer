# BrainOrganoid2024 — M2 arm amendment (2026-11-08, written before any M2 Brain result)

Base plan: `ANALYSIS_PLAN_BASE_20261103.md` (verbatim copy of the 11-03 frozen plan). Everything in it applies unless changed here.
Reason: the author froze VitaFormer M2 as the main model (`3D_OrganoidSeg/model_freeze_m2_20261108/MODEL_FREEZE_M2_20261108.md`).

1. **New arm "VitaFormer M2"**: script `scripts/train_brain_m2_20261108.py` = the 11-04 script with only the model constructor
   (`OrganoidSegMultiScale`, zero_cnn_input=False) and source checkpoint changed (M2 seed42 `final.pth`, SHA `0f12be44…6101`;
   all non-head tensors transferred, `seg_head` reinitialised to 2 classes). Diff: `model_freeze_m2_20261108/train_brain_m2_vs_20261104.diff`.
   Same folds (official `splits.csv`, SHA `493c7635…`, copied here as `official_splits.csv`), crops, augmentation, loss, optimiser
   (encoder 2e-6 / rest 2e-5, incl. the new CNN branch), 40k steps, validation every 4k on the full official validation partition,
   checkpoint = best validation image-level mean Dice, threshold 0.5, no TTA, native-resolution sliding window.
2. **Arms in the final comparison**: M2 (main), M0 (= the 11-04 "VitaFormer simplified" arm; fold0 exists, folds 1–4 to be run with
   the unchanged 11-04 script), nnU-Net (5/5 folds done), SegFormer-B2 (secondary, if completed), author SegFormer MiT-B0 OOF
   (literature reference row, paper 0.960 ± 0.040).
3. **Primary contrasts** (replace those of the base plan): M2 − nnU-Net and M2 − M0, image-level Dice over the 1,407 OOF images,
   organoid-ID cluster bootstrap (10,000), Holm for the two. M2 − SegFormer-B2 and M2 − author OOF are secondary.
4. **Execution order**: M2 folds 0–4 first (author instruction), then M0 folds 1–4, then SegFormer-B2 folds 1–4 if approved.
   No result of one arm changes the protocol of another. A fold that crashes is rerun into a new directory with the same command;
   crash and rerun are reported.
5. **Reporting**: all results are reported whether or not M2 is better; no change to M2 follows from them.
