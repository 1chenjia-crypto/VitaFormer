# Amendment 2 (2026-11-08 ~14:00, before any M2 Brain test result): M0 arm cancelled
Author decision: run only M2 on BrainOrganoid2024; M0 folds 1–4 are not run.
Mechanism: `m0/fold{1..4}/CANCELLED.txt` placeholders, so the queued M0 jobs in `run_v100_1_queue.sh` refuse to start
(FileExistsError, logged as EXIT 1 = cancelled, not a failure).
Consequences for AMENDMENT_M2_20261108.md §3: primary contrast = **M2 − nnU-Net** only (no Holm needed for one contrast);
M2 − SegFormer-B2 / author SegFormer OOF secondary. The existing M0 fold0 (11-04 run, fold0 only) may be shown descriptively
for fold0, not as a five-fold OOF contrast. M0 vs M2 evidence stays CRC-only (Stage D/F).
