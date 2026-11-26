# Amendment 4 — POST HOC selection-free sensitivity analysis (written 2026-11-06, AFTER seeing Stage C results)
Status: post hoc sensitivity analysis proposed after Stage C revealed checkpoint-selection confounding
(random-crop best_miou picked 6k–18k steps for Stage C vs 28k–36k for M0). NOT a pre-registered primary experiment.
Author approved 2026-11-06.
- Checkpoints: final.pth of all 9 runs (M0/M1/M2 x seeds 42/3407/2026); verified step == 40000 for all 9.
- No training; no change to models, losses, Rare-Dead, optimiser; existing checkpoints only read.
- Inference: infer_hard unchanged (a0_m0_val_inference.py; 224/112, reflect, ImageNet norm, softmax avg, argmax, autocast),
  164 CRC validation images. M1 = zero_cnn_input=True (as in training). Pipeline check per run: 2 val images re-predicted
  with best_miou.pth must equal the existing best-checkpoint masks.
- Analysis: scripts/stageD_analysis.py = stageC_analysis.py with mask source switched (--ckpt best|final); same metrics,
  same 154-well bootstrap. Reported: best vs final per seed, M2−M0, M2−M1, M1−M0 per seed and mean.
- GPUs: only truly free among 1/3/5/7. CRC test / external data not read. M3/M4 not started.
