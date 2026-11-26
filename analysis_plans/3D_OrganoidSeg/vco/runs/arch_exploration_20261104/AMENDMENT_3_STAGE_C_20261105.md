# Amendment 3 — Stage C execution (written 2026-11-05, before any Stage C run)
Author approval (2026-11-05): train M2 seeds 3407, 2026 and M1 seeds 42, 3407, 2026. M3/M4, deep supervision,
Rare-Dead / loss changes: not allowed.
Deviations from plan §5:
1. Locked CRC test evaluation and DeepOrganoid evaluation are DEFERRED (author instruction). Stage C is validation only.
2. Training: identical to `stageB_m2_seed42/launch_command.sh` except `--seed`, `--arch` (m1|m2), exp_name/out_dir.
   `--split_seed 42` kept for all runs (same split as M0 and Stage B). M1 = `zero_cnn_input=True` from step 0 (independent training).
   Checkpoint = best_miou.pth (random-crop val), as for M0 / Stage B.
3. Per-run inference uses `scripts/stageB_evaluate.py` unchanged (its 'outcome' field and "vs M0 seed42" files are
   by-products, NOT Stage C decisions). Stage C analysis = separate script, seed-matched pairs
   (M2−M1, M1−M0, M2−M0 per seed), image-level and 154-well-cluster paired bootstrap per seed, 3-seed means;
   descriptive vs nnU-Net (checkpoint-selection caveat). No new GO rule; report to the author.
4. Scheduling: GPUs 3/5/7 only, a GPU is used only if it has no compute process and <1 GB used on two checks 3 min apart.
   Order: M2 s3407, M2 s2026, M1 s42, M1 s3407, M1 s2026. An existing out_dir is never reused. No automatic retry; an
   infrastructure failure is reported and relaunched once into a new `_r2` directory only with the author informed.

## Amendment 3a (2026-11-05 15:0x, before any Stage C run started)
Author approved adding GPU 1 (GPUs 3/5/7 all occupied by other users). Allowed GPUs now 1/3/5/7; 2 and 4 still never.
The first queue (GPUs 3/5/7, no job had started) was stopped by PID and relaunched with this change only;
its log kept as logs/stageC_queue_attempt1_gpus357_stopped.full.log. Hashes re-frozen in stageC_frozen_hashes.txt.

## Amendment 3b (2026-11-05 ~15:15, author approved)
Node GPU-V100-2 (2x Tesla V100S 32 GB, both idle) added. M2 seed 2026 -> V100 GPU 0, M1 seed 42 -> V100 GPU 1, via
scripts/stageC_run_one.sh (identical training/eval command). Their out_dirs are created now so the A800 queue
(unreachable from this node; still waiting on M2 seed 3407) skips them. Remaining on the A800 queue: M2 3407, M1 3407, M1 2026.
Hardware differs from M0/M2-s42 (A800, TF32 convs by default) -> V100 pure fp32; recorded per run; latency comparisons
only within the same hardware and relative.

## Amendment 3c (2026-11-05 15:24, author approved "schedule the remaining jobs on free GPUs")
A800 queue started M2 seed 3407 on A800 GPU 3 at 15:14:41. M1 seeds 3407 and 2026 claimed (out_dirs created 15:24:29,
before the A800 queue's next existence check at ~15:24:41) and launched on node GPU-V100-1 (2x V100S, idle):
M1 s3407 -> GPU 0, M1 s2026 -> GPU 1, via scripts/stageC_run_one.sh (unchanged). A800 queue should log SKIP for all remaining jobs.
Hardware per run: M2 s3407 A800; M2 s2026, M1 s42 V100-2; M1 s3407, M1 s2026 V100-1.
