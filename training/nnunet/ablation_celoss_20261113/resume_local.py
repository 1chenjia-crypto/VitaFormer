#!/usr/bin/env python3
"""Resume after repeated NFS outages (2026-11-13 10:52; author-approved resume rule, Amendment 1/3).
Identical to the original launchers (train_p512_seed.py / ../nnunet_ablation_celoss_20261113/train_celoss_seed.py) except:
  * continue_training=True (official nnU-Net --c, from checkpoint_latest);
  * nnUNet_preprocessed = local RAM-disk copy /dev/shm/user_nnUNet_preprocessed (byte-identical, 2585 files sha256-verified).
Usage: resume_local.py {p512|celoss} SEED"""
import os, random, sys
from pathlib import Path
kind, seed = sys.argv[1], int(sys.argv[2])
B = Path('3D_OrganoidSeg/baselines_fair_20261030'); BASE = B / 'nnunet'
HERE = {'p512': B / 'nnunet_ablation_p512_20261112', 'celoss': B / 'nnunet_ablation_celoss_20261113'}[kind]
res = HERE / f'nnUNet_results_{kind}_seed{seed}'; assert res.is_dir()
trainer, plans = {'p512': ('nnUNetTrainer', 'nnUNetPlans_p512'), 'celoss': ('nnUNetTrainerCELoss', 'nnUNetPlans')}[kind]
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS'): os.environ[k] = '1'
os.environ['nnUNet_compile'] = 'f'
os.environ['nnUNet_raw'] = str(BASE / 'nnUNet_raw'); os.environ['nnUNet_preprocessed'] = '/dev/shm/user_nnUNet_preprocessed'
os.environ['nnUNet_results'] = str(res)
import numpy as np, torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
from nnunetv2.run.run_training import run_training
print(f'RESUME kind={kind} seed={seed} trainer={trainer} plans={plans} pre={os.environ["nnUNet_preprocessed"]} results={res}', flush=True)
run_training('801', '2d', 0, trainer_class_name=trainer, plans_identifier=plans, device=torch.device('cuda'), continue_training=True)
