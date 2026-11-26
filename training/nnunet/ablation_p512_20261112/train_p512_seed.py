#!/usr/bin/env python3
"""nnU-Net patch-512 ablation. Identical to ../nnunet_train_seed.py (run 2) except plans_identifier='nnUNetPlans_p512'
and the results root (nnunet_ablation_p512_20261112/nnUNet_results_p512_seed{N}). Run with env_totalseg2 python."""
import argparse, os, random
from pathlib import Path
BASE = Path('3D_OrganoidSeg/baselines_fair_20261030/nnunet')
HERE = Path('3D_OrganoidSeg/baselines_fair_20261030/nnunet_ablation_p512_20261112')
ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); args = ap.parse_args()
res = HERE / f'nnUNet_results_p512_seed{args.seed}'; res.mkdir(exist_ok=False)
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS'): os.environ[k] = '1'
os.environ['nnUNet_compile'] = 'f'
os.environ['nnUNet_raw'] = str(BASE / 'nnUNet_raw'); os.environ['nnUNet_preprocessed'] = str(BASE / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(res)
import numpy as np, torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
from nnunetv2.run.run_training import run_training
print(f'seed={args.seed} plans=nnUNetPlans_p512 nnUNet_results={res}', flush=True)
run_training('801', '2d', 0, trainer_class_name='nnUNetTrainer', plans_identifier='nnUNetPlans_p512', device=torch.device('cuda'))
