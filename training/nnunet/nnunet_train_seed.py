#!/usr/bin/env python3
"""nnU-Net v2 2D, Dataset801 fold 0 (custom split 481 train / 164 val; test never seen), one fixed seed.

Seeding: python/numpy/torch seeds are set in the main process before nnU-Net's official `run_training`, which
fixes network initialisation and the main-process sampling RNG. nnU-Net's augmentation workers are created
with seeds=None (upstream behaviour), so augmentation is not reproducible; the 3 runs are independent repeats
with fixed init seeds. Each seed writes to its own nnUNet_results root so runs cannot overwrite each other.
Official configuration otherwise (nnUNetTrainer, nnUNetPlans, 1000 epochs, checkpoint_final).

Run 2 (2026-10-31): the first launch crashed at epoch 0. This launcher now mirrors the official nnUNetv2_train
entry point (OMP/MKL/OPENBLAS threads = 1, torch intra/inter-op threads = 1); without that, every process opened
~64 threads and hit the 4096 per-user limit. torch.compile is disabled (nnUNet_compile=f) because the system gcc
4.8.5 lacks stdatomic.h and cannot build Triton kernels; this affects speed only, not the training recipe.
"""

import argparse
import os
import random
from pathlib import Path

BASE = Path('3D_OrganoidSeg/baselines_fair_20261030/nnunet')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--results-suffix', default='', help="e.g. '_run2' -> nnUNet_results_run2_seed{N}")
    args = ap.parse_args()
    res = BASE / f'nnUNet_results{args.results_suffix}_seed{args.seed}'
    res.mkdir(exist_ok=bool(args.results_suffix == ''))
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS'):
        os.environ[k] = '1'  # as in nnunetv2.run.run_training.__main__
    os.environ['nnUNet_compile'] = 'f'
    os.environ['nnUNet_raw'] = str(BASE / 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = str(BASE / 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = str(res)  # must be set before importing nnunetv2 (paths read at import)
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)  # as in run_training_entry for device cuda
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    from nnunetv2.run.run_training import run_training
    print(f'seed={args.seed} nnUNet_results={res}', flush=True)
    run_training('801', '2d', 0, trainer_class_name='nnUNetTrainer', plans_identifier='nnUNetPlans',
                 device=torch.device('cuda'))


if __name__ == '__main__':
    main()
