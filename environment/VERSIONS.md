# Software versions (as used)

| Environment | Used for | Python | PyTorch | Key packages |
|---|---|---|---|---|
| training (`env_pip_freeze_training_venv_organoid.txt`) | VitaFormer training/inference, SegFormer-B2, analyses | 3.11.5 | 2.6.0+cu124 (CUDA 12.4) | timm 1.0.27, open_clip_torch 3.3.0, transformers 5.16.1, numpy 1.26.4, pandas 2.2.3, scipy 1.11.4, scikit-learn 1.7.2 |
| baselines | U-Net, U-Net++, DeepLabv3 | 3.11.5 | 2.6.0+cu124 | see training scripts |
| nnU-Net | nnU-Net v2 | 3.10.4 | 2.5.1+cu124 | nnunetv2 2.8.1 |
| figures | figure and table scripts | 3.10.4 | 2.5.1+cu124 | matplotlib, pandas |
| OrganoidViT (`organoidvit_env_freeze.txt`) | OrganoidViT reproduction | 3.6.8 | 1.7.1+cu110 | mmseg-based author code (not redistributed) |
| tool (`requirements_*.txt`, `fresh_env_pip_freeze_tool_20261110.txt`) | desktop tool v1.1.0 | 3.11 | CPU or cu124 | see requirements |
