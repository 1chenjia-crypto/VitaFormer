# Environment and licences

## Environments (pinned files in this folder)
| File | What |
|---|---|
| `env_pip_freeze_training_venv_organoid.txt` | full pip freeze of the env that trained/inferred all paper M2 results (Python 3.11.5, torch 2.6.0+cu124) |
| `requirements_tool.txt` | minimal pinned set for the tool (torch 2.6.0, torchvision 0.21.0, timm 1.0.27, numpy 1.26.4, scipy 1.11.4, pandas 2.2.3, pillow 10.4.0) |
| `requirements_cpu.txt`, `requirements_gpu_cu124.txt` | installer variants used by `install.sh` / `install_windows.bat` |
| `fresh_env_pip_freeze_tool_20261110.txt` | pip freeze of the clean-install test (tool passed 25/25 + 19/19 tests there) |

Notes for the software plan's package list:
- **Not needed for inference:** `open_clip_torch` (the model uses timm's ViT; BiomedCLIP weights are only for training init),
  `tifffile`, `scikit-image` (TIFF is read with Pillow; connected components with scipy). Add them only if the new software uses them.
- UI: v1.1.0 uses only the Python standard library (`http.server`), no UI framework.
- `torchvision` must come from the same build/index as `torch` (CPU vs CUDA), otherwise import fails (found 11-10).
- oreval: Python ≥ 3.10, numpy, scipy, pandas, pillow only.

## Licences — status 2026-11-24 (⚠️ = must be confirmed by the authors before public release)
| Component | Licence | Status |
|---|---|---|
| Own code (VitaFormer model/training/inference, vitaformer_tool, oreval) | **not yet chosen**; no LICENSE file exists in any of the three | ⚠️ choose (e.g. MIT/Apache-2.0) |
| Trained M2 weights | follows own choice + BiomedCLIP terms (weights were initialised from BiomedCLIP) | ⚠️ |
| BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (Microsoft) | MIT according to its Hugging Face model card | ⚠️ licence text is NOT in the local cache; download the card/LICENSE and ship it |
| PyTorch, torchvision | BSD-3-Clause | standard |
| timm | Apache-2.0 | standard |
| numpy, scipy, pandas | BSD-3-Clause | standard |
| Pillow | MIT-CMU (HPND) | standard |
| open_clip (only if used for training) | MIT | standard |
| NViR example images (Zenodo 15374443) | MIT; notice in `NViR_example_LICENSE_AND_SOURCE.txt` | must ship with the images |
| CRC images (gold-standard set, tool `examples/`) | ⚠️ unconfirmed; v1.1.0 did **not** distribute them | do not put in a public release until cleared |
