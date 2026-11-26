# Third-party components

The installer bundles the following components. Their licences apply to those components.

| Component | Version | Licence |
|---|---|---|
| Python (embeddable distribution) | 3.12.0 | Python Software Foundation License |
| PyTorch (CPU build), torchvision | 2.6.0, 0.21.0 | BSD-3-Clause |
| timm | 1.0.27 | Apache-2.0 |
| NumPy, SciPy, pandas | 1.26.4, 1.11.4, 2.2.3 | BSD-3-Clause |
| Pillow | 10.4.0 | MIT-CMU (HPND) |
| Microsoft Edge WebView2 SDK | see `provenance/WebView2_SDK_LICENSE.txt` | Microsoft licence terms, `provenance/WebView2_SDK_NOTICE.txt` |
| BiomedCLIP (Microsoft), used only to initialise the encoder during training | — | MIT (model card on Hugging Face) |

Further packages installed as dependencies of the above are listed with their versions in `provenance/desktop_runtime_lock.txt`.

Example images:
- `examples/nvir/` — NViR dataset, Zenodo record 15374443, MIT licence (see the folder's `SOURCE_AND_LICENCE.md`).
- `examples/crc/` — see the folder's `SOURCE_AND_LICENCE.md`.
