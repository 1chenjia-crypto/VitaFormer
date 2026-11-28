# VitaFormer — bright-field organoid segmentation (desktop application and paper code)

**Version 1.1.0** · Windows 10/11 (64-bit) · runs offline on the CPU · no Python installation needed

VitaFormer segments bright-field images of organoids into four classes — background, **Alive**, **Dead** and **Bubble** —
and reports image-level measurements such as foreground area, region count and region shape. Measurements can be expressed
relative to control wells defined in a plate map. The application contains the frozen VitaFormer model (training seed 42)
described in the accompanying paper.

> **Important.** *Alive* and *Dead* are model classes. The Dead class was **not** validated against an external cell-death
> assay and must not be reported as a cell-death or viability percentage. Region counts are connected regions, not organoid
> counts. All sizes are in pixels. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Repository contents

This repository contains the Windows desktop application (documentation, examples and checks; the installer itself is
archived on Zenodo, https://doi.org/10.5281/zenodo.24554415) and the code used for the accompanying paper. The installer contains the frozen VitaFormer
weights of training seed 42; the weights of all three training seeds and the result tables will be added to the same
Zenodo record in a later version. No third-party image data are distributed.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24554415.svg)](https://doi.org/10.5281/zenodo.24554415)

| Folder | Content |
|---|---|
| `docs/`, `examples/`, `provenance/`, `installer/` | Desktop application: user guide, limitations, validation report, model card, example sets with a one-click check, pinned runtime versions |
| `vitaformer/` | VitaFormer model definition, inference code used for the paper masks, desktop/command-line tool source (v1.1.0) and frozen model configuration |
| `training/` | Training scripts and configurations for VitaFormer (including ablation flags), SegFormer-B2, nnU-Net, the source-publication baselines and fine-tuning on external datasets |
| `oreval/` | Evaluation workflow for uncalibrated segmentation-derived readouts, configuration examples and a synthetic example with expected outputs |
| `analysis/` | Analysis scripts per dataset and the scripts that draw every figure and supplementary table |
| `analysis_plans/` | Time-stamped analysis plans and amendments, written before the corresponding analyses |
| `environment/` | Package versions and third-party licence notes |

The paper code was run on a Linux server; paths are relative to a workspace root described in `PATHS.md`. `M2` is the
internal name of the frozen VitaFormer model and still appears in file names, variable names and the time-stamped plans.

---

## Contents of the desktop-application release

| Folder / file | What it contains |
|---|---|
| `installer/` | `SHA256SUMS.txt` of `VitaFormer_Setup_1.1.0.exe` (the installer itself is on Zenodo) |
| `examples/` | Example images with expected results and a one-click check (`check_examples.bat`) |
| `docs/` | User guide, limitations, validation report, model card, inference specification |
| `provenance/` | Model configuration, pinned runtime versions, third-party notices |
| `CHANGELOG.md` | Version history |
| `CITATION.cff` | How to cite VitaFormer |
| `THIRD_PARTY_LICENSES.md` | Licences of bundled components |

## System requirements

- Windows 10 or 11, 64-bit.
- About 2 GB of free disk space (recommended).
- No GPU and no internet connection are needed. This build uses the CPU only.
- Microsoft Edge WebView2 Runtime (already present on most Windows 10/11 computers; the installer adds it if it is missing).

## Install

1. Download `VitaFormer_Setup_1.1.0.exe` from the Zenodo record and, optionally, check its SHA256 against `installer/SHA256SUMS.txt`
   (PowerShell: `Get-FileHash .\VitaFormer_Setup_1.1.0.exe -Algorithm SHA256`).
2. Double-click the installer and follow the steps. Tick **Create a desktop shortcut** if you want one.
3. The installer is not code-signed. If Windows shows *"Windows protected your PC"*, click **More info → Run anyway**.

The application is installed for the current user (default: `%LOCALAPPDATA%\Programs\VitaFormer`). It can be removed in
**Settings → Apps → Installed apps → VitaFormer** or from the Start menu entry *Uninstall VitaFormer*. Uninstalling does
not delete your result folders.

## Quick start

1. Start **VitaFormer** from the Start menu or the desktop shortcut. The first start can take up to a minute while the model loads.
2. **Import & segment** — choose image files or a folder, or drag them into the window. Choose the **input size**
   (see below) and press **Start**.
3. **Inspect masks** — check the overlay for missed organoids, false positives at the well edge and merged neighbours.
4. **Controls (optional)** — assign groups and mark control images, or load a CSV plate map, then create a normalised table.
5. **Export** — all results are already saved in the run folder (default `%USERPROFILE%\VitaFormer_results\<date-time>_<name>`);
   **Save as ZIP...** additionally saves the whole run as one ZIP file.

A step-by-step guide with screenshots is in [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

### Choosing the input size

The input size changes the result. Choose it by inspecting the masks of a few annotated or clearly interpretable images,
**never** by looking at the functional readout you want to study. Three procedures are offered, each taken from an analysis
in the paper:

| Setting | Validated on | Value used in the paper |
|---|---|---|
| Native resolution | colorectal cancer organoids (CRC), DeepOrganoid | — |
| Scale factor, bilinear | GemcitabineScreen (PDAC organoids) | 0.25 |
| Square side S, box (square images only) | NViR liver spheroids | S = 256 |

For new data, no setting is validated in advance; there is no automatic size selection.

### Processing time

On a laptop CPU, one 2048 × 2044 px image at native resolution took about **3.5 minutes**, and the six 1216 × 1216 px example
images at S = 256 took about 30 seconds in total. Large batches at native resolution can therefore take hours.

## Outputs

Each analysis creates a new run folder; existing results and the original images are never modified.

| File | Content |
|---|---|
| `masks/` | Label masks, one PNG per image: 0 background, 1 Alive, 2 Dead, 3 Bubble. File names start with the input number. |
| `overlays/` | Colour overlays for inspection |
| `readouts_image_measurements.csv` | One row per image: foreground (Alive + Dead), Alive, Dead and Bubble areas in pixels, region statistics, mean intensity |
| `errors.csv` | Images that could not be processed, with the reason |
| `normalised_N/` | Readouts relative to the control median of each group (only if controls were defined) |
| `run_manifest.json` | Software version, weights SHA256, settings, software versions and SHA256 of every input |

Accepted inputs: 8-bit greyscale or RGB PNG, JPEG or TIFF, and single-channel 16-bit TIFF/PNG (one field of view per file).
Multi-page files, alpha channels and other formats are rejected with a reason rather than converted.

## Check your installation with the examples

`examples/` contains two small image sets with the results expected from this model. Run one of them in VitaFormer with the
input size given in [examples/README.md](examples/README.md), then drag the run folder onto `examples/check_examples.bat`.
The check compares your masks with the expected masks pixel by pixel and reports **PASS** or what differs.

## How to cite

If you use VitaFormer, please cite the software archive:

> Chen J. VitaFormer: bright-field organoid segmentation (Windows desktop application), version 1.1.0. Zenodo. https://doi.org/10.5281/zenodo.24554415

and the accompanying paper once it is published (the citation will be added here). A machine-readable citation is in
`CITATION.cff`.

## Licence

The VitaFormer code, documentation and model weights are released under the [MIT License](LICENSE). Third-party components are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). The example images keep the licences stated in their folders.

## Contact

Jia Chen. Questions and bug reports: <https://github.com/1chenjia-crypto/VitaFormer/issues>.
