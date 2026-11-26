# Known limitations and open items (v1.1.0, 2026-11-10)

## Scientific scope
- One frozen model: VitaFormer, seed 42.
  - This is the seed pre-specified as primary in every analysis plan; it was not chosen on functional results.
  - Seeds 3407 and 2026 are recognised by checksum but are not shipped.
- Trained on colorectal-cancer organoid images. Performance on other systems depends on the imaging setup and **input size**.
  Only three resize procedures are offered, each copied from a validated analysis. No automatic size selection exists.
- Dead is a model class; it was **not** validated externally (it did not activate on PDAC/PI or liver-spheroid/ATP data).
- No instance segmentation: touching organoids merge, so "region count" is not an organoid count.
- No physical units (µm) – all sizes are in pixels.
- No CellTiter-Glo / ATP prediction. Calibrated predictions need the user's own assay data and are outside this tool.
- Accepted inputs are limited to what the validated analyses used: 8-bit greyscale, 8-bit RGB, and single-channel 16-bit images
  (with a per-image 1–99th percentile stretch). Multi-page TIFF, alpha channels, colour 16-bit and other formats are rejected.

## Software
- CPU and GPU give masks that can differ in a few boundary pixels:
  - GPU runs use fp16, CPU runs fp32;
  - measured on the NViR example: 0 to 416 differing pixels per 1.48-million-pixel image (agreement ≥ 0.99972).
- "Auto" device uses the GPU when it works and otherwise falls back to the CPU; the reason is recorded in `run_manifest.json`.
- A run can be reopened in the page only while the tool is still running. After a restart, the results remain in the run folder
  but cannot be reopened in the page.
- The page is designed for current Chrome, Edge and Firefox. It was rendered with Chrome 78 for the screenshots; older browsers may
  space elements slightly differently.

## Not yet verified (must not be claimed)
- **Windows installer and launcher (`install_windows.bat`, `start_VitaFormer.bat`) have not been run on Windows.** They were written
  for Windows 10/11 64-bit with internet access. The equivalent Linux flow (`install.sh`, `start.sh`) was verified in a clean
  folder (see VALIDATION_RECORD.md).
- **GPU path in a fresh install:** not verified. The GPUs on the development server were occupied by other users during testing.
  All paper analyses used the GPU path.
- **macOS:** not tested.
- **Use by people outside the development team:** not done. See USER_TRIAL_CHECKLIST.md; this trial must be organised by the
  authors and cannot be replaced by automated tests.

## Licences to confirm before public release
- BiomedCLIP (encoder initialisation) and the licence under which the trained weights can be distributed.
- The NViR example images are MIT-licensed (Zenodo 15374443); the notice must ship with them.
- The CRC reference images used in development (`examples/`) have an unconfirmed licence and are **not** included in the package.
