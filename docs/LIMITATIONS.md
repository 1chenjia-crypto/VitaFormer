# Known limitations (v1.1.0)

## Scientific scope
- **One frozen model.** VitaFormer, training seed 42 — the seed pre-specified as primary in every analysis plan of the paper,
  not chosen on functional results. The paper reports two further seeds (3407, 2026); they are not included.
- **Training data.** Bright-field images of colorectal cancer organoids from a single source dataset (481 training images). Performance on other
  organoid types, microscopes and magnifications depends on the imaging setup and, above all, on the **input size**.
- **Input size.** Only three resize procedures are offered, each taken from an analysis in the paper. There is no automatic
  size selection. In the paper, all three main models failed at native resolution on large liver spheroids, and post hoc
  input settings changed some model comparisons.
- **Dead class.** Dead is a model class. It was not validated against an external cell-death assay: it was absent or sparse
  on PDAC organoids stained with propidium iodide and on liver spheroids with ATP measurements. Do not report Dead area as
  cell death or viability.
- **No instance segmentation.** Touching organoids merge; the region count is not an organoid count.
- **Pixels only.** No physical calibration (µm) is applied.
- **No assay prediction.** VitaFormer does not predict CellTiter-Glo, ATP or any other assay value. Calibrated predictions need
  your own assay data and are outside this application.
- **Accepted inputs** are limited to the formats used in the validated analyses (8-bit greyscale or RGB; single-channel 16-bit).

## Software
- **CPU only.** This build contains the CPU version of PyTorch. At native resolution a 2048 × 2044 px image takes about
  3.5 minutes on a laptop CPU.
- **CPU and GPU differ slightly.** The paper's analyses ran on a GPU with fp16; this application runs on the CPU with fp32.
  Masks can differ in a few boundary pixels (measured: 11 of 4,186,112 pixels on a CRC test image; ≤ 416 of 1,478,656 pixels
  per image on the NViR examples).
- **Windows only.** This installer has been tested on Windows 11 (64-bit).
- **Not code-signed.** Windows may show a SmartScreen warning at installation.
- Progress is reported per image, not within an image.
