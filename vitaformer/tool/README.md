# VitaFormer v1.1.0 – local tool for bright-field organoid segmentation

The tool segments bright-field organoid images with the frozen **VitaFormer** model. It exports label masks, overlays and
per-image measurements, and optionally values relative to controls. Everything runs on your own computer through a page in your
web browser; images are not uploaded.

- **New users:** read `docs/QUICKSTART.md` (English / 中文). Windows: run `install_windows.bat` once, then `start_VitaFormer.bat`.
- **Limitations and unverified items:** `docs/LIMITATIONS.md`.
- **What was checked and how:** `docs/VALIDATION_RECORD.md`.
- **Trial with a new user:** `docs/USER_TRIAL_CHECKLIST.md`.

## Model
- `weights/vitaformer_m2_seed42_final.pth`, SHA256 `0f12be44859d06723ebfa35e8416df8cd4b02ff83f3ac153654d0f68df226101`, 361 MB.
- Seed 42 is the default because it was pre-specified as the primary seed in every analysis plan. It was not chosen from
  functional results.
- The checksum is verified on every start; other files are refused.
- Inference is the frozen procedure: 224 px window, stride 112, reflect padding, ImageNet normalisation, softmax average and
  argmax. There is no threshold, test-time augmentation or post-processing. GPU runs use fp16 and CPU runs fp32.

## Input size – only validated procedures
| Setting | Procedure | Validated on |
|---|---|---|
| Native | no resizing | CRC (training data), DeepOrganoid |
| Scale factor, bilinear | resize by factor f (PIL bilinear); mask returned with nearest neighbour | GemcitabineScreen (f = 0.25) |
| Square side S, box | resize to S × S (PIL box); mask returned with nearest neighbour; square images only | NViR (S = 256) |

## Outputs (a new folder per run; nothing is overwritten)
| File | Content | Tier |
|---|---|---|
| `masks/NNNN_<name>.png` | labels: 0 background, 1 Alive, 2 Dead (model class, not externally validated), 3 Bubble | – |
| `overlays/NNNN_<name>.jpg` | preview | – |
| `readouts_image_measurements.csv` | areas (px), foreground fraction, region count (connected regions, not organoids), region size/diameter/circularity, intensity | 1 image measurement |
| `normalised_N/readouts_relative_to_controls.csv` | value ÷ median of the controls in the same group; explicit status for missing controls, zero control median, images not in the plate map | 2 relative to controls |
| `errors.csv` | rejected/failed inputs and why | – |
| `run_manifest.json` | tool version, weights SHA256, device (and GPU→CPU fallback reason), settings, software versions, input SHA256 | – |

Tier 3 (assay-calibrated predictions, for example of CellTiter-Glo) is **not** produced. It needs your own assay data.

## Command line (advanced)
```
python -m vitaformer predict --input <folder> --output <folder> [--resize square_box --resize-value 256] [--device cpu]
python -m vitaformer normalize --run <run folder> --plate-map plate_map.csv      # columns: image,group,role
python -m vitaformer app                                                           # the user interface
```
The CLI and the user interface call the same code (`vitaformer/pipeline.py`, `io.py`, `infer.py`, `readouts.py`). For
research-grade evaluation against functional assays, use the separate `oreval` workflow described in the paper.
