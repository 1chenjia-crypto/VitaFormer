# Validation report — VitaFormer desktop application v1.1.0

Test computer: Windows 11 (64-bit, build 26200), laptop CPU, no GPU used.

## 1. Model and code identity

| Check | Result |
|---|---|
| Weights file `vitaformer_m2_seed42_final.pth` (361,037,930 bytes) | SHA256 `0f12be44859d06723ebfa35e8416df8cd4b02ff83f3ac153654d0f68df226101`, verified by the application at start-up and recorded in every `run_manifest.json`; identical to the frozen VitaFormer seed-42 checkpoint of the paper |
| Model, inference and readout code (`model_v2.py`, `model_v3_multiscale.py`, `infer.py`, `pipeline.py`, `readouts.py`, `io.py`) | identical to the code verified against the paper's inference code (bit-identical masks on a GPU, `INFERENCE_SPEC.md`) |
| Runtime | Python 3.12.0, PyTorch 2.6.0+cpu, timm 1.0.27, NumPy 1.26.4, SciPy 1.11.4, pandas 2.2.3, Pillow 10.4.0 (`provenance/desktop_runtime_lock.txt`) |

## 2. Agreement with the paper's masks

The desktop application was run on a CRC test image at native resolution and the mask was compared with the VitaFormer
seed-42 mask used in the paper (computed on a GPU with fp16 autocast).

| Image | Pixels | Differing pixels | Agreement | Alive px (app / paper) | Dead px (app / paper) | Time |
|---|---|---|---|---|---|---|
| CRC_D12_J4M-CPT-11_WellE7_…_Seq0003 | 4,186,112 | 11 | 0.999997 | 1,630 / 1,632 | 30,434 / 30,429 | 208.5 s |

The input file was identical to the paper's image (SHA256 `9d657386…4897fb28`). All 11 differing pixels lie on class
boundaries, as expected from fp32 (CPU) versus fp16 (GPU) arithmetic.

## 3. Further checks of the same model and code

- NViR example (6 images, square side 256, CPU): masks identical to the expected output (agreement 1.0 for every image).
- CPU (fp32) versus GPU (fp16) on the NViR images: 0 to 416 differing pixels per 1,478,656-pixel image (agreement ≥ 0.99972).
- Input checks, batch processing, cancellation, duplicate file names, 16-bit input, paths with spaces and non-Latin characters,
  control normalisation and its error cases: 25/25 core tests and 19/19 interface tests passed.
- The command-line interface and the user interface produce identical masks and readouts.

These measurements were made with the release-candidate build, which differs from v1.1.0 only in the interface language and
labels; the model, inference and readout code and the weights are the same. The final installer was checked with the examples
(section 4).

## 4. Release check of the final installer

Installer `installer/VitaFormer_Setup_1.1.0.exe`, SHA256 `14aaf8a807aed7e8f64abdee22f83eabaeb2a15b69e929cea8a1d7d4f0606b19`
(`installer/SHA256SUMS.txt`). Installed for the current user on the test computer above; both example sets were run in the
installed application on 2026-11-24 and checked with `examples/check_examples.py`.

| Check | Result |
|---|---|
| `examples/nvir` (square side 256) → `check_examples.bat` | **PASS (identical)**: 6/6 masks identical to the expected output (agreement 1.000000 for every image); weights and input size confirmed; 10.0 s for 6 images |
| `examples/crc` (native resolution) → `check_examples.bat` | **PASS**: lowest per-image agreement 0.999997 (Seq0003: 11 differing boundary pixels, Alive 1,630 / 1,632, Dead 30,434 / 30,429; Seq0012: identical); weights and input size confirmed; 222 s for 2 images |

Running the NViR example at native resolution instead of square side 256 gives visibly fragmented masks, as expected from the
paper (large single spheroids fail at native resolution); `check_examples.bat` reports the wrong input size in that case.

## 5. Not tested

- The scale-factor (0.25) procedure in the desktop build.
- Installation on Windows 10 and on a computer without development tools.
- Use by people outside the development team.

## How to reproduce

Run the examples in `examples/` with the input size given in `examples/README.md` and drag the run folder onto
`examples/check_examples.bat`.
