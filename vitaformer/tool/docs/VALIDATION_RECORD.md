# Validation record – VitaFormer tool v1.1.0 (2026-11-10)

All checks ran on the development server (Linux, CPU, 8 threads). The GPUs were occupied by other users throughout, so the GPU path
was not exercised. "Identical" means bit-identical masks and readouts.

## A. Scientific consistency of the shared core
| # | Check | Result |
|---|---|---|
| A1 | v1.1.0 CLI vs v1.0.0 reference output, 3 CRC test images, native | **identical** masks; readouts differ by 0.0 |
| A2 | v1.0.0 tool (CPU fp32) vs frozen paper pipeline (Stage F, GPU fp16), same 3 images (recorded 11-10 in `examples/RUN_RECORD.md`) | 1 / 11 / 2 differing pixels of 4.19 M per image (agreement ≥ 0.999997) |
| A3 | "Square side 256, box" (CPU) vs the paper's NViR masks `m2_s42_S256` (GPU fp16), 6 example images | 2 images identical; 20 / 40 / 180 / 416 differing pixels of 1.48 M on the others (agreement ≥ 0.99972); foreground areas equal or within 145 px |
| A4 | Resize procedures compared with the analysis code | native = DeepOrganoid/CRC `infer`; factor + BILINEAR + NEAREST back = `gem_pi_infer_20261109.predict`; square + BOX + NEAREST back = `nvir_infer_20261109` (gray → RGB order is equivalent because the channels are identical) |
| A5 | Readout definitions | `image_readouts` unchanged from v1.0.0 (A1 shows identical values) |

## B. Batch behaviour and edge cases (`tests/test_tool.py`, 25/25 PASS)
Paths with Chinese characters and spaces were used for both inputs and outputs. The following cases were checked:
- **Accepted inputs:** 8-bit L, 8-bit RGB JPEG, and 16-bit TIFF (stretch recorded in the output).
- **Rejected inputs, each with its reason:** alpha channel, palette, tiny image, corrupt file, `.txt`, and a 2-page TIFF.
- **Batch:** continues after failures (6 processed, 6 rejected); a blank image gives 0 foreground.
- **Traceability:** duplicate file names get distinct outputs; the manifest has the weights SHA256, software versions and input
  checksums; originals are unchanged.
- **No overwriting:** a second run with the same name goes to a new folder.
- **Square mode:** rejects non-square images explicitly.
- **Cancel:** stops after the current image and keeps the finished ones (3 of 30 kept).
- **Control normalisation:** normal group = ok; a zero control median and a group without controls are both flagged with no value;
  an image missing from the plate map is flagged; an invalid role is refused.

## C. User interface through its HTTP API (`tests/test_app_api.py`, 19/19 PASS)
The results folder was under a Chinese/space path, with 6 NViR images uploaded plus 1 invalid file (Chinese name). The following
were checked:
- **Upload:** the invalid file was reported and not converted.
- **Run:** square 256 finished with 6 ok and 1 failed. The GPU was busy, so the run used the "auto" device's CPU fallback, and the
  reason is recorded.
- **Agreement with the CLI:** the interface's masks are identical to the CLI masks. Its exported readouts are identical to the
  CLI's; the on-screen JSON is rounded to 10 significant digits.
- **Controls:** the group preview showed 3 controls with status ok; a group without controls was flagged before normalising.
  The normalised values are identical to the CLI, and a second normalisation went to a new folder.
- **Export:** the ZIP contains the masks, overlays, readouts, errors, manifest and normalised folder.
- **Second launch:** detects the running instance and opens it instead of starting another.

## D. Bugs found by these checks and fixed
1. **Device:** "auto" chose a busy GPU and failed to start; it now falls back to the CPU and records why.
2. **Browser compatibility:** the page used JavaScript optional chaining, which fails in Chrome 78; it was replaced with syntax that
   older browsers support. Margins were added as a fallback for flex `gap`.
3. **Inspection view:** after reopening a run, the image was drawn while its panel was hidden (zoom 0); the view now refits when
   shown.
4. **Reopened runs:** the settings shown did not reflect the run's real input size; fixed.
5. **Elapsed time:** kept counting after the run finished; fixed.

## E. Clean installation from the distributed ZIP (Linux; 2026-11-10 16:46–17:11)
Procedure: unzip `VitaFormer_v1.1.0.zip` into a new folder whose path contains Chinese characters and spaces; check
`SHA256SUMS.txt` (all OK); run `install.sh`.

| Step | Result |
|---|---|
| Install | **Success in 24 min.** Most of the time was the network (~100–300 KB/s); one PyTorch download was interrupted and resumed automatically by pip. Packages: Python 3.11.5, torch 2.6.0+cpu (as pinned). |
| CLI on the NViR example in the clean environment vs the shipped expected output | **PASS (identical)**: all 6 masks agreement 1.000000; readouts differ by 0 |
| `tests/test_tool.py` in the clean environment | **25 / 25 PASS** |
| `start.sh` + `tests/test_app_api.py` in the clean environment | **19 / 19 PASS** (CPU-only build: device cpu, 6 images in 10 s; second launch detected) |

**Not tested:**
- `install_windows.bat` / `start_VitaFormer.bat` on Windows. They must be run on a Windows 10/11 computer before release.
- GPU builds (`requirements_gpu_cu124.txt`, `install … gpu`).
- macOS.
- Use by a person outside the development team; see USER_TRIAL_CHECKLIST.md.

**Note for users:** on a slow network the first installation can take more than 30 minutes. Re-running the installer skips
finished steps.

## F. Screenshots (`docs/screenshots/`)
Real runs, rendered with headless Chrome 78 at 1480 × 940:
- import, inspect (overlay, side by side) and export pages (English);
- inspect and controls pages (中文).
