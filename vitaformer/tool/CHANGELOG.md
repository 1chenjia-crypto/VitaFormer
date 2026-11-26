# Changelog

## 1.1.0 (2026-11-10)
- New local user interface (`python -m vitaformer app`, `start_VitaFormer.bat`, `start.sh`) with four steps: import and segment,
  inspect, controls, export. It is bilingual (English/中文), uses only the Python standard library for the server, and binds to
  127.0.0.1 only.
- One shared core for the CLI and the interface (`pipeline.py`):
  - per-image error capture, so a single failure does not stop the batch;
  - cancellation that keeps finished results;
  - a new run folder each time, so nothing is overwritten;
  - traceable `NNNN_` file names;
  - a run manifest with input SHA256s.
- **Input validation** (`io.py`). Only the input types used in the validated analyses are accepted. Multi-page files, alpha
  channels, palette, float and colour 16-bit images are rejected with a reason.
- **Resize procedures corrected.** v1.0.0 offered a scale factor with BOX down-sizing, a combination that was not used in any
  validated analysis. v1.1.0 offers only the three validated procedures: native, factor + bilinear (GemcitabineScreen), and
  square S + box (NViR). Native-resolution results are unchanged; the v1.0.0 reference output is reproduced exactly.
- **Control normalisation.** Each group's controls are shown before normalising. Missing controls, a zero control median, images
  not in the plate map and invalid roles are flagged explicitly; no default values are inserted.
- The "auto" device falls back to the CPU when the GPU cannot be used, and records why.
- New example: NViR images (MIT licence) with a plate map and expected output. The CRC development images are not distributed.

## 1.0.0 (2026-11-10)
- Command-line tool with the frozen VitaFormer weights, readouts and control normalisation; verified in a fresh environment (CPU).
