# Changelog

## 1.1.0 — 2026-11-24 (first public release)
- Windows desktop application with an installer (`VitaFormer_Setup_1.1.0.exe`): own application window (Microsoft Edge
  WebView2), English interface, Start-menu entry, optional desktop shortcut and uninstaller. No Python installation, browser
  or internet connection is needed.
- Bundled Python 3.12 runtime with the CPU build of PyTorch 2.6.0 and the frozen VitaFormer seed-42 weights
  (SHA256 verified at every start and recorded in every run).
- Four-step workflow: import and segment, inspect masks, controls (optional), export; results saved in a new run folder per
  analysis, with a run manifest.
- Only the three resize procedures validated in the paper are offered.
- Example sets (CRC, NViR) with expected outputs and a one-click check (`examples/check_examples.bat`).

Earlier development builds were internal and were not distributed.
