# Tool verification record (2026-11-10)

1. **Reference output** (`expected_output/`): venv_organoid (frozen environment, torch 2.6.0+cu124), `--device cpu`, M2 seed 42
   final.pth (SHA256 0f12be44…f226101); 3 CRC test images (2048×2044); 124 s.
2. **Equivalence to the frozen pipeline**: tool masks (CPU, fp32) vs Stage F M2 seed-42 test masks (GPU, fp16, infer_hard):
   differing pixels 1 / 11 / 2 of 4.19 M per image (agreement ≥ 0.999997). This confirms that building the encoder without the
   BiomedCLIP file and loading the checkpoint strictly reproduces the frozen model; the residual pixels come from fp16 vs fp32.
3. **Fresh environment**: new conda env (Python 3.11.5) + `requirements.txt` (torch/torchvision CPU builds); the tool folder was copied
   to a scratch location and run there.
   - First attempt failed: `RuntimeError: operator torchvision::nms does not exist` (timm pulled a CUDA torchvision build from PyPI
     while torch was the CPU build). Fix: torchvision==0.21.0 pinned and installed from the same index as torch. No code change.
   - Second attempt: 3 images in 118 s (127.8 s wall, max RSS 4.0 GB, 8 threads). `check_example.py`: masks 100% identical,
     every readout identical → **PASS**. Package list: `fresh_env_pip_freeze_20261110.txt`.
4. `vitaformer normalize` tested on the format demo plate map (writes `*_rel_control` columns).
5. GUI not built (out of scope). GPU path not tested in the fresh env (both GPUs occupied by other users during the test);
   the GPU path is the one used in all paper analyses.
