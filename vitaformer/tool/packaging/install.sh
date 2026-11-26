#!/usr/bin/env bash
# VitaFormer first-time setup for Linux/macOS: private micromamba + Python 3.11.5 + pinned packages inside ./runtime
set -euo pipefail
cd "$(dirname "$0")"
RT="$PWD/runtime"; REQ=requirements_cpu.txt; [ "${1:-}" = gpu ] && REQ=requirements_gpu_cu124.txt
OS=linux-64; [ "$(uname)" = Darwin ] && OS=osx-64 && [ "$(uname -m)" = arm64 ] && OS=osx-arm64
mkdir -p "$RT"
if [ ! -x "$RT/bin/micromamba" ]; then
  echo "[1/3] downloading micromamba"; curl -L --fail "https://micro.mamba.pm/api/micromamba/$OS/latest" | tar -xj -C "$RT" bin/micromamba
fi
if [ ! -x "$RT/env/bin/python" ]; then
  echo "[2/3] creating Python 3.11.5"; "$RT/bin/micromamba" create -y -q -r "$RT/mamba" -p "$RT/env" -c conda-forge python=3.11.5
fi
echo "[3/3] installing pinned packages ($REQ)"
if [ "$OS" != linux-64 ]; then sed 's/+cpu//' "$REQ" > "$RT/req.txt"; REQ="$RT/req.txt"; fi
"$RT/env/bin/python" -m pip install -q -r "$REQ"
"$RT/env/bin/python" -c "import torch, timm, scipy, pandas, PIL; print('packages OK, torch', torch.__version__)"
echo "Setup finished. Start with ./start.sh"
