#!/usr/bin/env bash
# Rebuild the OrganoidViT reproduction env (the original /tmp/venv-OrganoidSegbaseline was wiped).
set -euxo pipefail
V=${WORKSPACE:-.}/OrganoidSegBaseline/venv-orgvit-py36
[ -x $V/bin/python ] || ${PYTHON36:-python3.6} -m venv $V
#$V/bin/pip install -q --upgrade "pip<22"
$V/bin/python -c "import torch" 2>/dev/null || $V/bin/pip install --retries 20 --timeout 120 torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
$V/bin/pip install --retries 20 --timeout 120 opencv-python==4.5.1.48 numpy==1.19.5
$V/bin/pip install --retries 20 --timeout 120 mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
$V/bin/pip install --retries 20 --timeout 120 timm==0.3.2 opencv-python==4.5.1.48 numpy==1.19.5 Pillow==8.4.0 PyYAML==5.4.1 tqdm==4.64.1 matplotlib==3.3.4 scipy terminaltables prettytable tensorboard
$V/bin/pip freeze > ${WORKSPACE:-.}/OrganoidSegBaseline/orgvit_3seed_20261111/env_freeze.txt
cd ${WORKSPACE:-.}/OrganoidSegBaseline/Orgvit
PYTHONPATH=$PWD $V/bin/python -c "import torch,mmcv,mmseg,timm;print('ENV_OK',torch.__version__,torch.version.cuda,mmcv.__version__,mmseg.__file__,timm.__version__,torch.cuda.is_available())"
