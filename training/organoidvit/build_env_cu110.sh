#!/usr/bin/env bash
# Rebuild the OrganoidViT reproduction env (the original /tmp/venv-OrganoidSegbaseline was wiped).
set -euxo pipefail
V=${WORKSPACE:-.}/OrganoidSegBaseline/venv-orgvit-py36-cu110
[ -x $V/bin/python ] || ${PYTHON36:-python3.6} -m venv $V
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY $V/bin/pip install -q --upgrade "pip<22"
W=${WORKSPACE:-.}/OrganoidSegBaseline/orgvit_3seed_20261111/wheels_cu110
for u in https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp36-cp36m-linux_x86_64.whl https://download.pytorch.org/whl/cu110/torchvision-0.8.2%2Bcu110-cp36-cp36m-linux_x86_64.whl https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/mmcv_full-1.2.7-cp36-cp36m-manylinux1_x86_64.whl; do
  f=$W/$(basename "$u" | sed 's/%2B/+/'); until curl -sSL --retry 20 -C - -o "$f" "$u"; do sleep 5; done
done
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= all_proxy= ALL_PROXY=
$V/bin/python -c "import torch" 2>/dev/null || $V/bin/pip install --retries 20 --timeout 120 $W/torch-1.7.1+cu110-cp36-cp36m-linux_x86_64.whl $W/torchvision-0.8.2+cu110-cp36-cp36m-linux_x86_64.whl
$V/bin/pip install --retries 20 --timeout 120 opencv-python==4.5.1.48 numpy==1.19.5
$V/bin/pip install --retries 20 --timeout 120 $W/mmcv_full-1.2.7-cp36-cp36m-manylinux1_x86_64.whl
$V/bin/pip install --retries 20 --timeout 120 timm==0.3.2 opencv-python==4.5.1.48 numpy==1.19.5 Pillow==8.4.0 PyYAML==5.4.1 tqdm==4.64.1 matplotlib==3.3.4 scipy terminaltables prettytable tensorboard ipython==7.16.3 attrs
$V/bin/pip freeze > ${WORKSPACE:-.}/OrganoidSegBaseline/orgvit_3seed_20261111/env_freeze_cu110.txt
cd ${WORKSPACE:-.}/OrganoidSegBaseline/Orgvit
PYTHONPATH=$PWD $V/bin/python -c "import torch,mmcv,mmseg,timm;print('ENV_OK',torch.__version__,torch.version.cuda,mmcv.__version__,mmseg.__file__,timm.__version__,torch.cuda.is_available())"
