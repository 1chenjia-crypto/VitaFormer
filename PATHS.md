# Path conventions

All absolute server paths were removed. Scripts now use paths relative to a workspace root:

| In the scripts | Meaning |
|---|---|
| `3D_OrganoidSeg/...` | main project directory (data, runs, analyses) |
| `OrganoidSegBaseline/...` | baseline (U-Net, U-Net++, DeepLabv3, OrganoidViT) directory |
| `pretrained/biomedclip/open_clip_pytorch_model.bin` | BiomedCLIP weights (training initialisation only; not needed for inference) |
| `pretrained/torch_hub/checkpoints/vit_b_16-c867db91.pth` | torchvision ViT-B/16 weights (ImageNet ablation only) |
| `$WORKSPACE`, `$PYTHON`, `$PYTHON36`, `$EXPECTED_HOST` (shell) | workspace root, Python interpreter, Python 3.6 for the OrganoidViT environment, expected GPU host name |

The folder layout of this repository differs from the original workspace. Scripts that import sibling modules through
`sys.path` (for example `3D_OrganoidSeg/scripts`) expect the original layout; the original location of every file is listed in
`MANIFEST.csv` of the release package.
