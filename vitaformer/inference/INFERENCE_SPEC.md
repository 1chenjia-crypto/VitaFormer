# VitaFormer inference specification (frozen; matches the paper pixel for pixel)

Source of truth: `paper_pipeline/a0_m0_val_inference.py::infer_hard` (called by `stageF_test_inference.py` for the CRC test
masks of the paper). The tool copy is `C_readouts_and_tool/vitaformer_tool_v1.1.0/vitaformer/infer.py::predict_probs/predict_mask`.
Verified 2026-11-24: tool core vs paper masks = **bit-identical on GPU** for 5 CRC + 1 NViR images × 3 seeds
(`F_gold_standard/results_gpu/agreement.csv`); CPU see `F_gold_standard/results_cpu/`.

## 1. Model construction and loading
```python
from model_v3_multiscale import OrganoidSegMultiScale          # needs model_v2.py next to it (imported by name)
model = OrganoidSegMultiScale('random', num_classes=4, freeze_blocks=0, zero_cnn_input=False)
ck = torch.load(path, map_location='cpu', weights_only=True)
model.load_state_dict(ck['model_state'], strict=True)          # checkpoint keys: 'step' (=40000), 'model_state'
model.eval()
```
- `'random'` = build timm `vit_base_patch16_224` without pretrained weights; **every** weight comes from the VitaFormer checkpoint, so the
  BiomedCLIP file is NOT needed for inference and nothing is downloaded (timm `pretrained=False`). BiomedCLIP is only needed to re-train.
- `freeze_blocks` only affects `requires_grad` (training); irrelevant for inference.
- Total parameters 90,234,572 (61.29 M trainable during training, because patch-embed + ViT blocks 1–4 were frozen).
- Output of `model(x)`: logits `(B, 4, 224, 224)` (the code also accepts a tuple and takes element 0).

## 2. Image reading / preprocessing (tool `vitaformer/io.py`)
| Input | Handling |
|---|---|
| 8-bit RGB (jpg/png/tif) | used as is |
| 8-bit grey (mode L) | replicated to 3 channels |
| 16-bit single channel (mode I;16 / I) | per-image 1st–99th percentile stretch to 0–255 uint8, then 3 channels (same as DeepOrganoid p1–p99 analysis) |
| alpha, palette, float, colour 16-bit, multi-page TIFF | **rejected with a reason** (not validated) |

Paper CRC: `Image.open(jpg).convert('RGB')`. Paper NViR: 8-bit grey crops (`images_labeled_cropped_0.2`), grey → resize → 3 channels
(tool does 3 channels → resize; equivalent because channels are identical; verified identical).

## 3. Normalisation
`x/255`, then ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
(NOT the BiomedCLIP/OpenAI mean/std in `open_clip_config.json` — training used ImageNet values.)

## 4. Input scale (user-adjustable; the paper shows it changes results)
| Mode | Procedure | Validated on | Paper value |
|---|---|---|---|
| `native` | no resize | CRC, DeepOrganoid | – |
| `factor_bilinear` | resize by f with PIL BILINEAR; hard mask back to input size with NEAREST | GemcitabineScreen | f = 0.25 |
| `square_box` | resize to S×S with PIL BOX; mask back with NEAREST; square input only; down-sizing only | NViR | S = 256 |

Rules: choose the scale **without** looking at functional readouts (NViR: chosen by Dice vs reference masks on train/val,
`paper_pipeline/scale_selection_*`); no automatic scale rule exists (AutoScale v1 was a NO-GO).

## 5. Sliding window
- window 224, stride 112; image reflect-padded at bottom/right only up to ≥224 (`np.pad(..., mode='reflect')`);
  window grid = range(0, H−224+1, 112) plus a last window flush with the edge.
- per window: `softmax(logits.float())`; probabilities are **summed and divided by the overlap count** (softmax averaging,
  not logit averaging), then `argmax` over the 4 classes.
- GPU: `torch.amp.autocast('cuda')` (fp16) for the forward pass; CPU: fp32. This is the only source of CPU/GPU differences
  (a few boundary pixels; agreement ≥ 0.9997 measured in v1.1.0 validation).
- Batch size does not change results (paper 128, tool 64).
- No threshold, no TTA, no post-processing on the mask; no seed ensemble (paper reports 3 seeds separately, default seed 42).

## 6. Readouts (computed on the hard mask)
- foreground = labels {1, 2} (Alive + Dead); Bubble is not foreground.
- connected regions: 8-connectivity (`scipy.ndimage.label`, 3×3 structure); regions < 50 px dropped for region statistics.
- see `oreval/core.py::readouts_from_mask` and tool `readouts.py::image_readouts` (same definitions; B8 check: identical values).

## Corrections to the software plan (2026-11-24 check)
1. `gen_vitaformer_area_errors.py` is the **old Full-model script** (`expFinal_fromscratch/best_miou.pth`, OrganoidSegV2) — it is
   not the VitaFormer path. The paper's CRC VitaFormer masks come from `stageF_test_inference.py → a0_m0_val_inference.infer_hard`. Kept only as
   `paper_pipeline/gen_vitaformer_area_errors__OLD_FullModel_NOT_M2.py` for reference.
2. Overlap fusion is softmax-probability averaging, not logit averaging.
3. CNN detail branch channels are 32/64/128 (112/56/28 px); the ViT taps l4/l8/l12 are projected with 1×1 conv to 128/256/256.
4. Inference does not need `open_clip_torch`; it uses timm's ViT definition.
