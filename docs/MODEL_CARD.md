# Model card — VitaFormer (seed 42)

## Model
- **Architecture.** ViT-B/16 encoder (timm `vit_base_patch16_224`, initialised from BiomedCLIP for training) with features
  taken after blocks 4, 8 and 12 and projected by 1 × 1 convolutions; a convolutional detail branch at 112, 56 and 28 px;
  and a four-stage multi-scale decoder. 90.2 M parameters in total (61.3 M trainable during training; patch embedding and
  blocks 1–4 frozen).
- **Output.** Four classes per pixel: background, Alive, Dead, Bubble.
- **Inference.** 224 px sliding window with stride 112, reflect padding, ImageNet normalisation, softmax averaging of
  overlapping windows, argmax. No threshold, test-time augmentation or post-processing. Details: `INFERENCE_SPEC.md`.
- **Training.** Colorectal cancer organoid bright-field images (481 training, 164 validation, 167 test images) with a
  training recipe that up-weights the rare Dead class. Three training seeds were trained; this release contains seed 42,
  the seed pre-specified as primary.

## Performance reported in the paper (mean of three seeds unless stated)
- CRC test set: mIoU 83.3 % (nnU-Net 85.0 %, SegFormer-B2 82.9 %). Dead recall 77.8 %, Dead precision 58.0 %.
- External functional datasets (frozen weights, no assay labels used):
  - NViR liver spheroids: Spearman ρ between control-normalised foreground area and ATP, 0.87 (square side 256).
  - GemcitabineScreen PDAC organoids: ρ between area fold change and propidium iodide at 52 h, −0.85 (factor 0.25).
  - DeepOrganoid CRC lines: mean within-drug Spearman ρ between area- and CellTiter-Glo-based line rankings, 0.72.

## Intended use
Research use: segmentation of bright-field organoid images and image-derived readouts (area, region statistics), used
alongside or in place of an endpoint assay after the user has checked masks and input size on their own images.

## Out of scope
- Clinical or diagnostic use.
- Reporting Dead area as cell death or viability (not externally validated).
- Counting organoids (no instance segmentation).
- Predicting assay values.

## Caveats
- Performance depends strongly on the input size; select it on annotated or clearly interpretable images, not on the
  readout of interest.
- Results from a different training seed can differ; the paper reports seed ranges.

Development files refer to this model by its internal identifier M2.
