# Attribution and scope

- Competition and images: NSF National Solar Observatory,
  [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
  Dataset access and competition terms apply separately from the implementation.
- U-Net: Ronneberger, Fischer and Brox (2015), https://arxiv.org/abs/1505.04597.
- ResNet: He, Zhang, Ren and Sun (2015), https://arxiv.org/abs/1512.03385.
- ConvNeXt: Liu et al. (2022), https://arxiv.org/abs/2201.03545.
- DINOv3: Siméoni et al. (2025), https://arxiv.org/abs/2508.10104.
  The fine-tuned ConvNeXt parameters derive from DINOv3 pretraining. Its license
  is included unchanged as `DINOv3-LICENSE.md`; this package does not relicense
  those underlying parameters under a different license.
- Segmentation Models PyTorch: Pavel Iakubovskii and contributors,
  https://github.com/qubvel-org/segmentation_models.pytorch (MIT).
- timm: Ross Wightman and contributors,
  https://github.com/huggingface/pytorch-image-models (Apache-2.0 code;
  individual pretrained model terms can differ).
- COCO RLE operations: pycocotools, https://github.com/ppwwyyxx/cocoapi.
- Local PQ implementation follows the competition organizer's public
  self-evaluation notebook, credited in `src/filaments/metric.py`.

The package contains the author's task-trained checkpoints and code, not claims
of authorship of the underlying neural architectures, datasets or dependencies.
