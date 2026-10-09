# Deep-Learning Architecture and Experimental Setup

## Task definition

The project performs multiclass semantic segmentation of remote-sensing RGB
imagery. The principal study is cross-domain generalization: models learn from
OpenEarthMap (OEM) or IRSAMap source imagery after label harmonization, then are
evaluated on the CEI target domain without CEI fine-tuning.

For the primary experiments:

```text
input  = RGB image tensor [B, 3, H, W]
output = CEI logits       [B, 7, H, W]
target = CEI class index  [B, H, W], with 255 ignored
```

The seven output classes, in channel order, are Rangeland, Agriculture, Tree,
Water, Building, Road, and Non-vegetated.

## Model factory

`src/models/model_factory.py` constructs models from the `model` section of a
YAML config. It currently supports U-Net, DeepLabV3, UPerNet, SegFormer,
UNetFormer, and FT-UNetFormer. DeepLabV3 is available in code but is not one of
the five active OEM-to-CEI comparison configs.

### Five-model OEM-to-CEI comparison

| ID | Architecture | Encoder | Decoder/output behavior | Train config |
| --- | --- | --- | --- | --- |
| M1 | FT-UNetFormer | fixed Swin-B | Global-Local Transformer decoder; single output | `ftunetformer_swinb_oem2cei.yml` |
| M2 | U-Net | EfficientNet-B4 | SMP U-Net decoder | `unet_effb4_oem2cei.yml` |
| M3 | UNetFormer | ResNet-101 via timm | GLTB decoder; main + auxiliary outputs in training | `unetformer_r101_oem2cei.yml` |
| M4 | UPerNet | Swin-B via timm/SMP | Pyramid pooling and feature-pyramid fusion | `upernet_swinb_oem2cei.yml` |
| M5 | SegFormer | MiT-B5 | Lightweight all-MLP decoder | `segformer_mitb5_oem2cei.yml` |

All five configs are under `configs/cei_oem/train/`. Their matching CEI
evaluation configs are under `configs/cei_oem/test/`.

## Architecture details

### U-Net with EfficientNet-B4

The U-Net is built by `segmentation-models-pytorch` (SMP). EfficientNet-B4
provides multi-scale encoder features, and the U-Net decoder progressively
upsamples and fuses skip connections to recover fine spatial boundaries. The
segmentation head produces seven logits per pixel.

The factory also supports an external OEM-SAR variant with SCSE decoder
attention and a nine-class head. `LeadingChannelDrop` removes its leading
background channel so its remaining eight outputs align with native OEM
classes. That baseline is separate from the seven-class M2 experiment.

### UNetFormer with ResNet-101

The local UNetFormer implementation uses a timm backbone with four feature
levels and a Global-Local Transformer Block decoder. Windowed self-attention
captures contextual relationships, while convolutional branches retain local
detail. Learned weighted fusion combines decoder and skip features, followed by
a feature-refinement head.

During training, UNetFormer returns a main full-resolution prediction and an
auxiliary prediction assembled from intermediate decoder features. During
evaluation, it returns only the main prediction.

### FT-UNetFormer with Swin-B

FT-UNetFormer is the fully-transformer variant. Its encoder is a local Swin-B
implementation with embedding dimension 128, stage depths `(2, 2, 18, 2)`, and
attention heads `(4, 8, 16, 32)`. Its decoder uses the Global-Local Transformer
family with 256 decode channels. It has no auxiliary output.

When `encoder_weights` is non-null, the factory maps compatible timm Swin-B
ImageNet tensors into this backbone. The model reports how many tensors were
mapped so a partial initialization is visible.

### UPerNet with Swin-B

SMP UPerNet uses the timm Swin-B encoder
`tu-swin_base_patch4_window7_224.ms_in22k_ft_in1k`. Its config sets
`encoder_params.img_size: 512`, matching the training crop. The CEI test config
uses 1024 because its full test tiles are 1024x1024. The checkpoint weights are
resolution-independent, but model construction still needs the intended input
size.

### SegFormer with MiT-B5

SMP SegFormer uses the MiT-B5 hierarchical transformer. MiT avoids fixed
absolute positional embeddings, so the 512-pixel training crop and 1024-pixel
CEI test resolution do not require an `img_size` override.

## Primary experimental protocol

The five OEM models share:

- OEM source root `data/OpenEarthMap/OpenEarthMap_wo_xBD`;
- split files `train_split_60.txt`, `val_split_10.txt`, and
  `test_split_30.txt`;
- `label_map: oem_to_cei`;
- 512x512 random training crops;
- seven output classes and `ignore_index: 255`;
- ImageNet normalization and encoder initialization;
- `augment: false`, so random crop is the only stochastic spatial transform;
- full-image validation;
- AdamW, cross-entropy, CUDA mixed precision when available;
- best-checkpoint selection by validation mIoU.

They differ in model capacity and the configured optimization budget:

| Model | Batch | Epochs | Learning rate | Weight decay | Scheduler |
| --- | ---: | ---: | ---: | ---: | --- |
| M1 FT-UNetFormer/Swin-B | 4 | 100 | 0.00006 | 0.01 | none |
| M2 U-Net/EfficientNet-B4 | 8 | 200 | 0.0001 | 0.000001 | none |
| M3 UNetFormer/ResNet-101 | 8 | 200 | 0.0001 | 0.000001 | none |
| M4 UPerNet/Swin-B | 4 | 200 | 0.00006 | 0.01 | none |
| M5 SegFormer/MiT-B5 | 4 | 100 | 0.00006 | 0.01 | none |

M3 additionally uses auxiliary-loss weight 0.4. Since epochs and optimizer
hyperparameters are not identical across all five models, results compare the
configured model recipes rather than architecture under a perfectly controlled
single optimization schedule.

## Source-domain comparison

`configs/cei_irsa/unet_effb4_irsa2cei.yml` trains the same seven-class
U-Net/EfficientNet-B4 family on IRSAMap instead of OEM. It uses batch size 8,
200 epochs, learning rate 0.0001, weight decay 0.000001, cross-entropy, no
scheduler, and the same 512 crop and ImageNet normalization.

Two test routes distinguish in-domain quality from transfer quality:

- `test_irsa2cei_on_irsa.yml`: evaluate on IRSAMap test data;
- `test_irsa2cei_on_cei.yml`: evaluate the same checkpoint on CEI.

The OEM-trained M2 and IRSA-trained U-Net can therefore be compared on the same
CEI taxonomy and CEI test split.

## CEI evaluation setup

The main CEI test configs use:

- `data/CEI_data/images` and `data/CEI_data/masks`;
- `test_split.txt`;
- `label_map: cei`;
- full 1024x1024 tiles, padded only when needed;
- one image per evaluation batch;
- cross-entropy as the reported test loss;
- optional four-way flip TTA from the command line.

The test config must reconstruct the architecture that produced the checkpoint.
`evaluate.py` checks model name, encoder name, and number of classes when the
checkpoint stores its training config, then performs strict state-dictionary
loading.

## Reproducibility and interpretation

Each training run copies its YAML into the experiment directory and each
checkpoint stores that resolved config. However, the code currently does not
set global random seeds or deterministic PyTorch settings. Hardware, library
versions, and random initialization/data order may therefore change results
across repeated runs.

For a defensible comparison, retain the copied config, checkpoint epoch, TTA
setting, dataset split files, dependency versions, hardware, and ideally the
mean and variation from multiple seeded trials.
