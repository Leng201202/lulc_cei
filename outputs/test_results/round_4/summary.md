# Test results -- round_4

## Per-run results (CEI test set)

| Key | Model | Source | OA | mIoU | mF1 |
| --- | --- | --- | --- | --- | --- |
| m1 | FT-UNetFormer Swin-B | oem | 0.7616 | 0.6005 | 0.7324 |
| m2 | U-Net EfficientNet-B4 | oem | 0.7591 | 0.5870 | 0.7246 |
| m3 | UNetFormer ResNet-101 | oem | 0.7581 | 0.5976 | 0.7291 |
| m4 | UPerNet Swin-B | oem | 0.7689 | 0.6222 | 0.7491 |
| m5 | SegFormer MiT-B5 | oem | 0.7623 | 0.5794 | 0.7193 |
| m1 | FT-UNetFormer Swin-B | irsa | 0.6637 | 0.5280 | 0.6648 |
| m2 | U-Net EfficientNet-B4 | irsa | 0.7311 | 0.5986 | 0.7260 |
| m3 | UNetFormer ResNet-101 | irsa | 0.6575 | 0.5469 | 0.6783 |
| m4 | UPerNet Swin-B | irsa | 0.6828 | 0.5296 | 0.6695 |
| m5 | SegFormer MiT-B5 | irsa | 0.6885 | 0.5520 | 0.6773 |

## Generalization comparison (mIoU on CEI)

| Key | Model | OEM->CEI | IRSA->CEI |
| --- | --- | --- | --- |
| m1 | FT-UNetFormer Swin-B | 0.6005 | 0.5280 |
| m2 | U-Net EfficientNet-B4 | 0.5870 | 0.5986 |
| m3 | UNetFormer ResNet-101 | 0.5976 | 0.5469 |
| m4 | UPerNet Swin-B | 0.6222 | 0.5296 |
| m5 | SegFormer MiT-B5 | 0.5794 | 0.5520 |
