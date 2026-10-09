# Test results -- round_8

## Per-run results (CEI test set)

| Key | Model | Source | OA | mIoU | mF1 |
| --- | --- | --- | --- | --- | --- |
| m1 | FT-UNetFormer Swin-B | oem | 0.8381 | 0.5638 | 0.6861 |
| m2 | U-Net EfficientNet-B4 | oem | 0.6381 | 0.4529 | 0.6055 |
| m3 | UNetFormer ResNet-101 | oem | 0.8014 | 0.5179 | 0.6403 |
| m4 | UPerNet Swin-B | oem | 0.8367 | 0.5659 | 0.6829 |
| m5 | SegFormer MiT-B5 | oem | 0.6238 | 0.4269 | 0.5756 |
| m1 | FT-UNetFormer Swin-B | irsa | 0.8563 | 0.5690 | 0.6934 |
| m2 | U-Net EfficientNet-B4 | irsa | 0.8351 | 0.5554 | 0.6798 |
| m3 | UNetFormer ResNet-101 | irsa | 0.8239 | 0.5452 | 0.6680 |
| m4 | UPerNet Swin-B | irsa | 0.8580 | 0.5677 | 0.6948 |
| m5 | SegFormer MiT-B5 | irsa | 0.8591 | 0.5640 | 0.6872 |

## Generalization comparison (mIoU on CEI)

| Key | Model | OEM->CEI | IRSA->CEI |
| --- | --- | --- | --- |
| m1 | FT-UNetFormer Swin-B | 0.5638 | 0.5690 |
| m2 | U-Net EfficientNet-B4 | 0.4529 | 0.5554 |
| m3 | UNetFormer ResNet-101 | 0.5179 | 0.5452 |
| m4 | UPerNet Swin-B | 0.5659 | 0.5677 |
| m5 | SegFormer MiT-B5 | 0.4269 | 0.5640 |
