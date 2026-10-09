# Test results -- round_5

## Per-run results (CEI test set)

| Key | Model | Source | OA | mIoU | mF1 |
| --- | --- | --- | --- | --- | --- |
| m1 | FT-UNetFormer Swin-B | oem | 0.8479 | 0.6016 | 0.7199 |
| m2 | U-Net EfficientNet-B4 | oem | 0.6442 | 0.4729 | 0.6275 |
| m3 | UNetFormer ResNet-101 | oem | 0.8095 | 0.5473 | 0.6684 |
| m4 | UPerNet Swin-B | oem | 0.8456 | 0.6014 | 0.7158 |
| m5 | SegFormer MiT-B5 | oem | 0.6318 | 0.4499 | 0.6006 |
| m1 | FT-UNetFormer Swin-B | irsa | 0.8535 | 0.5743 | 0.7008 |
| m2 | U-Net EfficientNet-B4 | irsa | 0.8360 | 0.5636 | 0.6887 |
| m3 | UNetFormer ResNet-101 | irsa | 0.8273 | 0.5626 | 0.6896 |
| m4 | UPerNet Swin-B | irsa | 0.8535 | 0.5678 | 0.6959 |
| m5 | SegFormer MiT-B5 | irsa | 0.8504 | 0.5591 | 0.6836 |

## Generalization comparison (mIoU on CEI)

| Key | Model | OEM->CEI | IRSA->CEI |
| --- | --- | --- | --- |
| m1 | FT-UNetFormer Swin-B | 0.6016 | 0.5743 |
| m2 | U-Net EfficientNet-B4 | 0.4729 | 0.5636 |
| m3 | UNetFormer ResNet-101 | 0.5473 | 0.5626 |
| m4 | UPerNet Swin-B | 0.6014 | 0.5678 |
| m5 | SegFormer MiT-B5 | 0.4499 | 0.5591 |
