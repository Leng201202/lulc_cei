# Test results -- round_2

## Per-run results (CEI test set)

| Key | Model | Source | OA | mIoU | mF1 |
| --- | --- | --- | --- | --- | --- |
| m1 | FT-UNetFormer Swin-B | oem | 0.7797 | 0.6151 | 0.7451 |
| m2 | U-Net EfficientNet-B4 | oem | 0.7811 | 0.6079 | 0.7427 |
| m3 | UNetFormer ResNet-101 | oem | 0.7716 | 0.6105 | 0.7400 |
| m4 | UPerNet Swin-B | oem | 0.7860 | 0.6355 | 0.7601 |
| m5 | SegFormer MiT-B5 | oem | 0.7768 | 0.5919 | 0.7301 |
| m1 | FT-UNetFormer Swin-B | irsa | 0.6801 | 0.5401 | 0.6768 |
| m2 | U-Net EfficientNet-B4 | irsa | 0.7477 | 0.6107 | 0.7369 |
| m3 | UNetFormer ResNet-101 | irsa | 0.6726 | 0.5563 | 0.6875 |
| m4 | UPerNet Swin-B | irsa | 0.6979 | 0.5415 | 0.6806 |
| m5 | SegFormer MiT-B5 | irsa | 0.7046 | 0.5645 | 0.6892 |

## Generalization comparison (mIoU on CEI)

| Key | Model | OEM->CEI | IRSA->CEI |
| --- | --- | --- | --- |
| m1 | FT-UNetFormer Swin-B | 0.6151 | 0.5401 |
| m2 | U-Net EfficientNet-B4 | 0.6079 | 0.6107 |
| m3 | UNetFormer ResNet-101 | 0.6105 | 0.5563 |
| m4 | UPerNet Swin-B | 0.6355 | 0.5415 |
| m5 | SegFormer MiT-B5 | 0.5919 | 0.5645 |
