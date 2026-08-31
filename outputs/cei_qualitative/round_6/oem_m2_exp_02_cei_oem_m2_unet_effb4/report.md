# CEI qualitative report -- OEM-trained (U-Net EfficientNet-B4)

- Selected: `m2` -- explicit --checkpoint override (experiments/exp_02_cei_oem_m2_unet_effb4/checkpoints/best_checkpoint.pth)
- Checkpoint: `experiments/exp_02_cei_oem_m2_unet_effb4/checkpoints/best_checkpoint.pth`
- Config: `configs/cei_oem/test/test_m2_uneteffb4_cei.yml`
- TTA: True

## CEI test set

OA **0.6512**  mIoU **0.4155**  mF1 **0.5376**

## Worst confusions (% of the true class)

| True | Predicted as | Share |
| --- | --- | --- |
| Non-vegetated | Rangeland | 68.6% |
| Water | Agriculture | 42.3% |
| Agriculture | Rangeland | 16.2% |
| Agriculture | Tree | 12.1% |
| Road | Rangeland | 10.6% |
| Non-vegetated | Agriculture | 9.5% |
| Tree | Rangeland | 7.4% |
| Road | Tree | 7.4% |

Confusion heatmap: `confusion_row.png`

10 tile figure(s) in this folder.
