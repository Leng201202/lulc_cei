# CEI qualitative report -- OEM-trained (FT-UNetFormer Swin-B)

- Selected: `m1` -- best native val mIoU 0.7194 @ epoch 95 (not the final epoch, 100)
- Checkpoint: `experiments\cei_oem_m1_ftunetformer_swinb\checkpoints\best_checkpoint.pth`
- Config: `configs/cei_oem/test/test_m1_ftunetformer_cei.yml`
- TTA: True

## CEI test set

OA **0.8381**  mIoU **0.5638**  mF1 **0.6861**

## Worst confusions (% of the true class)

| True | Predicted as | Share |
| --- | --- | --- |
| Non-vegetated | Rangeland | 60.2% |
| Road | Non-vegetated | 26.3% |
| Agriculture | Rangeland | 23.3% |
| Rangeland | Tree | 12.3% |
| Agriculture | Tree | 11.8% |
| Building | Water | 11.7% |
| Building | Non-vegetated | 9.3% |
| Road | Tree | 8.7% |

Confusion heatmap: `confusion_row.png`

10 tile figure(s) in this folder.
