# CEI qualitative report -- IRSA-trained (SegFormer MiT-B5)

- Selected: `m5` -- best native val mIoU 0.8368 @ epoch 95 (not the final epoch, 100)
- Checkpoint: `experiments\cei_irsa_m5_segformer_mitb5\checkpoints\best_checkpoint.pth`
- Config: `configs/cei_irsa/test/test_m5_segformer_on_cei.yml`
- TTA: True

## CEI test set

OA **0.8591**  mIoU **0.5640**  mF1 **0.6872**

## Worst confusions (% of the true class)

| True | Predicted as | Share |
| --- | --- | --- |
| Rangeland | Non-vegetated | 40.7% |
| Road | Non-vegetated | 39.2% |
| Building | Non-vegetated | 38.8% |
| Agriculture | Rangeland | 36.5% |
| Agriculture | Non-vegetated | 24.1% |
| Agriculture | Tree | 15.6% |
| Non-vegetated | Rangeland | 15.2% |
| Rangeland | Tree | 13.5% |

Confusion heatmap: `confusion_row.png`

10 tile figure(s) in this folder.
