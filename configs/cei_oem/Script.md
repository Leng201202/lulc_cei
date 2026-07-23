# Train on OpenEarthMap, test on CEI — four architectures

Each model is trained on OEM imagery remapped into the 7-class CEI scheme, then
tested on the hand-labeled CEI tiles it has never seen. Data, splits, classes and
schedule are identical across models, so the comparison isolates architecture.

| | |
| --- | --- |
| Train / val | OpenEarthMap, 2100 / 350 tiles, `label_map: oem_to_cei` |
| Test | CEI, 100 tiles (`maesuai_1..100`), `label_map: cei` |
| Classes | 7 (Unlabeled is ignore, not a channel) |

## The models

| # | Model | Train config | Test config | Params |
| --- | --- | --- | --- | --- |
| 1 | FT-UNetFormer + Swin-B | `ftunetformer_swinb_oem2cei.yml` | `test/test_m1_ftunetformer_cei.yml` | 96.0M |
| 2 | U-Net + EfficientNet-B4 | `unet_effb4_oem2cei.yml` | `test/test_m2_uneteffb4_cei.yml` | 20.2M |
| 3 | UNetFormer + ResNet-101 | `unetformer_r101_oem2cei.yml` | `test/test_m3_unetformer_cei.yml` | 43.2M |
| 4 | UPerNet + Swin-B | `upernet_swinb_oem2cei.yml` | `test/test_m4_upernet_cei.yml` | 96.9M |
| 5 | SegFormer + MiT-B5 | `segformer_mitb5_oem2cei.yml` | `test/test_m5_segformer_cei.yml` | 82.0M |

Batch size and learning rate differ by architecture (CNNs take batch 8 / lr 1e-4;
the three transformer models take batch 4 / lr 6e-5 / weight decay 0.01, the
standard transformer recipe). That is deliberate -- matching them would handicap
one family -- but it does mean the comparison is not a pure architecture-only
ablation.

One asymmetry worth knowing when reading results: the two Swin models (1 and 4)
are built for a fixed input size, so their test configs set the encoder to 1024
to match the CEI tiles. Their weights are resolution-independent, so a
512-trained checkpoint loads unchanged -- but it is an extra moving part.
SegFormer and the CNNs need no such adjustment.

## Smoke test first

Before committing days of GPU time, run three epochs of every model and score
each on CEI:

```powershell
python tools/cei/make_smoke_configs.py --epochs 3
python tools/cei/run_smoke_tests.py
```

Failures are recorded and the suite continues, so one broken architecture does
not hide the state of the others; the exit code is non-zero if anything failed.
Results land in `experiments/smoke_suite_results.json`.

A pass means the model builds, trains without NaN or OOM at its configured batch
size, and scores on CEI end to end. It does **not** rank the architectures --
three epochs is far too few. For reference, model 2's Water IoU went 0.00 -> 0.72
between epochs 2 and 5, so any ordering this early is noise.

---

## Train

```powershell
python train.py --config configs/unet/cei/unet_effb4_oem2cei.yml
python train.py --config configs/unet/cei/unetformer_r101_oem2cei.yml
python train.py --config configs/unet/cei/ftunetformer_swinb_oem2cei.yml
python train.py --config configs/unet/cei/upernet_swinb_oem2cei.yml
python train.py --config configs/unet/cei/segformer_mitb5_oem2cei.yml
```

Each writes to its own `experiments/<name>/`, so the four runs never collide.

EfficientNet-B4 runs about 2 min/epoch on the RTX 4000 (~6-7 h for 200 epochs).
The 96M-parameter Swin models are several times slower -- budget considerably
more, and consider a shorter schedule for the first comparison since relative
ranking usually settles well before the final metric does.

Writes as it goes:

```
experiments/cei_exp01_oem2cei/
    config.yml                          copy of the recipe used
    checkpoints/best_checkpoint.pth     highest OEM-val mIoU  <- test with this
    checkpoints/last_checkpoint.pth     most recent epoch
    logs/training_logs.json             full per-epoch history
```

There is no resume flag: re-running starts from scratch and overwrites the folder.

## Test on CEI

Pair each model's test config with its own checkpoint:

```powershell
python evaluate.py --config configs/unet/cei/test/test_m2_uneteffb4_cei.yml --checkpoint experiments/cei_exp01_oem2cei/checkpoints/best_checkpoint.pth --split test --tta --output experiments/cei_exp01_oem2cei/logs/cei_test_tta.json

python evaluate.py --config configs/unet/cei/test/test_m3_unetformer_cei.yml --checkpoint experiments/cei_m3_unetformer_r101/checkpoints/best_checkpoint.pth --split test --tta --output experiments/cei_m3_unetformer_r101/logs/cei_test_tta.json

python evaluate.py --config configs/unet/cei/test/test_m1_ftunetformer_cei.yml --checkpoint experiments/cei_m1_ftunetformer_swinb/checkpoints/best_checkpoint.pth --split test --tta --output experiments/cei_m1_ftunetformer_swinb/logs/cei_test_tta.json

python evaluate.py --config configs/unet/cei/test/test_m4_upernet_cei.yml --checkpoint experiments/cei_m4_upernet_swinb/checkpoints/best_checkpoint.pth --split test --tta --output experiments/cei_m4_upernet_swinb/logs/cei_test_tta.json

python evaluate.py --config configs/unet/cei/test/test_m5_segformer_cei.yml --checkpoint experiments/cei_m5_segformer_mitb5/checkpoints/best_checkpoint.pth --split test --tta --output experiments/cei_m5_segformer_mitb5/logs/cei_test_tta.json
```

`--tta` averages predictions over 4 flips. It is slower but consistently better
(+0.03 mIoU in the 5-epoch trial), so use it for any number you report. Drop it
for a quick check.

Prints OA / mIoU / mF1 and saves per-class IoU, per-class F1, class support, and
the confusion matrix to the `--output` file.

### Same model on the OEM test set

For the in-domain number to compare against:

```powershell
python evaluate.py --config configs/unet/cei/unet_effb4_oem2cei.yml --checkpoint experiments/cei_exp01_oem2cei/checkpoints/best_checkpoint.pth --split test --tta --output experiments/cei_exp01_oem2cei/logs/oem_test_tta.json
```

The gap between the two is the cross-dataset generalization result exp_1 exists
to measure.

## Predict masks

```powershell
python predict.py --config configs/unet/cei/test_cei.yml --checkpoint experiments/cei_exp01_oem2cei/checkpoints/best_checkpoint.pth --input data/CEI_data/images --output experiments/cei_exp01_oem2cei/predictions --panel --tta
```

`--panel` also saves an image+prediction side-by-side view per tile, which is the
fastest way to eyeball where the model fails. Add `--format tiff` for lossless
masks to hand-correct in GIMP, and `--tile_size 1024` for imagery too large to
fit in memory.

---

## Rebuild the test split

Re-run after each labeling batch — do not edit the list by hand:

```powershell
python tools/cei/make_test_split.py --start 1 --end 100
```

Writes `data/CEI_data/test_split.txt`, skips masks with no matching image or with
label values outside 0-7, and prints the class distribution. Blank/nodata tiles
are kept; pass `--exclude-blank` to drop them.

## Reference

| Item | Value |
| --- | --- |
| Train config | `configs/unet/cei/unet_effb4_oem2cei.yml` |
| Test config | `configs/unet/cei/test_cei.yml` |
| Class definitions | `src/datasets/taxonomy.py` |
| Code explained | `docs/CODE_WALKTHROUGH.md` |

`--split` accepts `train`, `val`, or `test`. Both configs must agree on
`num_classes`, `ignore_index`, and `normalization`, or results are silently
wrong — `validate_config` catches the first, the other two are on you.
