# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A semantic-segmentation pipeline for land-use/land-cover (LULC) mapping, built around a CEI-specific 7-class land-cover scheme (the repo does not spell out what "CEI" stands for). Models are trained on public benchmark datasets — **OpenEarthMap (OEM)** and **IRSAMap** — with their labels remapped into the CEI taxonomy, then evaluated on a small hand-labeled CEI test set (100 tiles from `maesuai_1..100`). The core research question the repo is set up to answer is "which architecture / which source dataset generalizes best to CEI imagery," so most configs exist in matched train/test pairs across five architectures.

Full narrative documentation already exists and should be your first stop before re-deriving things from code:
- `README.md` — architecture/pipeline overview (**partially stale** — see "Known inconsistencies" below).
- `GUIDELINE.md` — command-line reference with copy-paste examples for every script (**mostly current**, but its `configs/unet/...` example paths are stale — substitute `configs/oem/` and `configs/cei_oem/`).
- `docs/CODE_WALKTHROUGH.md` — line-by-line walkthrough of the training pipeline in call order (**conceptually accurate**, but was written before UNetFormer/FT-UNetFormer/UPerNet/SegFormer and IRSAMap were added — treat model/dataset specifics with that in mind).
- `configs/cei_oem/Script.md` — the 5-architecture OEM→CEI comparison experiment (models, smoke test, train/test commands).
- `configs/cei_irsa/Script.md` — the IRSAMap→CEI counterpart experiment, including the two IRSA-specific quirks (background handling, `nodata_to_ignore`).

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Prepare data (once per dataset)
python tools/oem/create_labeled_splits.py --config configs/oem/unet_effb4_oem.yml
python tools/irsa/make_splits.py --stratify

# Train / fine-tune
python train.py --config <config.yml>
python train.py --config <config.yml> --init-weights <checkpoint_or_state_dict>

# Evaluate / test
python evaluate.py --config <config.yml> --checkpoint <ckpt.pth> --split test [--tta]
python test.py --config <config.yml> --checkpoint <ckpt.pth>        # shortcut for --split test

# Predict
python predict.py --config <config.yml> --checkpoint <ckpt.pth> --input <file_or_dir> [--panel] [--tile_size N --overlap N] [--tta]

# Smoke-test all 5 CEI architectures (3 epochs each, train + eval on real data — not a unit test)
python tools/cei/make_smoke_configs.py --epochs 3
python tools/cei/run_smoke_tests.py [--models m2 m5] [--skip-test]

# Diagnostics
python tools/oem/check_dataset.py --config <config.yml> --split train --num_samples 5
python tools/oem/check_missing_files.py --config <config.yml> --split train
python tools/oem/count_pixels.py --config <config.yml> --split train
python tools/oem/test_dataset_loader.py
python tools/cei/compare_predictions.py --config <config.yml> --checkpoint <ckpt.pth> --tta --num 12 --out <out.png>
python tools/show_confusion.py <metrics.json>   # check actual CLI in the file before relying on this
```

There is currently **no unit test suite** in the repo (README's `tests/smoke_test.py` no longer exists). `tools/cei/run_smoke_tests.py` is the closest thing — it's an integration smoke test that actually trains each of the 5 CEI architectures for a few epochs on real data and evaluates on CEI, not a fast synthetic-data check. There is no linter/formatter config either.

## Architecture

### Pipeline

```
YAML config
  → build_dataset(config, split)      dataset_factory.py dispatches on dataset.name
  → OpenEarthMapDataset / IRSADataset  read image+mask, remap labels, augment, normalize
  → DataLoader
  → build_model(config)               model_factory.py dispatches on model.name
  → build_loss(config)                loss_factory.py dispatches on training.loss
  → train_one_epoch / validate_one_epoch   (src/engine/)
  → SegmentationMetrics                confusion-matrix-based OA / mIoU / mF1 (src/metrics/)
  → checkpoints (last + best-by-mIoU), JSON logs
```

Everything is config-driven from one YAML file passed via `--config`; `train.py:validate_config()` enforces the required top-level sections (`experiment`, `dataset`, `model`, `training`) and that `dataset.num_classes == model.num_classes` before anything expensive runs.

### The taxonomy layer (`src/datasets/taxonomy.py`) is the crux of this codebase

Three label spaces coexist and this module is the single source of truth for converting between them:
- **OEM native**: raw codes `1-8` on disk → training indices `0-7` (`label_map: oem`).
- **CEI** (the target scheme, 7 classes): OEM's Bareland + Developed space merged into one "Non-vegetated" class; every other class maps 1:1. Raw CEI files are already `1-7` (`label_map: cei`); OEM imagery can be remapped straight into CEI space (`label_map: oem_to_cei`) via `OEM_TO_CEI`.
- **IRSAMap**: two-digit codes (tens=major class, ones=subclass) mapped into CEI via `IRSA_TO_CEI` (`label_map: irsa_to_cei`). Unlike OEM, IRSA's background code `0` maps to a *real* class (Non-vegetated), not ignore — see the "IRSA specifics" note below.

`build_label_lut(label_map, ignore_index)` builds a 256-entry uint8 LUT so a whole mask is remapped with one `lut[mask]` vector op; any raw value not in the scheme (including 0 for OEM/CEI) becomes `ignore_index`. Both dataset classes validate raw mask values against the allowed set before converting, so a `label_map` mismatched to the data fails loudly instead of silently dropping classes.

`src/datasets/label_mapping.py` wraps this LUT logic behind two small injectable-collaborator abstractions (`LabelMapper`/`LutLabelMapper`, `MaskRefiner`/`NodataMasker`) so `IRSADataset` can add IRSA-specific behavior without subclassing or forking `OpenEarthMapDataset`.

**IRSA specifics**: IRSAMap leaves bareland unannotated, so its background code `0` is mostly real ground and maps to Non-vegetated rather than ignore — but ~14% of that background is actually near-black nodata border padding sharing the same code. `nodata_to_ignore` (an image-brightness threshold, default `8` for IRSA) tells them apart via `NodataMasker`, which inspects the *image* pixels (not just the mask) to reclassify near-black regions as ignore. This must match between a training config and any test config sharing that data, or class balance shifts between them. Also note IRSAMap ships two non-interchangeable combined-mask folders (`SegLabel_vwsbr` vs `SegLabel_rvwsb`, differing in road/building overlap priority) — `mask_dir` picks one explicitly.

### Dataset loading (`src/datasets/`)

- `dataset_factory.py` — `build_dataset(config, split)`. `OpenEarthMap` and `CEI` share `OpenEarthMapDataset` (identical `<region>/images` + `<region>/labels` layout, `.tif`, differing only in `label_map`); `IRSA_Map` uses `IRSADataset`; `LoveDA` is an unimplemented stub.
- `openearthmap_dataset.py` — region resolved from filename via `rsplit("_", 1)` (so multi-word regions like `santa_rosa` survive); train transforms pad-then-random-crop (+ optional flip/rotate augmentation, off by default to match the OEM paper recipe); eval transforms are deterministic, either `eval_mode: full` (pad to a multiple of 32, keep native resolution — the paper protocol) or `crop` (center-crop, faster).
- `irsa_dataset.py` — separate train/test directory trees, own split files (generated by `tools/irsa/make_splits.py`, not checked in), injects `LutLabelMapper`/`NodataMasker` with IRSA defaults.
- `transforms.py` — normalization presets (`imagenet` vs `zero_one`); **must match what the checkpoint's weights were trained with** — using the wrong one produces incoherent (not just slightly worse) predictions.

### Models (`src/models/model_factory.py`)

Five architectures behind one `build_model(config)` dispatch on `model.name`: `unet` (segmentation-models-pytorch), `deeplabv3`, `upernet` (needs a `tu-` timm backbone name + `encoder_params.img_size`), `unetformer` (CNN encoder + Global-Local Transformer decoder, custom impl in `unetformer.py`), `ftunetformer` (fixed Swin-B encoder, custom impl in `ftunetformer.py`), `segformer`. UNetFormer/FT-UNetFormer return `(main, aux)` during training — `src/engine/trainer.py:compute_loss()` is the one place that branches on tuple-vs-tensor output to add the auxiliary loss (weight 0.4, the paper's value).

`LeadingChannelDrop` exists solely to consume the external OpenEarthMap-SAR pretrained U-Net, which has a 9-class head (index 0 = background); it slices off leading channel(s) so the rest lines up with this project's 0-based class order. Loading that checkpoint also requires `dataset.normalization: zero_one` (not `imagenet`) and a bare-`state_dict` file, both handled by `src/models/checkpoint.py:load_model_weights()` and the `unet_effb4_oem_pretrained.yml` / `unet_effb4_oem_finetune.yml` configs.

### Training / eval loop (`src/engine/`, `train.py`, `evaluate.py`)

- `train.py` is the entry point: builds everything, runs the epoch loop, writes `experiments/<name>/checkpoints/{last,best}_checkpoint.pth` and `logs/training_logs.json` every epoch (so a crash loses at most the current epoch). **mIoU on the validation set decides "best"**, not loss or accuracy. There is no resume-from-checkpoint flag — re-running a config starts fresh and overwrites its output folder. `--init-weights` warm-starts from a checkpoint or bare state_dict (for fine-tuning) and skips the ImageNet encoder download.
- `evaluate.py` loads a checkpoint, and — because every checkpoint saved here embeds the config that produced it — `check_architecture_matches()` compares the checkpoint's stored architecture against the config passed on the command line and fails with a readable diff instead of a wall of `state_dict` key-mismatch errors. This is the guard against the most common real mistake: pairing one model's test config with another model's checkpoint. `--tta` wraps the model in `FlipTTA` (4-way flip averaging in logit space) for both `evaluate.py` and `predict.py`.
- Mixed precision (`training.mix_precision`) only engages on CUDA; it's silently skipped on CPU/MPS.
- Validation uses batch size 1 whenever `eval_mode: full`, because full-resolution OEM/IRSA tiles aren't uniformly sized and can't be stacked into a batch.

### Metrics (`src/metrics/segmentation_metrics.py`)

`SegmentationMetrics` accumulates a confusion matrix across the whole split (not averaged per-batch) via one vectorized `bincount` per update, then computes OA/mIoU/mF1/per-class IoU-F1/support/full confusion matrix in `compute()`. Ignored and out-of-range pixels are dropped before updating. `mIoU` (not OA) is the model-selection metric because OA can look good while rare classes fail completely.

### The CEI hand-labeling workflow (`tools/cei/`)

CEI imagery has no ground truth, so labels are bootstrapped from model predictions and hand-corrected in GIMP:
```
predict_for_gimp.py → (edit *_mask.tif in GIMP) → import_from_gimp.py → data/CEI_data/labels/
```
`predict_for_gimp.py` never overwrites an existing `_mask.tif` unless `--overwrite` is passed, so re-running to add a new batch of tiles is always safe. `import_from_gimp.py` snaps each pixel to the nearest palette color (tolerates GIMP anti-aliasing) and writes raw CEI-encoded (`1-7`, `0`=unlabeled) single-channel PNGs — warns if too few pixels are exactly on-palette (re-export as TIFF, not JPEG) or if a mask's size drifted from its image. `make_test_split.py` regenerates the CEI test split from whatever tiles currently have labels (labeling happens incrementally in batches of ~100).

## Known inconsistencies (current as of this file, verify before trusting)

- `README.md`'s repository-structure tree, its `configs/unet/unet_effb4_oem.yml`-style paths, and its "OpenEarthMap only, IRSA/LoveDA reserved" framing are stale. Configs actually live under `configs/oem/`, `configs/cei_oem/`, `configs/cei_irsa/`; `IRSA_Map` is implemented (`src/datasets/irsa_dataset.py`); the model factory supports 5 architectures, not just U-Net/DeepLabV3.
- `tools/cei/make_smoke_configs.py` and `tools/cei/run_smoke_tests.py` hard-code `CONFIG_DIR = configs/unet/cei`, a path that no longer exists (renamed to `configs/cei_oem` in commit `3180f2e`). **These two scripts are currently broken** until `CONFIG_DIR` (and `SMOKE_DIR` in `run_smoke_tests.py`) are updated to `configs/cei_oem`.
- `README.md` references `tests/smoke_test.py`; it does not exist in this repo.
