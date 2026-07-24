# Train on IRSAMap, test on CEI

The IRSA counterpart of `configs/cei_oem/`. Same model, same recipe, same 7 CEI
classes -- only the training source differs, so results are directly comparable
on the shared 100-tile CEI test set.

| | |
| --- | --- |
| Train / val | IRSAMap, 4155 / 462 tiles from `train/`, `label_map: irsa_to_cei` |
| In-domain test | IRSAMap, 912 tiles from `test/` |
| Cross-dataset test | CEI, 100 tiles (`maesuai_1..100`), `label_map: cei` |
| Output | `experiments/cei_irsa01_irsa2cei/` |

## Regenerate splits

IRSAMap ships no split files and no validation set, so these are generated:

```powershell
python tools/irsa/make_splits.py --stratify
```

Validates every mask against the known IRSA code set, carves a 10% validation
set out of `train/` with a fixed seed, and reports per-split class distribution.
`--stratify` spreads the rare classes (sport, water subtype c) across train/val
instead of splitting purely at random.

## Train

```powershell
python train.py --config configs/cei_irsa/unet_effb4_irsa2cei.yml
```

4155 tiles vs OpenEarthMap's 2100, so epochs take roughly twice as long.

## Test

Cross-dataset -- the number to compare against the OEM-trained models:

```powershell
python evaluate.py --config configs/cei_irsa/test/test_irsa2cei_on_cei.yml --checkpoint experiments/cei_irsa01_irsa2cei/checkpoints/best_checkpoint.pth --split test --tta --output experiments/cei_irsa01_irsa2cei/logs/cei_test_tta.json
```

In-domain, for the generalisation gap:

```powershell
python evaluate.py --config configs/cei_irsa/test/test_irsa2cei_on_irsa.yml --checkpoint experiments/cei_irsa01_irsa2cei/checkpoints/best_checkpoint.pth --split test --tta --output experiments/cei_irsa01_irsa2cei/logs/irsa_test_tta.json
```

---

## Two things specific to IRSA

**Background is Non-vegetated, not ignore.** IRSAMap annotates five thematic
classes and leaves bareland unlabeled, so code `0` is mostly real
Non-vegetated ground -- 23.6% of all pixels, of which 85.7% is bare soil,
concrete or paved lots. Sending it to ignore would leave Non-vegetated with only
sport: 0.21% of IRSA pixels against 8.26% in the CEI test set.

**`nodata_to_ignore: 8` handles the exception.** The other 14.3% of background is
the near-black border padding of the source imagery. Both carry mask value `0`,
so the loader uses the image to tell them apart: background stays Non-vegetated
unless the pixel is near-black. Without this, the model learns "black region ->
Non-vegetated" and mislabels the blank CEI captures (`maesuai_1`, `maesuai_5`),
which are ground-truthed as Water.

This flag must match between the training config and the in-domain test config,
or the class balance shifts between them.

## Reference

| Item | Value |
| --- | --- |
| Label mapping | `IRSA_TO_CEI` in `src/datasets/taxonomy.py` |
| Split tool | `tools/irsa/make_splits.py` |
| Design notes | `docs/IRSA_IMPLEMENTATION_PLAN.md` |

`SegLabel_vwsbr` is used rather than `SegLabel_rvwsb`. They are not copies -- 169
of 200 sampled tiles differ -- and encode different overlap priority: `vwsbr`
puts road above building, `rvwsb` the reverse. Switching is a one-line change in
`mask_dir` and worth an ablation.
