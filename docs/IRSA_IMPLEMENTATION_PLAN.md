# IRSAMap implementation plan

Goal: put IRSAMap through the same flow OpenEarthMap already follows -- train on
it in the 7-class CEI scheme, then test on the held-out CEI tiles.

Everything below is based on inspecting the data on disk, not on assumptions
about how the dataset is documented elsewhere.

---

## 1. What is actually there

```
data/IRSAMap/
    train/                        4617 images
        image/                    1024x1024 RGB .png
        SegLabel_rvwsb/           4617 combined masks   <- full coverage
        SegLabel_vwsbr/           4617 combined masks   <- full coverage
        label_building/            3891 \
        label_roadarea/            4078  |  per-class masks, sparse:
        label_sport/                328  |  only tiles containing that class
        label_vege/                4500  |
        label_water/               1672 /
        label_geojson_*/                 vector sources for the above
    test/                          912 images, same structure
```

Findings that shape the plan:

* **Images are 1024x1024 PNG**, not the variable-size TIFs of OpenEarthMap.
* **`SegLabel_*` are the only densely-covering labels.** Every image has one. The
  `label_<class>` folders are sparse (a tile appears only if it contains that
  class), so they are per-class sources, not an alternative to `SegLabel_*`.
* **The two `SegLabel_` folders are not copies.** Of 200 tiles compared, 169
  differ. They contain the same class codes with the same total footprint; what
  changes is which class wins where regions overlap. The suffixes are the burn
  order -- `rvwsb` = road, vege, water, sport, building; `vwsbr` = vege, water,
  sport, building, road. So `rvwsb` puts **building on top**, `vwsbr` puts
  **road on top**.
* **There is no validation split.** Only `train/` and `test/` exist.

## 2. The label codes

The masks use a two-digit scheme: tens digit = major class, ones digit =
subclass. Decoded by intersecting `SegLabel_vwsbr` with each `label_<class>`
folder, then confirmed by eye on sampled image patches:

| Code | Pixels (sample) | Meaning | Evidence |
| --- | --- | --- | --- |
| 0 | 23.05% | background / unlabeled | -- |
| 10 | 9.06% | cropland | 100% inside `label_vege`; patches show ploughed rows, orchards |
| 11 | 20.42% | forest / tree | inside `label_vege`; dense canopy |
| 12 | 20.67% | grass / sparse vegetation | inside `label_vege`; dry scrub, thin grass |
| 21 | 5.00% | water | 100% inside `label_water` |
| 22 | 1.61% | water | inside `label_water` |
| 23 | 0.09% | water (rare) | inside `label_water`; only 1 clean sample in 400 tiles |
| 24 | 2.58% | water | inside `label_water` |
| 31 | 11.48% | building | 99.6% inside `label_building`; rooftops |
| 32 | 5.88% | road | 100% inside `label_roadarea`; lane markings, car parks |
| 34 | 0.16% | sport | 99.9% inside `label_sport`; pitches, running tracks |

Codes `30` and `33` never appear. `33` is most likely road-centreline
(`label_geojson_roadline` exists with no raster counterpart) but nothing on disk
confirms it, so the plan treats any unexpected code as an error rather than
guessing.

## 3. Key finding: no new dataset class is needed

`src/datasets/irsa_dataset.py` is a reserved stub and the factory raises
`NotImplementedError`. Neither needs to be filled in.

`OpenEarthMapDataset._resolve_region_path` uses a directory pattern as-is when it
contains no `<region>` token -- the same property that let CEI reuse this loader.
Verified directly:

```
image_dir: train/image        -> data/IRSAMap/train/image/1252.png      exists
mask_dir:  train/SegLabel_vwsbr -> data/IRSAMap/train/SegLabel_vwsbr/1252.png  exists
```

`cv2.imread` handles PNG and TIF identically, and `_convert_mask` already does
arbitrary code remapping through a 256-entry lookup table.

**So IRSA is a taxonomy + config problem, not a loader problem.** That keeps one
data path under test for all three datasets instead of three near-copies.

The one structural difference from OEM: train and test live in *different*
directories. That is already handled the same way as OEM->CEI -- a separate test
config with its own `image_dir`/`mask_dir` -- so it costs nothing.

---

## 4. Steps

### Step 1 -- Add IRSA label maps to `taxonomy.py`

Two maps, mirroring how OEM has both a native and a CEI-facing scheme:

```python
# Raw IRSA code -> CEI class index. Water subclasses collapse to one class;
# the three vegetation subclasses carry distinctions CEI also draws.
IRSA_TO_CEI = {
    10: 1,   # cropland          -> Agriculture
    11: 2,   # forest            -> Tree
    12: 0,   # grass / sparse    -> Rangeland
    21: 3, 22: 3, 23: 3, 24: 3,   # all water -> Water
    31: 4,   # building          -> Building
    32: 5,   # road              -> Road
    34: 6,   # sport             -> Non-vegetated
}
```

Add an `"irsa_to_cei"` branch to `build_label_lut`. Code `0` falls through to
`ignore_index` automatically, as unlabeled already does for OEM/CEI.

`num_classes` is derived from `len(set(pairs.values()))`, so it returns 7 with no
extra bookkeeping.

**Verify:** assert every code in the table appears in the data and vice versa --
a code present on disk but missing from the map would otherwise become `ignore`
silently, quietly deleting a class.

### Step 2 -- Generate split files

IRSA ships no split files and no validation set. Add
`tools/irsa/make_splits.py`, mirroring `tools/cei/make_test_split.py`:

* `train/` (4617) -> `train_split.txt` + `val_split.txt`, split ~90/10 with a
  fixed seed so the split is reproducible.
* `test/` (912) -> `test_split.txt`.
* Validate as it goes: every image has a mask, every mask contains only known
  codes, and report the class distribution of each split.

Write splits to `data/IRSAMap/`, matching where OEM keeps its `*.txt`.

**Watch for:** `sport` is 0.16% of pixels and appears in only 328/4617 tiles. A
naive random split can leave it badly represented in `val`. Report per-split
class coverage so this is visible; stratify if it turns out lopsided.

### Step 3 -- Register the dataset name

In `dataset_factory.py`, replace the `NotImplementedError` with the same reuse
CEI already gets:

```python
if dataset_name in ("OpenEarthMap", "CEI", "IRSA_Map"):
    return OpenEarthMapDataset(config, split=split)
```

Update the comment to say why three names share one loader.

### Step 4 -- Configs

Three files, matching the existing naming:

| File | Purpose |
| --- | --- |
| `configs/irsa/unet_effb4_irsa2cei.yml` | train on IRSA in the CEI scheme |
| `configs/irsa/test/test_irsa2cei_on_cei.yml` | test that model on the 100 CEI tiles |
| `configs/irsa/test/test_irsa2cei_on_irsa.yml` | in-domain test on IRSA's own test split |

The training config differs from `unet_effb4_oem2cei.yml` in only five lines:

```yaml
dataset:
  name: IRSA_Map
  root: data/IRSAMap
  image_dir: train/image           # no <region> token -- used as-is
  mask_dir: train/SegLabel_vwsbr   # road wins on overlap; see decision 2
  label_map: irsa_to_cei
```

The in-domain test config points at `test/image` and `test/SegLabel_vwsbr`; the
CEI test config is reused unchanged from the existing five-model set.

Reuse the CEI test config unchanged -- it already points at `data/CEI_data` and
is model-agnostic apart from the `model:` block.

### Step 5 -- Verify before training

The checks that would have caught the bugs hit on the CEI work:

1. Every split loads; sample shapes and class values are in range `0..6` + 255.
2. Decoded masks match the imagery visually -- render image / ground truth for a
   few tiles, as was done for CEI. This is what confirms code 11 really is forest
   and not, say, cropland.
3. Class distribution of IRSA-as-CEI vs OEM-as-CEI vs the CEI test set, so the
   domain gap is known before it shows up as a bad number.
4. A 3-epoch smoke run via the existing `run_smoke_tests.py` pattern.

### Step 6 -- Train and compare

Once `irsa_to_cei` works, IRSA slots into the existing five-model comparison with
no further changes -- the model configs are independent of which dataset feeds
them. The interesting result is the three-way comparison on the same CEI test
set: **OEM-trained vs IRSA-trained vs both combined**.

---

## 5. Decisions taken

**1. `sport` (34) -> Non-vegetated.** CEI's Non-vegetated definition explicitly
includes sport surfaces, and a CEI running track (tile 81) was confirmed labeled
Non-vegetated. The caveat stands: sampled IRSA `sport` patches are mostly *grass
pitches*, so this teaches the model that some green grass is Non-vegetated.
Chosen anyway because consistency with the CEI test set matters more than
appearance -- the alternative would train against the convention the model is
scored on. At 0.16% of pixels the blast radius is small. Splitting grass pitches
from hard courts is not possible: code 34 does not distinguish them.

**2. `SegLabel_vwsbr`.** Road wins where it overlaps building. Chosen because
road continuity matters for the Road class, already the second-weakest in the
OEM->CEI results (IoU 0.47) while Building is strong (0.74). Worth an ablation
against `rvwsb` later, but not a blocking decision.

**3. Vegetation mapping confirmed:** 10 -> Agriculture, 11 -> Tree,
12 -> Rangeland. Still verify visually in Step 5 before training -- this is
inferred from sampled patches, not documentation, and covers 50% of labeled
pixels.

**4. Scope: IRSA -> CEI only.** No native 10-class scheme for now. A 10-class
model could not be tested on CEI without a second remapping, and the question
this answers -- does IRSA transfer to CEI better than OEM? -- needs only the
7-class scheme.

### Accepted, not a blocker: 23% background

Nearly a quarter of every IRSA image is unlabeled and will be ignored by the
loss. OEM is densely labeled by comparison. This is not a bug -- `ignore_index`
handles it correctly -- but it means IRSA delivers roughly 77% of the supervision
per pixel that its tile count suggests, and any region the annotators skipped is
invisible during training.

## 6. Risks

**The vegetation mapping is the highest-stakes guess.** 10/11/12 -> Agriculture /
Tree / Rangeland comes from visual inspection of four patches each, not from
dataset documentation. Those three classes are 50% of all labeled pixels, so
getting one wrong would corrupt half the training signal while still producing a
plausible-looking loss curve. Step 5's visual check is what catches this, and it
should be done on a few dozen tiles, not four.

**Rangeland is already the weakest CEI class** (IoU 0.30 in the OEM->CEI run,
confused with Tree 34% of the time). IRSA code 12 maps straight onto it, so if
12 is really "sparse/dry vegetation" and CEI Rangeland means managed grassland,
IRSA may make that class worse rather than better.

**Scale.** 4617 IRSA training tiles at 1024x1024 versus 2100 OEM tiles. Epochs
will take roughly twice as long at the same crop size.

---

## 7. Effort

| Step | Work |
| --- | --- |
| 1. Taxonomy map | ~25 lines in `taxonomy.py` + tests |
| 2. Split tool | ~150 lines, closely modeled on `tools/cei/make_test_split.py` |
| 3. Factory | 2 lines |
| 4. Configs | 3 files, mostly copied |
| 5. Verification | the real work -- visual confirmation of the class mapping |
| 6. Smoke run | one command, existing tooling |

No new dataset class, no changes to the trainer, validator, metrics, or model
factory. The load-bearing risk is entirely in Step 5 confirming Step 1's mapping.
