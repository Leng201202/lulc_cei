# OpenEarthMap → CEI Domain Adaptation: Project Context for AI Assistants

**Document purpose:** Transfer the full known background, experiment configuration, quantitative results, observed failures, and next requested coding task to another AI (e.g., Claude Code).  
**Last updated:** 2026-10-09  
**Status:** Normal fine-tuning and strong Agriculture-aware sampling have results. **Mild Agriculture-aware sampling is proposed but no result has yet been supplied.**

> **Evidence levels:** Values in the tables below were supplied by the user. Statements about what caused the results are hypotheses until tested. Details not verified against the repository are identified as such. Do **not** fabricate missing implementation details, training-set counts, hyperparameters, experiments, or test results.

## 1. Research problem and objective

An image semantic segmentation model trained on **OpenEarthMap (OEM)** transfers poorly to a **local CEI target-domain imagery dataset in Thailand**. A key target-domain difference is the prevalence of **perennial agricultural crops** (e.g., rubber plantations, mango orchards, longan orchards). These remain **Agriculture** semantically but may look like dense trees, orchards, or grass/shrub areas in overhead RGB imagery. The objective is to improve CEI segmentation through **supervised fine-tuning of a pre-existing OEM checkpoint**, while preserving performance across all classes.

The principal research questions are:

1. How much does small-sample CEI fine-tuning improve segmentation over OEM-only inference on CEI?
2. How does performance change with **1, 5, 10, 20, 30, 50, 75, 100** CEI-labeled training images?
3. Can agriculture-aware crop sampling improve Agriculture IoU without sacrificing Tree, Water, or global mIoU?
4. How many **diverse** CEI images are needed before local performance begins to stabilize? There is **no known fixed universal threshold**.

## 2. Existing architecture and data conventions

- **Model:** U-Net with **EfficientNet-B4 encoder**.
- **Initial weights for target adaptation:** The saved **best OEM-trained checkpoint**. Never initialize new CEI experiments from random weights or continue each subset size from the previous size unless a separate, explicitly labeled sequential experiment is planned.
- **Training crop size:** `512 × 512`.
- **Task:** pixel-level multiclass semantic segmentation of RGB remote-sensing imagery.
- **Current class count:** **7**; this comes from merging OEM **Bareland** and **Developed Space** into one class, here called **Non-vegetated**.
- **Current reporting order / class names:** `Rangeland`, `Agriculture`, `Tree`, `Water`, `Building`, `Road`, `Non-vegetated`.
- **Class IDs, RGB mask colors, ignore index, remapping rules, normalization, saved checkpoint schema and paths:** **Must be read from the real existing code/configs; do not assume the class reporting order equals numeric IDs.** Earlier discussions referenced remapped OEM labels and an ignored class, but the implementation must be authoritative.
- **Training procedure:** Existing fine-tuning workflow is functional; exact committed hyperparameters and whether freezing/unfreezing is used must be verified in the repository. Low learning rates (~`1e-5` for full-model fine-tuning), early stopping on CEI validation mIoU, moderate paired augmentations, and optional staged unfreezing were proposed.

### Important class semantics

- **Agriculture:** Keep perennial and annual cultivated land within Agriculture where that matches CEI's annotation policy: rubber fields, mango orchards, longan orchards, plantations, and normal farmland.
- **Tree:** Not a synonym for agricultural plantations merely because canopies appear tree-like. Annotation policy, not canopy appearance alone, determines the ground truth.
- **Rangeland:** Grassland/shrub-like areas, subject to the precise OEM/CEI annotation guide.
- **Non-vegetated:** Deliberate merged class, from OEM Bareland + Developed Space. **This merge is separate from the CEI Agriculture domain-shift question**; do not merge Agriculture with Tree/Rangeland to inflate accuracy.
- **All CEI masks must have exactly the same 7-class semantic meaning and training IDs as the saved OEM checkpoint.**

## 3. OEM source-domain context

OEM combines geographically and sensor-diverse aerial/satellite RGB imagery. Sources discussed from the OEM paper include **xBD, Inria, SpaceNet, Open Cities AI, AIRS, HTCD, GeoNRW, Landcover.ai**, OpenAerialMap, and other geospatial sources. The benchmark examines:

- **Regional-level unsupervised domain adaptation (UDA):** 73 source regions and 24 target regions (paper's setting; this is **not the same** as the present supervised OEM→CEI adaptation).
- **Continent-wise UDA:** cross-continent source/target transfer.

The OEM benchmark's commonly discussed segmentation split is **3,000 train / 500 validation / 1,500 test** (total 5,000), but this is **not a CEI split**. OEM distributed data may omit some **xBD RGB imagery** that must be obtained from the original provider for exact replication. Verify actual images/checkpoints used in this project.

### What is actually shifting?

The main concern is **within-class appearance shift** and potentially class-frequency/context shift: Thailand's perennial Agriculture looks different from a subset of agricultural scenes represented in OEM. This does **not automatically imply different Agriculture definitions**. Potential contributors beyond canopy appearance include field arrangement, age of plantations, seasonality, imaging resolution, sensor characteristics, labeling consistency and class imbalance. These are **plausible factors, not all confirmed by the results**.

## 4. Established experimental controls

For a credible controlled comparison:

1. Keep **one fixed CEI validation set and one fixed CEI test set**, with neither used to update model weights.
2. Baseline and every fine-tuned model must be evaluated on the **same fixed CEI test set**, using identical preprocessing, masks, ignore rules and metric calculations.
3. Each fine-tuning run should **start from the exact same original best OEM checkpoint**.
4. Prefer nested CEI training subsets, subject to enough labeled **training-pool** images existing:
   `1 ⊂ 5 ⊂ 10 ⊂ 20 ⊂ 30 ⊂ 50 ⊂ 75 ⊂ 100`.
5. **100 training images requires at least 100 CEI training images in addition to validation/test images**. Do not claim to run CEI-100 if only 100 total annotated CEI images exist and some are held out.
6. Keep training setup constant when comparing crop sampling; **only sampling** should intentionally vary.
7. Preserve original checkpoints and log seeds, exact train/val/test filenames, class pixel fractions, run commands, epochs/steps, and checkpoints.
8. Splits should be spatially distinct where feasible (nearby or overlapping aerial tiles can leak information).
9. Repeated seeds are valuable for a tiny-data experiment; report mean ± standard deviation if done. No repeats have been reported yet.

## 5. Results supplied to date

All values are **fractions** (e.g., `0.5201` means **52.01% mIoU**), evaluated on what was reported as the **fixed CEI test set**. `cached`/`ok` are statuses in the user's evaluation output, not quality assessments.

### 5.1 Aggregate metrics

| Model | OA | mIoU | mF1 | Reported status |
| --- | ---: | ---: | ---: | --- |
| OEM baseline | 0.6381 | 0.4529 | 0.6055 | cached |
| CEI fine-tuned, **normal sampling** | 0.7661 | **0.5201** | **0.6406** | cached |
| CEI fine-tuned, **strong Agriculture-aware sampling** | 0.6417 | 0.4539 | 0.6039 | ok |

Normal fine-tuning vs OEM baseline: **+0.1280 OA** (+12.80 percentage points), **+0.0672 mIoU** (+6.72 points), **+0.0351 mF1** (+3.51 points).

Strong Agriculture-aware vs normal fine-tuning: **−0.1244 OA**, **−0.0662 mIoU**, **−0.0367 mF1**.

Strong Agriculture-aware vs OEM baseline: **+0.0010 mIoU** (0.10 percentage points), despite better Agriculture IoU.

### 5.2 Per-class IoU

| Model | Rangeland | Agriculture | Tree | Water | Building | Road | Non-vegetated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OEM baseline | 0.300 | 0.303 | 0.608 | 0.403 | 0.679 | 0.625 | 0.252 |
| Normal fine-tuning | 0.324 | **0.124** | **0.875** | **0.754** | 0.660 | 0.663 | 0.242 |
| Strong Agriculture-aware | 0.330 | **0.314** | 0.609 | 0.404 | 0.656 | **0.670** | 0.195 |

### 5.3 Per-class F1

| Model | Rangeland | Agriculture | Tree | Water | Building | Road | Non-vegetated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OEM baseline | 0.462 | 0.465 | 0.756 | 0.575 | 0.809 | 0.769 | 0.402 |
| Normal fine-tuning | 0.489 | **0.220** | **0.933** | **0.860** | 0.795 | 0.797 | 0.389 |
| Strong Agriculture-aware | 0.496 | **0.478** | 0.757 | 0.575 | 0.792 | **0.802** | 0.326 |

### 5.4 True Agriculture pixel distribution from confusion matrix

These percentages are presumed **row-normalized confusion matrix percentages** for actual Agriculture. That means “Correctly Agriculture” here is **recall** for Agriculture, **not precision**. Verify row orientation and evaluation code.

| Model | Agriculture IoU | True Agriculture → Agriculture | → Rangeland | → Tree |
| --- | ---: | ---: | ---: | ---: |
| OEM baseline | 0.303 | 38.6% | 44.5% | 13.4% |
| Normal fine-tuning | **0.124** | **90.5%** | **1.6%** | **6.9%** |
| Strong Agriculture-aware | **0.314** | **77.9%** | **9.2%** | **12.6%** |

**Important metric distinction:** Normal fine-tuning has **very high Agriculture recall (90.5%) but low Agriculture IoU (0.124)**. This is compatible with **very low Agriculture precision**, meaning many non-Agriculture pixels are predicted as Agriculture. Do not interpret low IoU as “the model fails to recognize most true Agriculture pixels.” Inspect columns of the confusion matrix and per-class precision to confirm.

The OEM baseline confuses true Agriculture mostly with Rangeland and Tree. The normal-fine-tuned model appears to swing toward widespread Agriculture predictions; this inference should be verified with a prediction-frequency breakdown and pixel-level confusion matrix. Strong Agriculture-aware sampling improves Agriculture IoU but appears to erase almost all the normal-tuning gains for Tree and Water.

**Potential concern to audit:** These results show an unusual combination of Agriculture recall changes and substantial class-wise swings. Confirm that all three confusion matrices and metric files correspond to the correct checkpoint, dataset split, label order, masks, and evaluation run. Use raw confusion-matrix counts for precision/recall/IoU reconstruction.

## 6. Experiment configurations and observations

### Experiment A: OEM baseline (`CEI-0`)

- Evaluate OEM best 7-class checkpoint on held-out CEI images without any CEI weight updates.
- mIoU `0.4529`, Agriculture IoU `0.303`.

### Experiment B: Normal CEI fine-tuning

- Existing conventional crop sampling and fine-tuning on labeled CEI images; exact number of training images **not supplied in this conversation**.
- mIoU `0.5201`; substantial gains Tree/Water, decline Agriculture IoU `0.124`.
- Agriculture recall reported as ~`90.5%` despite low IoU; check overprediction/precision.

### Experiment C: Strong Agriculture-aware CEI fine-tuning

Intended sampling mixture:

- **50%** normal random 512×512 crops.
- **30%** Agriculture-focused crops, preferentially containing **≥20% Agriculture among valid pixels**.
- **20%** difficult/minority-class crops.
- Bounded retries, fallback to normal crop, moderate augmentation, matched image/mask transforms.

Observed:

- Agriculture IoU rose to `0.314`, but global mIoU fell to `0.4539`.
- Tree IoU `0.609`, Water `0.404` (both far below normal fine-tuning).
- **The intended sampling configuration must be verified in the actual experiment config/logs**; do not equate a suggested setting with confirmed executed parameters unless repository confirms.

### Experiment D: **Next planned** mild Agriculture-aware sampling (not yet run/reported)

**Minimal-code-change request:** Keep all conditions in the existing fine-tuning and evaluation system identical except for sampler probabilities / threshold.

- **70%** ordinary random crops.
- **15%** Agriculture-focused crops.
- **15%** crops for other difficult/minority classes.
- Agriculture-focused acceptance: **≥10–15% Agriculture among valid (non-ignore) pixels**; choose and log an exact threshold before the run; suggested initial setting `0.10` or `0.15`.
- Bounded attempts and documented fallback to random crops.
- Same `512×512` image/mask spatial crop; nearest-neighbor mask transformations.
- Same OEM initialization, train/val/test files, model, 7-class mapping, ignore index, normalization, loss, optimizer, learning rate(s), scheduling, augmentation, seed, early stopping and metric computation.
- Do **not** introduce weighted CE, focal loss, architecture changes, or extra data in this controlled sampler comparison.
- Evaluate model D on precisely the fixed CEI test set against A/B/C.

**This run's desired outcome is a testable goal, not a predicted or guaranteed result:** Recover much of the `0.5201` mIoU, Tree `0.875`, Water `0.754` of normal fine-tuning while lifting Agriculture IoU above `0.124`, ideally retaining useful Road performance. The model may fail; report faithfully.

## 7. What the next AI / Claude Code should implement

1. **Inspect the actual repository before changing it.** Locate sampler implementation, source checkpoint path/loader, fine-tuning training loop, CEI image/mask loader, seven-class remapper, configuration files, split manifests, evaluation script, existing experiment outputs and logs.
2. Check **which CEI image subset size** produced the reported normal and strong sampling results; it is **unknown from these chat results**.
3. Verify the strong sampler was truly applied at requested rates and threshold; collect actual sample distribution per run (requested mode counts, successful agriculture crops, fallbacks, class pixel frequency).
4. Implement the **mild 70/15/15 sampler** as a separate named config / run. Preserve all other settings; start from original OEM checkpoint, not strong/normal fine-tuned checkpoint.
5. Fix and log random seed; ensure split manifests are identical across experiments and test isn't used for checkpoint selection.
6. Confirm class ID alignment, ignored pixels, shape matching, spatial independence, class coverage and Agriculture crop logic. For any remote sensing image/mask resizing, masks use **nearest-neighbor interpolation**.
7. Use existing evaluation code to produce **OA, mIoU, mF1, per-class IoU/F1/precision/recall, raw and normalized confusion matrices, and predicted class frequencies**.
8. Include explicit **Agriculture false-positive sources**: of predicted Agriculture pixels, what proportions come from true Tree, Rangeland, etc.? Also include recall confusion of true Agriculture into Rangeland/Tree.
9. Plot comparable fixed CEI test examples: RGB, CEI truth, OEM baseline prediction, normal CEI prediction, strong Agriculture-aware prediction, mild Agriculture-aware prediction.
10. Save model checkpoints, config snapshots, split manifests, logs, metrics, and figures to **new run directories**; do not overwrite existing model files or results.
11. Show exact commands to execute new run and comparison report. Do not claim an experiment executed until logs/results actually exist.

## 8. Metrics interpretation and methodological cautions

- **Recall for class A** = true A predicted A / all true A pixels. Row-normalized Agriculture diagonal `90.5%` indicates recall, not IoU.
- **Precision for class A** = true A predicted A / all pixels predicted A. Low precision can cause Agriculture IoU to be small despite 90.5% recall.
- **IoU for class A** = TP / (TP + FP + FN).
- **F1** = 2TP / (2TP + FP + FN).
- **mIoU** generally means mean over 7 classes; audit handling of classes absent from test, ignore pixels, and metric aggregation to ensure like-for-like comparison.
- Do not select a model solely by Agriculture recall or by overall accuracy. Compare **mIoU + per-class metrics + Agriculture precision/recall + qualitative maps**.
- Agriculture-focused sampling does not introduce previously unseen orchard types; make sure real labeled training images cover crop species, plantation maturity, season, and surroundings.
- Because datasets are small, class-frequency changes and chance seed effects may dominate. If possible, run several repeat seeds for confidence intervals and preserve spatially independent testing.
- Current results demonstrate **an overall gain from normal fine-tuning** and **an Agriculture-specific gain but lower overall accuracy from strong class-aware sampling**. They do **not yet demonstrate** that mild sampling succeeds, or quantify the number of labels required for “full” domain adaptation.

## 9. Data scaling study (separate from sampler ablation)

There was a separate plan for **1, 5, 10, 20, 30, 50, 75, 100** training images. Do not conflate that with strong vs mild sampler ablations.

- Hold validation and test constant and outside the training pool.
- Select **nested** subsets from local labeled training pool; aim for diverse annual/perennial crops (rubber, mango, longan, others), Tree, Water, Road, Rangeland, Building, Non-vegetated, landscapes, seasons and image sources.
- Start **each independent subset run from the original OEM checkpoint**.
- Compare OA, mIoU, mF1 and all per-class metrics on **identical CEI test images**.
- Plot CEI performance against amount of labeled CEI data; any saturation point must be inferred from observed measurements, not imposed in advance.
- If full CEI dataset contains only 100 labeled images total, reserve fixed val/test first and reduce the maximum training subset accordingly (or label additional images).

## 10. Known unknowns / required repository verification

- Actual filesystem locations of OEM and CEI images/masks, best OEM checkpoint, saved normal/strong CEI checkpoints.
- CEI dataset size and exact number of labeled training examples for the three currently measured runs.
- Exact numeric mapping and ignore index used by the code (class reporting order is known but numeric class IDs are not verified).
- Fixed split manifests and spatial independence of tiles.
- True training settings used (LR, optimizer, frozen stages, epochs, augmentation, early stopping, number of crops per image).
- Whether the strong 50/30/20 mixture and ≥20% threshold were successfully applied, and the realized crop distribution.
- Exact confusion matrix counts and predicted Agriculture precision.
- Results from a **mild 70/15/15 run** and from the **1–100 images scaling experiment** have **not been provided**.

## 11. Ready-to-use Claude Code task

```text
Inspect my existing OEM→CEI semantic-segmentation repository before editing. I use U-Net EfficientNet-B4 and an existing 7-class OEM checkpoint (Bareland + Developed Space merged into Non-vegetated). Keep class mapping, splits, normalization, mask/ignore rules, loss, optimizer, training schedule, and evaluation unchanged.

Known CEI fixed-test results:
- OEM baseline: OA .6381, mIoU .4529, mF1 .6055, Agriculture IoU .303.
- Normal CEI fine-tuning: OA .7661, mIoU .5201, mF1 .6406, Agriculture IoU .124, Tree IoU .875, Water IoU .754.
- Strong Agriculture-aware fine-tuning: OA .6417, mIoU .4539, mF1 .6039, Agriculture IoU .314, Tree IoU .609, Water IoU .404.

CEI Agriculture includes perennial crops (rubber, mango, longan, etc.). Strong Agriculture sampling helped Agriculture IoU but severely reduced overall performance. Normal fine-tuning gave ~90.5% Agriculture recall but only .124 Agriculture IoU, so inspect predicted Agriculture precision and false positives.

Implement a NEW controlled mild Agriculture-aware crop sampler experiment: 70% random 512x512 crops, 15% Agriculture-focused, 15% other difficult-class crops. Accept Agriculture crops with >=10-15% of VALID pixels as Agriculture (choose/document one exact value); bounded retry and fallback. Use same original OEM checkpoint, same fixed CEI train/validation/test split, same seed and all existing hyperparameters. Do not change class weights or introduce focal loss. Save to new output path.

Evaluate all 4 models (OEM baseline, normal, strong, mild) on the exact same CEI test set. Report OA, mIoU, mF1, per-class IoU/F1/precision/recall, raw + normalized confusion matrix, Agriculture false positives, Tree/Water performance, and sample visualizations. Check file/label/checkpoint alignment. Log the actual sampling composition and exact run commands. Do not overwrite existing checkpoints or claim unexecuted results.
```

---

**Current next action:** Run and measure Experiment D (mild sampler), then decide based on verified Agriculture **precision and recall**, per-class IoU and mIoU whether another adjustment is justified. Keep the separate 1–100 label-count study methodologically distinct.
