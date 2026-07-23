# Code Walkthrough

A line-by-line explanation of every Python file in the training pipeline, in the
order the code actually runs. For *how to run* the commands, see
[GUIDELINE.md](../GUIDELINE.md); for architecture, see [README.md](../README.md).

Reading order follows one command:

```powershell
python train.py --config configs/unet/cei/unet_effb4_oem2cei.yml
```

| # | File | Role |
| --- | --- | --- |
| 1 | `train.py` | the conductor: sets everything up, runs the epoch loop |
| 2 | `src/utils/config.py` | reads the YAML file |
| 3 | `src/datasets/dataset_factory.py` | picks which dataset class to use |
| 4 | `src/datasets/taxonomy.py` | the class definitions and label mappings |
| 5 | `src/datasets/transforms.py` | image normalization |
| 6 | `src/datasets/openearthmap_dataset.py` | loads and prepares one image+mask |
| 7 | `src/models/model_factory.py` | builds the neural network |
| 8 | `src/losses/loss_factory.py` | builds the loss function |
| 9 | `src/engine/trainer.py` | one epoch of learning |
| 10 | `src/engine/validator.py` | one epoch of scoring |
| 11 | `src/metrics/segmentation_metrics.py` | computes OA / mIoU / mF1 |
| 12 | `src/models/checkpoint.py` | loads saved weights |

---

## 1. `train.py`

The entry point. Everything else is called from here.

### `create_output_dir(output_dir)`

```python
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
```

Creates `experiments/<name>/`, plus `checkpoints/` and `logs/` inside it.
`exist_ok=True` means "do nothing if it already exists" instead of crashing, so
re-running a training run is safe.

### `select_device()`

```python
if torch.cuda.is_available():   return torch.device("cuda")   # NVIDIA GPU
if ...mps.is_available():       return torch.device("mps")    # Apple Silicon
return torch.device("cpu")
```

Picks the fastest available hardware, in order. On your machine this returns
`cuda`. A "device" is just *where the numbers live* — GPU memory or system RAM.
Tensors must be on the same device to interact, which is why you keep seeing
`.to(device)` later.

### `validate_config(config)`

```python
for section in ["experiment", "dataset", "model", "training"]:
    if section not in config:
        raise KeyError(f"Missing config section: {section}")
```

Fails fast if the YAML is missing a whole section.

```python
if dataset_config["num_classes"] != model_config["num_classes"]:
    raise ValueError(...)
```

**The most important check in the file.** If the dataset produces 7 classes but
the model has 8 output channels, training would either crash deep inside the
loss with a confusing message, or silently learn nonsense. This catches it in
the first second.

```python
weight_decay = training_config.get("weight_decay", 0.0)
if weight_decay is None:
    training_config["weight_decay"] = 0.0
elif isinstance(weight_decay, str):
    raise TypeError(...)
```

YAML quirk guard. Writing `weight_decay:` with no value gives Python `None`, and
writing `weight_decay: "1e-6"` (quoted) gives a *string* — which would crash the
optimizer. This normalizes the first case and rejects the second with a clear
message.

### `build_optimizer(model, training_config)`

```python
optimizer_name = training_config["optimizer"].lower()
learning_rate  = float(training_config["learning_rate"])
weight_decay   = float(training_config.get("weight_decay", 0.0) or 0.0)
```

`.lower()` so `AdamW`, `adamw`, `ADAMW` all work. `float(...)` because YAML can
hand back a string. The `or 0.0` turns a leftover `None` into `0.0`.

```python
if optimizer_name == "adam":  return torch.optim.Adam(...)
if optimizer_name == "adamw": return torch.optim.AdamW(...)
raise ValueError(...)
```

The **optimizer** is what actually changes the model's weights. AdamW applies
weight decay (a pull toward zero that reduces overfitting) correctly separated
from the gradient, which is why it is preferred over Adam.

### `build_scheduler(optimizer, training_config, epochs)`

```python
name = training_config.get("scheduler")
if name in (None, "none", "constant"):
    return None
if name == "cosine":
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
```

A **scheduler** changes the learning rate over time. `cosine` starts at your
configured LR and smoothly decreases to ~0 by the last epoch, so late epochs make
small refining steps instead of bouncing around. Your config uses `none`, so the
LR stays fixed the whole run.

### `save_checkpoint(...)`

```python
os.makedirs(os.path.dirname(path), exist_ok=True)
```

Re-creates the folder before *every* save. This looks redundant, but on
OneDrive/antivirus-scanned drives a folder can be moved out from under a
long-running job — so this guards each write instead of trusting the mkdir from
hours earlier.

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),      # the learned weights
    "optimizer_state_dict": optimizer.state_dict(),
    "metrics": metrics,
    "config": config,
    "best_miou": best_miou,
}
torch.save(checkpoint, path)
```

Saves weights **plus** context. Storing the config means you can always tell what
produced a checkpoint; storing the optimizer state means training could be
resumed exactly.

### `main()` — startup

```python
config = load_config(args.config)
validate_config(config)
```

Read the YAML, then check it.

```python
if args.init_weights:
    config["model"]["encoder_weights"] = None
```

If you are warm-starting from your own weights, skip downloading the ImageNet
encoder — it would just be overwritten. (Not used in exp_1.)

```python
shutil.copy(args.config, os.path.join(output_dir, "config.yml"))
```

Copies the config into the experiment folder — your record of the exact recipe.

```python
train_dataset = build_dataset(config, split="train")
val_dataset   = build_dataset(config, split="val")
```

Builds the two datasets (2100 and 350 samples for exp_1).

```python
train_loader = DataLoader(train_dataset, batch_size=..., shuffle=True,
                          num_workers=..., pin_memory=device.type == "cuda")
```

A **DataLoader** groups samples into batches and loads them in background
workers. `shuffle=True` reorders every epoch so the model never sees the same
sequence twice. `pin_memory` speeds up CPU→GPU transfer.

```python
eval_batch_size = 1 if dataset_config.get("eval_mode", "full") == "full" \
    else training_config["batch_size"]
```

**Why validation uses batch size 1:** `eval_mode: full` keeps each image at its
original size. OpenEarthMap tiles are not all the same size, and tensors of
different sizes cannot be stacked into one batch — so they go one at a time.

```python
model = build_model(config).to(device)
criterion = build_loss(config).to(device)
optimizer = build_optimizer(model, training_config)
scheduler = build_scheduler(...)
```

`.to(device)` moves the model onto the GPU. `criterion.to(device)` matters when
using `class_weights` — the weight tensor must sit on the same device as the
model's output.

```python
if mixed_precision and device.type == "cuda":
    scaler = torch.amp.GradScaler("cuda")
```

Mixed precision (16-bit math) is ~2x faster and uses about half the memory. Tiny
gradients can vanish in 16-bit, so `GradScaler` multiplies the loss up before
`backward()` and divides back afterwards.

### `main()` — the epoch loop

```python
for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(...)
    val_result = validate_one_epoch(...)
```

Learn on the training set, then score on the validation set.

```python
current_lr = optimizer.param_groups[0]["lr"]
if scheduler is not None:
    scheduler.step()
```

Reads the LR used *this* epoch **before** stepping the scheduler, so the log
shows the rate that was actually applied.

```python
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(logs, f, indent=4)
```

Rewrites the whole history every epoch. If the run dies at epoch 137, you still
have all 137 epochs on disk.

```python
is_best = val_miou is not None and val_miou > best_miou
if is_best:
    best_miou = val_miou
```

**mIoU decides "best"** — not loss, not accuracy. The `is not None` guard matters
because metrics return `None` when a class is absent.

```python
save_checkpoint(..., path=last_checkpoint_path, ...)     # every epoch
if is_best:
    save_checkpoint(..., path=best_checkpoint_path, ...) # only on improvement
```

`last_checkpoint.pth` = where training currently is. `best_checkpoint.pth` = the
best-scoring epoch, and the one you evaluate on CEI.

---

## 2. `src/utils/config.py`

```python
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
```

The whole file. Turns the YAML text into a Python dictionary, so
`config["dataset"]["num_classes"]` gives `7`.

`safe_load` (not `load`) refuses to execute arbitrary Python embedded in the
YAML — the safe default.

---

## 3. `src/datasets/dataset_factory.py`

```python
if dataset_name in ("OpenEarthMap", "CEI"):
    return OpenEarthMapDataset(config, split=split)
elif dataset_name == "IRSA_Map":
    raise NotImplementedError(...)
```

A **factory**: converts a *name in the config* into a *real object*. This is why
switching datasets needs no code change, only a different YAML.

`OpenEarthMap` and `CEI` share one class because they share one on-disk layout
(`images/` + `masks/`, `.tif`). Only their label encoding differs, and that is
handled by `label_map`.

---

## 4. `src/datasets/taxonomy.py`

The rulebook: what the CEI classes are, and how OEM labels convert into them.

```python
CEI_CLASS_NAMES = ["Rangeland", "Agriculture", "Tree", "Water",
                   "Building", "Road", "Non-vegetated"]
```

Index `i` here equals CEI on-disk id `i + 1`. That is why converting a prediction
into a CEI label file is just `+ 1`.

```python
CEI_CLASS_COLORS = [(0,255,36), (75,181,73), (34,97,38), (0,69,255),
                    (222,31,7), (255,255,255), (128,0,0)]
```

Used for drawing masks and for the GIMP palette.

```python
OEM_TO_CEI = [6, 0, 6, 5, 2, 3, 1, 4]
```

Indexed by OEM training index (0-7), gives the CEI index (0-6). Position `[0]`
(Bareland) and position `[2]` (Developed space) **both give 6** — that is the
merge into Non-vegetated, expressed as data rather than `if` statements.

### `build_label_lut(label_map, ignore_index=255)`

```python
if label_map == "oem":
    pairs = {raw: raw - 1 for raw in range(1, 9)}          # 1-8 -> 0-7
elif label_map == "oem_to_cei":
    pairs = {raw: OEM_TO_CEI[raw - 1] for raw in range(1, 9)}
elif label_map == "cei":
    pairs = {raw: raw - 1 for raw in range(1, 8)}          # 1-7 -> 0-6
```

Three translation schemes. Note `cei` is just "subtract 1" because CEI files are
already stored in CEI ids.

```python
lut = np.full(256, ignore_index, dtype=np.uint8)
for raw_value, internal_index in pairs.items():
    lut[raw_value] = internal_index
```

Builds a 256-slot **lookup table**. Every possible pixel value 0-255 gets an
answer. Values not in `pairs` — including `0` (unlabeled) — stay `ignore_index`.

This is the trick that makes conversion fast: instead of testing every pixel,
`lut[mask]` translates a whole million-pixel image in one vectorized step, and
masks out ignore pixels at the same time.

```python
allowed_raw = set(pairs) | {0}
num_classes = len(set(pairs.values()))
```

`allowed_raw` lets the loader detect a corrupt or mismatched mask.
`num_classes` counts *distinct outputs* — for `oem_to_cei` that is **7**, not 8,
because two OEM classes collapse into one.

---

## 5. `src/datasets/transforms.py`

```python
NORMALIZATIONS = {
    "imagenet": dict(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), max_pixel_value=255.0),
    "zero_one": dict(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
}
```

**Normalization** rescales pixel values so the network sees numbers centered near
zero. `imagenet` does `(x/255 - mean) / std`; `zero_one` only does `x/255`.

This must match how the weights were trained. Using ImageNet statistics with
weights that expect `zero_one` produces garbage predictions — that is the entire
reason this file exists.

```python
def build_normalize(config):
    name = get_normalization_name(config)
    if name not in NORMALIZATIONS:
        raise ValueError(...)
    return A.Normalize(**NORMALIZATIONS[name])
```

Returns the Albumentations transform selected by `dataset.normalization`.

---

## 6. `src/datasets/openearthmap_dataset.py`

Loads one image+mask pair. PyTorch requires two methods: `__len__` and
`__getitem__`.

### `__init__`

```python
self.root      = dataset_config["root"]
self.image_dir = dataset_config["image_dir"]
self.mask_dir  = dataset_config["mask_dir"]
self.mask_suffix = dataset_config.get("mask_suffix", "")
```

Paths. `mask_suffix` supports datasets that name masks after the image plus a
tag (e.g. `tile_1.tif` -> `tile_1_label.tif`); empty for both OEM and CEI,
where mask and image share the same filename.

```python
self.label_map = dataset_config.get("label_map", "oem")
self._label_lut, self._allowed_raw, self._num_label_classes = build_label_lut(
    self.label_map, ignore_index=self.ignore_index)
```

Builds the lookup table **once**, not per image.

```python
split_file = dataset_config[f"{split}_split"]
```

`split="train"` reads the `train_split` key. One class serves train/val/test.

### `_resolve_region_path(pattern, filename, suffix="")`

```python
region = os.path.splitext(filename)[0].rsplit("_", 1)[0]
```

`santa_rosa_10.tif` -> `santa_rosa`. `rsplit("_", 1)` splits on the *last*
underscore only, so multi-word regions survive.

```python
folder = pattern.replace("<region>", region)
```

`<region>/images` becomes `santa_rosa/images`. A pattern with no `<region>` (like
CEI's flat `images`) passes through unchanged — which is why one function handles
both layouts.

```python
if suffix:
    stem, extension = os.path.splitext(filename)
    filename = f"{stem}{suffix}{extension}"
```

Inserts the suffix (when one is configured) **before** the extension.

### `_convert_mask(mask, mask_path)`

```python
unexpected = sorted(set(np.unique(mask).tolist()) - self._allowed_raw)
if unexpected:
    raise ValueError(...)
```

Catches a wrong `label_map` loudly. Loading an 8-class OEM mask with the 7-class
`cei` map stops here instead of silently deleting all the value-8 pixels.

```python
return self._label_lut[mask]
```

The whole conversion. Fancy indexing: every pixel is replaced by its table entry.

### `_build_transform(split)`

```python
def pad_to(min_h, min_w, div=None):
    return A.PadIfNeeded(..., border_mode=cv2.BORDER_CONSTANT,
                         value=0, mask_value=self.ignore_index)
```

Padding must differ for image and mask: the image is padded with black, the mask
with `ignore_index` so **invented border pixels never affect loss or metrics**.

```python
if split == "train":
    transforms = [pad_to(self.crop_size, self.crop_size),
                  A.RandomCrop(height=self.crop_size, width=self.crop_size)]
```

Pad first (some OEM tiles are only 406px, smaller than the 512 crop), then take a
random 512x512 window. Random cropping is itself augmentation — a different piece
each epoch.

```python
    if dataset_config.get("augment", True):
        transforms += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                       A.RandomRotate90(p=0.5)]
```

Optional extra augmentation. Your config sets `augment: false` to match the
OpenEarthMap paper recipe.

> Albumentations applies geometric operations to image and mask **together**, so
> a flipped image always keeps its matching flipped mask.

```python
eval_mode = dataset_config.get("eval_mode", "full")
if eval_mode == "full":
    return A.Compose([pad_to(None, None, div=32), normalize, ToTensorV2()])
```

Validation is deterministic — no randomness. `div=32` pads to a multiple of 32
because the encoder halves the resolution 5 times (2^5 = 32); a size not divisible
by 32 would misalign. Padded pixels are `ignore_index`, so they are excluded from
scoring.

### `__getitem__(index)`

```python
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

OpenCV reads **BGR**; pretrained models expect **RGB**. Skipping this conversion
silently degrades accuracy — the model sees red where blue should be.

```python
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
if mask.ndim == 3:
    mask = mask[:, :, 0]
```

`IMREAD_UNCHANGED` preserves the raw class numbers (`IMREAD_COLOR` would convert
them to a 3-channel image and destroy them). If the mask was saved with 3
identical channels, take the first.

```python
transformed = self.transform(image=image, mask=mask)
image = transformed["image"]
mask  = transformed["mask"].long()
```

Image and mask go through **together** to stay aligned. `.long()` because
CrossEntropyLoss requires 64-bit integer targets.

---

## 7. `src/models/model_factory.py`

```python
model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                 in_channels=in_channels, classes=head_classes,
                 decoder_attention_type=decoder_attention_type)
```

Builds a **U-Net**: an encoder compresses the image into features, a decoder
expands it back to full resolution, with skip connections carrying fine detail
across. `classes=7` sets the number of output channels — one score map per class.

`encoder_weights="imagenet"` starts the encoder from weights trained on millions
of photos, so it already knows edges and textures.

### `LeadingChannelDrop`

```python
def forward(self, x):
    logits = self.inner(x)
    return logits[:, self.drop:, :, :]
```

Only for the external OpenEarthMap-SAR weights, which predict 9 classes where
index 0 is "background". This slices off the leading channel(s) so the rest line
up with this project's class order. Not used in exp_1.

---

## 8. `src/losses/loss_factory.py`

The **loss** measures how wrong the model is — the number that gets minimized.

### `_build_class_weights(config)`

```python
tensor = torch.tensor([float(w) for w in weights], dtype=torch.float32)
expected = config["dataset"]["num_classes"]
if tensor.numel() != expected:
    raise ValueError(...)
```

Optional per-class importance, for when rare classes are being ignored by the
model. The length check must equal `num_classes` — **7** in your CEI configs.

### `build_loss(config)`

```python
if loss_name == "cross_entropy":
    return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
```

Your config uses this. Cross-entropy asks, per pixel: "how much probability did
you assign to the correct class?" `ignore_index=255` makes it **skip** unlabeled
pixels entirely — they contribute nothing.

Other options: `dice` optimizes region overlap (helps small classes),
`ce_dice` adds both, `focal` down-weights easy pixels to focus on hard ones.

---

## 9. `src/engine/trainer.py`

```python
model.train()
```

Switches to training mode (affects dropout/batch-norm behavior).

```python
for images, masks in progress_bar:
    images = images.to(device)
    masks  = masks.to(device).long()
```

Moves the batch to the GPU.

```python
    optimizer.zero_grad()
```

**Essential.** PyTorch *accumulates* gradients by default; without clearing them,
each step would include every previous step's gradients.

```python
    if use_amp:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

The mixed-precision path:
1. `autocast` runs the forward pass in 16-bit where it is safe.
2. `scaler.scale(loss)` multiplies the loss up so small gradients survive 16-bit.
3. `scaler.step` unscales and applies the update (skipping it if numbers overflowed).
4. `scaler.update` adjusts the scale factor for next time.

```python
    else:
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
```

The plain path. `backward()` computes gradients; `step()` applies them.

```python
    total_loss += loss.item()
```

`.item()` extracts a plain Python number — keeping the tensor would hold onto the
whole computation graph and leak memory.

```python
return total_loss / max(num_batches, 1)
```

Average loss. `max(..., 1)` avoids divide-by-zero on an empty loader.

---

## 10. `src/engine/validator.py`

```python
model.eval()
```

Evaluation mode — the counterpart to `model.train()`.

```python
with torch.no_grad():
```

Disables gradient tracking: faster and much less memory, because we are only
measuring, not learning.

```python
        outputs = model(images)
        loss = criterion(outputs, masks)
        preds = torch.argmax(outputs, dim=1)
```

`outputs` is `[B, 7, H, W]` — seven scores per pixel. `argmax(dim=1)` picks the
index of the highest score, collapsing it to `[B, H, W]`: the predicted class.

```python
        metrics.update(preds, masks)
```

Accumulates into the confusion matrix.

```python
result = metrics.compute()
result["loss"] = avg_loss
return result
```

Metrics are computed **once at the end**, over all images together — not averaged
per batch, which would distort the result.

---

## 11. `src/metrics/segmentation_metrics.py`

```python
self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
```

A 7x7 table. Row = true class, column = predicted class. Cell `[i][j]` counts
pixels that were really class `i` but predicted class `j`. The diagonal is
correct predictions; everything else is a specific type of mistake.

### `update(preds, targets)`

```python
valid_mask = targets != self.ignore_index
preds = preds[valid_mask]; targets = targets[valid_mask]
```

Drops ignored pixels so they never affect the score.

```python
indices = self.num_classes * targets + preds
cm = np.bincount(indices, minlength=self.num_classes ** 2).reshape(...)
```

The clever part. Each (true, predicted) pair is encoded as a single number
`7*true + pred`, unique for every combination. `bincount` counts all of them at
once and `reshape` folds the result into the 7x7 table — no Python loop.

### `compute()`

```python
true_positive  = np.diag(cm)                        # correct
false_positive = cm.sum(axis=0) - true_positive     # predicted X, wasn't X
false_negative = cm.sum(axis=1) - true_positive     # was X, missed it
```

```python
overall_accuracy = total_correct / total_pixels
```

**OA** — the fraction of pixels correct. Can look good while rare classes fail
completely, which is why it is not used to pick the best checkpoint.

```python
iou[present] = true_positive[present] / (tp + fp + fn)[present]
```

**IoU** (Intersection over Union) per class: overlap divided by combined area.
Harsher than accuracy — it punishes both misses and false alarms.

```python
f1[present] = (2 * true_positive[present]) / (2*tp + fp + fn)[present]
```

**F1** — the balance of precision and recall. Always slightly higher than IoU.

```python
miou = np.nanmean(iou)
```

**mIoU** — the headline number: the average of per-class IoU. Every class counts
equally, so a model that ignores Buildings cannot hide behind lots of correct
Trees. `nanmean` skips classes that were absent.

```python
"class_support": cm.sum(axis=1).tolist(),
```

How many true pixels each class had. Essential context: an IoU computed from 200
pixels is noise, not a measurement.

---

## 12. `src/models/checkpoint.py`

```python
def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint
```

Handles two file formats: this project's checkpoints (a dict with extra info) and
a bare `state_dict` (how external weights are usually distributed).

```python
try:
    model.load_state_dict(state_dict, strict=True)
except RuntimeError:
    inner = getattr(model, "inner", None)
    if inner is None: raise
    inner.load_state_dict(state_dict, strict=True)
```

Tries the wrapper first, then the wrapped model. This is what lets both plain and
`LeadingChannelDrop`-wrapped models load without special-casing at the call site.
`strict=True` means every weight must match — no silent partial loads.

---

## The whole flow in one picture

```
config.yml
   │  load_config → validate_config
   ▼
build_dataset ──► OpenEarthMapDataset
   │                 read .tif → label_map (taxonomy LUT) → crop → normalize → tensor
   ▼
DataLoader ──────► batch: images [8,3,512,512]  masks [8,512,512]
   ▼
build_model ─────► U-Net(EfficientNet-B4) → outputs [8,7,512,512]
   ▼
build_loss ──────► CrossEntropy(outputs, masks), ignoring 255
   ▼
backward() + optimizer.step()          ← the model learns
   ▼
[end of epoch] validate → argmax → confusion matrix → OA / mIoU / mF1
   ▼
save last_checkpoint.pth, and best_checkpoint.pth when mIoU improves
```
