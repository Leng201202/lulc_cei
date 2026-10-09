# How This Project Works / โปรเจกต์นี้ทำงานอย่างไร

A bilingual (English / ไทย) explanation of the `lulc_cei` semantic-segmentation
pipeline — what it does, how the pieces fit together, and how to run it.

เอกสารสองภาษา (อังกฤษ / ไทย) อธิบายว่าไปป์ไลน์ semantic segmentation ชื่อ
`lulc_cei` ทำอะไร แต่ละส่วนประกอบกันอย่างไร และรันอย่างไร

> Related docs / เอกสารที่เกี่ยวข้อง: [README.md](README.md) (overview),
> [GUIDELINE.md](GUIDELINE.md) (CLI reference), [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md)
> (line-by-line walkthrough), [configs/cei_oem/Script.md](configs/cei_oem/Script.md) and
> [configs/cei_irsa/Script.md](configs/cei_irsa/Script.md) (the two comparison experiments).

---

## 1. What the project is / โปรเจกต์นี้คืออะไร

**EN** — This is a deep-learning pipeline for **land-use / land-cover (LULC) mapping**:
given an aerial/satellite image tile, the model assigns every single pixel to one of
7 land-cover classes. That task is called *semantic segmentation*.

The target label scheme is called **CEI** — a 7-class taxonomy. The problem is that
the CEI study area (Mae Suai, tiles named `maesuai_1..N`) has **no ground-truth labels**
of its own at scale. So the project:

1. trains models on two large **public benchmark datasets** — OpenEarthMap (OEM) and
   IRSAMap — whose labels are **remapped** into the CEI 7-class scheme,
2. tests those models on a small **hand-labeled CEI test set** (100 tiles),
3. compares 5 architectures × 2 source datasets to answer:
   *"which architecture and which source dataset generalize best to CEI imagery?"*

**TH** — โปรเจกต์นี้คือไปป์ไลน์ deep learning สำหรับ **การทำแผนที่การใช้ประโยชน์ที่ดิน/สิ่งปกคลุมดิน (LULC)**
กล่าวคือ เมื่อป้อนภาพถ่ายทางอากาศ/ดาวเทียมเข้าไป โมเดลจะจำแนก **ทุก ๆ พิกเซล** ว่าเป็น
สิ่งปกคลุมดินชนิดใดใน 7 ชนิด งานลักษณะนี้เรียกว่า *semantic segmentation*

ระบบ label เป้าหมายเรียกว่า **CEI** ซึ่งมี 7 คลาส ปัญหาคือพื้นที่ศึกษาของ CEI (แม่สรวย,
ไทล์ชื่อ `maesuai_1..N`) **ไม่มีข้อมูล ground truth** ในปริมาณมากพอ ดังนั้นโปรเจกต์จึง:

1. เทรนโมเดลบน **ชุดข้อมูลสาธารณะขนาดใหญ่ 2 ชุด** คือ OpenEarthMap (OEM) และ IRSAMap
   โดย **แปลง (remap)** label ของทั้งสองชุดให้เข้ากับระบบ 7 คลาสของ CEI
2. ทดสอบโมเดลเหล่านั้นกับ **ชุดทดสอบ CEI ที่ label ด้วยมือ** (100 ไทล์)
3. เปรียบเทียบ 5 สถาปัตยกรรม × 2 ชุดข้อมูลต้นทาง เพื่อตอบคำถามว่า
   *"สถาปัตยกรรมใด และชุดข้อมูลต้นทางใด generalize ไปยังภาพ CEI ได้ดีที่สุด"*

---

## 2. The big picture / ภาพรวมการทำงาน

```
                    ┌─────────────────────────────────────────┐
                    │  ONE YAML CONFIG  (configs/**/*.yml)    │
                    │  experiment / dataset / model / training│
                    └───────────────────┬─────────────────────┘
                                        │  --config
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
   train.py                        evaluate.py                     predict.py
   (train + validate)              (score a checkpoint)            (inference only)
        │                               │                               │
        └───────────────┬───────────────┴───────────────┬───────────────┘
                        │                               │
             build_dataset(config, split)      build_model(config)
             dataset_factory.py                model_factory.py
                        │                               │
      OpenEarthMapDataset / IRSADataset        unet | deeplabv3 | upernet |
      read image + mask                        unetformer | ftunetformer |
      → remap labels (taxonomy.py LUT)         segformer
      → augment / normalize                            │
                        │                               │
                        └──────────► DataLoader ────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     │  train_one_epoch / validate_one_epoch│   src/engine/
                     │  build_loss(config)                  │   src/losses/
                     │  SegmentationMetrics (confusion mtx) │   src/metrics/
                     └──────────────────┬──────────────────┘
                                        │
                    experiments/<name>/checkpoints/{last,best}_checkpoint.pth
                    experiments/<name>/logs/training_logs.json
                    experiments/<name>/logs/<round>/cei_test_tta.json (+ confusion PNG)
```

**EN** — Everything is **config-driven**. One YAML file describes the experiment, the
dataset, the model and the training recipe. The three entry points (`train.py`,
`evaluate.py`, `predict.py`) all read the *same* config format, so a training config
and its matching test config differ only in which data they point at.

**TH** — ทุกอย่างถูก **ขับเคลื่อนด้วยไฟล์ config** ไฟล์ YAML หนึ่งไฟล์อธิบายทั้ง experiment,
dataset, model และสูตรการเทรน จุดเริ่มต้นทั้งสาม (`train.py`, `evaluate.py`, `predict.py`)
อ่าน config รูปแบบ *เดียวกัน* ดังนั้น config สำหรับเทรนกับ config สำหรับทดสอบจึงต่างกัน
แค่ว่าชี้ไปที่ข้อมูลชุดไหนเท่านั้น

---

## 3. The heart of the project: the taxonomy layer / หัวใจของโปรเจกต์: ชั้นแปลง label

**EN** — [src/datasets/taxonomy.py](src/datasets/taxonomy.py) is the single source of
truth for converting between three different label spaces. This is the most important
file to understand, because everything else depends on it being right.

**TH** — [src/datasets/taxonomy.py](src/datasets/taxonomy.py) คือแหล่งความจริงเพียงหนึ่งเดียว
สำหรับการแปลงระหว่างระบบ label 3 แบบ นี่คือไฟล์ที่สำคัญที่สุดที่ต้องเข้าใจ เพราะทุกส่วนที่เหลือ
ขึ้นอยู่กับความถูกต้องของไฟล์นี้

### 3.1 The 7 CEI classes / คลาส CEI ทั้ง 7

| Internal index<br>(ดัชนีภายใน) | On-disk id<br>(รหัสในไฟล์) | Class (EN) | คลาส (ไทย) | RGB |
| --- | --- | --- | --- | --- |
| 0 | 1 | Rangeland | ทุ่งหญ้า / พื้นที่หญ้า | `(0, 255, 36)` |
| 1 | 2 | Agriculture | พื้นที่เกษตรกรรม | `(75, 181, 73)` |
| 2 | 3 | Tree | ต้นไม้ / ป่าไม้ | `(34, 97, 38)` |
| 3 | 4 | Water | แหล่งน้ำ | `(0, 69, 255)` |
| 4 | 5 | Building | อาคาร | `(222, 31, 7)` |
| 5 | 6 | Road | ถนน | `(255, 255, 255)` |
| 6 | 7 | Non-vegetated | พื้นที่ไม่มีพืชปกคลุม (ดินเปล่า) | `(128, 0, 0)` |
| — | 0 | *Unlabeled → ignore* | *ไม่ระบุ → ข้าม* | `(0, 0, 0)` |

**Rule / กฎ:** `CEI internal index i` ⟷ `on-disk id i + 1`. Turning a model prediction
back into a CEI label file is just `+ 1`.
การแปลงผลทำนายของโมเดลกลับเป็นไฟล์ label ของ CEI ทำได้แค่ `+ 1`

### 3.2 The three label spaces / ระบบ label ทั้ง 3

| `label_map` | Source / ต้นทาง | Raw codes / รหัสดิบ | Result / ผลลัพธ์ |
| --- | --- | --- | --- |
| `oem` | OpenEarthMap native | `1–8` | indices `0–7` (8-class OEM training) |
| `oem_to_cei` | OEM imagery, CEI scheme | `1–8` | indices `0–6` (CEI) |
| `cei` | Hand-labeled CEI files | `1–7` | indices `0–6` |
| `irsa_to_cei` | IRSAMap two-digit codes | `0,10,11,12,21–24,31,32,34` | indices `0–6` |

**EN** — `build_label_lut(label_map, ignore_index)` builds a **256-entry uint8 lookup
table**, so a whole mask is remapped with one vectorized `lut[mask]` operation instead
of a Python loop. Any raw value not in the scheme becomes `ignore_index` (255).

Crucially, both dataset classes **validate raw mask values before converting**. If you
load 8-class OEM labels with `label_map: cei`, it raises a clear error instead of
silently dropping a class.

**TH** — `build_label_lut(label_map, ignore_index)` สร้าง **ตารางค้นหา (LUT) ขนาด 256 ช่อง
แบบ uint8** ทำให้แปลง mask ทั้งภาพได้ด้วยการทำ `lut[mask]` เพียงครั้งเดียว ไม่ต้องวน loop
ใน Python ค่าดิบใดที่ไม่อยู่ในระบบจะกลายเป็น `ignore_index` (255)

ที่สำคัญคือ dataset ทั้งสองคลาส **ตรวจสอบค่าดิบของ mask ก่อนแปลงเสมอ** ถ้าคุณโหลด label
OEM 8 คลาสด้วย `label_map: cei` ระบบจะโยน error ที่อ่านเข้าใจได้ทันที แทนที่จะทิ้งคลาสไปเงียบ ๆ

### 3.3 OEM → CEI mapping

```
OEM idx  OEM class            CEI idx  CEI class
  0      Bareland          →    6      Non-vegetated
  1      Rangeland         →    0      Rangeland
  2      Developed space   →    6      Non-vegetated
  3      Road              →    5      Road
  4      Tree              →    2      Tree
  5      Water             →    3      Water
  6      Agriculture land  →    1      Agriculture
  7      Building          →    4      Building
```

**EN** — "Developed space" and Bareland **both** map to Non-vegetated. Sending
Developed space to `ignore` instead was tried, on the theory that paved surfaces are
distinct from true bare ground. Three runs measured it on the CEI test set and all
three got *worse* at Non-vegetated — IoU roughly halved (U-Net 0.252 → 0.091, UPerNet
0.180 → 0.088). The class balance explains it: Bareland is 1.5% of OEM pixels and
Developed space 16%, while CEI's hand-labelled Non-vegetated is 13.6% of the test set,
so CEI's labellers do put paved and developed surfaces there. Excluding Developed space
removed most of that class's training signal.

**TH** — "Developed space" กับ Bareland ถูก map ไปเป็น Non-vegetated **ทั้งคู่** เคยลองส่ง
Developed space ไป `ignore` โดยคิดว่าพื้นผิวลาดปูนต่างจากดินเปล่าจริง แต่วัดผลบนชุดเทส CEI
สามรอบแล้ว **แย่ลงทุกรอบ** ในคลาสที่ตั้งใจจะปกป้อง — IoU ลดลงเกือบครึ่ง (U-Net 0.252 → 0.091,
UPerNet 0.180 → 0.088) เหตุผลอยู่ที่สัดส่วนคลาส: Bareland เป็นแค่ 1.5% ของพิกเซล OEM ส่วน
Developed space เป็น 16% ขณะที่ Non-vegetated ที่ label ด้วยมือในชุดเทส CEI มีถึง 13.6%
แปลว่าผู้ label CEI นับพื้นผิวลาดปูนเป็น Non-vegetated จริง การตัด Developed space ออกจึงเท่ากับ
ตัดสัญญาณการเรียนรู้ของคลาสนั้นไปเกือบหมด

### 3.4 IRSAMap → CEI mapping (and its two quirks / และข้อควรระวัง 2 ข้อ)

```
 0  background        →  6  Non-vegetated   ← ไม่ใช่ ignore!
10  cropland          →  1  Agriculture
11  forest            →  2  Tree
12  grass / sparse    →  0  Rangeland
21,22,23,24  water    →  3  Water
31  building          →  4  Building
32  road              →  5  Road
34  sport surfaces    →  –  IGNORE
```

**Quirk 1 — background is a real class / ข้อควรระวังที่ 1 — background คือคลาสจริง**

**EN** — Unlike OEM, IRSAMap's code `0` is **not** "unlabeled". IRSAMap annotates only
five thematic classes and leaves bare soil, concrete, parking lots and construction
ground as `0`. Measured over 300 tiles, background is 23.6% of all pixels, and 85.7% of
it is real ground. Sending it to ignore would leave Non-vegetated so under-represented
that the model would never predict it — so it maps to Non-vegetated instead.

**TH** — ต่างจาก OEM ตรงที่รหัส `0` ของ IRSAMap **ไม่ใช่** "ไม่ระบุ" IRSAMap ทำ annotation
เฉพาะ 5 คลาสหลัก แล้วปล่อยดินเปล่า คอนกรีต ลานจอดรถ และพื้นที่ก่อสร้างไว้เป็น `0` จากการวัดบน
300 ไทล์ พบว่า background คิดเป็น 23.6% ของพิกเซลทั้งหมด และ 85.7% ของนั้นคือพื้นดินจริง
ถ้าส่งไป ignore คลาส Non-vegetated จะมีตัวอย่างน้อยเกินไปจนโมเดลไม่เคยทำนายคลาสนี้เลย
จึงแม็ปเป็น Non-vegetated แทน

**Quirk 2 — `nodata_to_ignore` / ข้อควรระวังที่ 2 — `nodata_to_ignore`**

**EN** — The remaining ~14% of that background is **near-black nodata border padding**
from the source imagery — not ground at all. It carries the same mask value `0`, so the
only way to tell them apart is to look at the **image** pixels. `nodata_to_ignore: 8`
(a brightness threshold) tells `NodataMasker` to reclassify any pixel at or below that
brightness as ignore.

Without it, the model learns "black region → Non-vegetated" and then confidently
mislabels the blank CEI captures (`maesuai_1`, `maesuai_5`).

⚠️ This value **must match** between a training config and any test config sharing that
data, or class balance shifts between them.

**TH** — background ที่เหลืออีกประมาณ 14% คือ **ขอบภาพสีดำ (nodata padding)** จากภาพต้นฉบับ
ซึ่งไม่ใช่พื้นดินเลย แต่มีค่า mask เป็น `0` เหมือนกัน วิธีเดียวที่จะแยกออกคือดูที่พิกเซลของ **ภาพ**
ค่า `nodata_to_ignore: 8` (เกณฑ์ความสว่าง) สั่งให้ `NodataMasker` เปลี่ยนพิกเซลที่สว่างน้อยกว่า
หรือเท่ากับค่านี้ให้เป็น ignore

ถ้าไม่มีขั้นตอนนี้ โมเดลจะเรียนรู้ว่า "บริเวณสีดำ → Non-vegetated" แล้วทำนายภาพ CEI ที่ว่างเปล่า
(`maesuai_1`, `maesuai_5`) ผิดอย่างมั่นใจ

⚠️ ค่านี้ **ต้องตรงกัน** ระหว่าง config ที่ใช้เทรนกับ config ที่ใช้ทดสอบบนข้อมูลเดียวกัน
มิฉะนั้นสัดส่วนคลาสจะเปลี่ยนไป

**Quirk 3 — two mask folders / ข้อควรระวังที่ 3 — โฟลเดอร์ mask 2 ชุด**

IRSAMap ships `SegLabel_vwsbr` and `SegLabel_rvwsb`. They are **not copies** — they
encode a different road/building overlap priority (169 of 200 sampled tiles differ).
`mask_dir` picks one explicitly; `vwsbr` is the default.

IRSAMap มีโฟลเดอร์ `SegLabel_vwsbr` และ `SegLabel_rvwsb` ทั้งสอง **ไม่ใช่สำเนากัน** —
ลำดับการทับซ้อนระหว่างถนนกับอาคารต่างกัน (169 จาก 200 ไทล์ที่สุ่มมาต่างกัน)
`mask_dir` เป็นตัวเลือกว่าจะใช้ชุดไหน โดยค่าเริ่มต้นคือ `vwsbr`

---

## 4. Data loading / การโหลดข้อมูล

**EN** — [src/datasets/dataset_factory.py](src/datasets/dataset_factory.py) dispatches on
`dataset.name`:

| `dataset.name` | Loader | Notes |
| --- | --- | --- |
| `OpenEarthMap` | `OpenEarthMapDataset` | `<region>/images` + `<region>/labels`, `.tif` |
| `CEI` | `OpenEarthMapDataset` | same loader — flat `images/` + `masks/`, only `label_map` differs |
| `IRSA_Map` | `IRSADataset` | separate train/test trees, IRSA defaults injected |
| `OEM_CEI_Mix` | `build_mix_dataset` | concatenates several configured sources |
| `LoveDA` | — | unimplemented stub |

**TH** — [src/datasets/dataset_factory.py](src/datasets/dataset_factory.py) เลือก loader
จากค่า `dataset.name` ตามตารางด้านบน จุดที่น่าสนใจคือ `OpenEarthMap` และ `CEI` ใช้
loader **ตัวเดียวกัน** เพราะโครงสร้างโฟลเดอร์และการอ่านไฟล์เหมือนกัน ต่างกันแค่ `label_map`

### 4.1 What one sample looks like / ตัวอย่างข้อมูลหนึ่งชิ้น

`__getitem__` does, in order / ทำตามลำดับนี้:

1. Read image with OpenCV → convert **BGR → RGB** (pretrained models expect RGB).
   อ่านภาพด้วย OpenCV แล้วแปลง **BGR → RGB** (โมเดล pretrained ต้องการ RGB)
2. Read mask, take channel 0 if it is 3-channel.
   อ่าน mask และเลือก channel แรกถ้าเป็นภาพ 3 ช่อง
3. **Validate + remap** raw values through the LUT → class indices + ignore.
   **ตรวจสอบและแปลง** ค่าดิบผ่าน LUT → ดัชนีคลาส + ignore
4. Optionally apply `nodata_to_ignore` (IRSA only), using the image pixels.
   ใช้ `nodata_to_ignore` ถ้ากำหนดไว้ (เฉพาะ IRSA) โดยดูจากพิกเซลของภาพ
5. Run the albumentations pipeline — **image and mask together**, so every geometric
   transform stays pixel-aligned.
   รัน albumentations pipeline โดยส่ง **ภาพและ mask ไปพร้อมกัน** เพื่อให้การแปลงเชิงเรขาคณิต
   ตรงกันทุกพิกเซล
6. Return `(image_tensor, mask.long())` — cross-entropy needs `torch.long` targets.
   คืนค่า `(image_tensor, mask.long())` เพราะ cross-entropy ต้องการเป้าหมายชนิด `torch.long`

### 4.2 Transforms / การแปลงภาพ

| Split | Pipeline | คำอธิบาย |
| --- | --- | --- |
| `train` | pad to `crop_size` → `RandomCrop` → *(optional flip/rot90)* → normalize → tensor | เติมขอบให้ถึงขนาด crop แล้วสุ่มตัด (OEM ไทล์ไม่เท่ากัน บางไทล์เล็กแค่ 406px) |
| `val`/`test`, `eval_mode: full` | pad to a multiple of 32 → normalize → tensor | **โปรโตคอลของ paper** — ให้คะแนนทั้งภาพที่ความละเอียดจริง |
| `val`/`test`, `eval_mode: crop` | pad → `CenterCrop` → normalize → tensor | เร็วกว่า แต่ตัดขอบภาพทิ้ง |

**Padding rule / กฎการเติมขอบ:** the image is padded with **zeros**, the mask with
**`ignore_index`** — so invented border pixels never contribute to loss or metrics.
ภาพถูกเติมด้วย **ศูนย์** ส่วน mask ถูกเติมด้วย **`ignore_index`** ดังนั้นพิกเซลขอบที่สร้างขึ้นมา
จะไม่ถูกนับใน loss หรือ metric

**Augmentation is OFF by default** (`augment: false`) in the main configs — that matches
the OpenEarthMap paper recipe of *random cropping only*.
**Augmentation ถูกปิดโดยค่าเริ่มต้น** (`augment: false`) ใน config หลัก ซึ่งตรงกับสูตรของ
paper OpenEarthMap ที่ใช้ *การสุ่มตัดภาพอย่างเดียว*

### 4.3 Normalization — a trap / การ normalize — จุดที่พลาดง่าย

| `normalization` | Formula | Use with / ใช้กับ |
| --- | --- | --- |
| `imagenet` | `(x/255 − mean) / std` with ImageNet stats | every model trained in this project |
| `zero_one` | `x / 255` only | the external OpenEarthMap-SAR baseline weights |

⚠️ **EN** — This must match what the checkpoint's weights were trained with. Using the
wrong one produces **incoherent** predictions, not just slightly worse ones.

⚠️ **TH** — ค่านี้ต้องตรงกับตอนที่เทรน checkpoint นั้นมา ถ้าใช้ผิดจะได้ผลทำนายที่
**มั่วไปเลย** ไม่ใช่แค่แย่ลงเล็กน้อย

### 4.4 Class-aware sampling (opt-in) / การสุ่ม crop แบบเจาะจงคลาส (เปิดใช้เอง)

**EN** — When `dataset.class_aware_sampling` is set (training split only), the plain
`RandomCrop` is replaced by a crop **search**: roll a 50/30/20 split between a normal
random crop, a crop containing a target class, and a crop containing a minority class.
It retries up to `max_attempts` times until at least `min_class_fraction` of the crop's
valid pixels belong to the wanted class. Used to rescue rare classes such as Agriculture.

**TH** — เมื่อกำหนด `dataset.class_aware_sampling` (เฉพาะชุดเทรน) `RandomCrop` ธรรมดา
จะถูกแทนด้วยการ **ค้นหา crop**: สุ่มแบบ 50/30/20 ระหว่าง crop ปกติ, crop ที่มีคลาสเป้าหมาย
และ crop ที่มีคลาสส่วนน้อย โดยจะลองซ้ำสูงสุด `max_attempts` ครั้งจนกว่าจะได้ crop ที่มีพิกเซล
ของคลาสที่ต้องการอย่างน้อย `min_class_fraction` ใช้เพื่อกู้คลาสที่หายาก เช่น Agriculture

---

## 5. Models / โมเดล

**EN** — [src/models/model_factory.py](src/models/model_factory.py) exposes six
architectures behind one `build_model(config)` call, dispatching on `model.name`.

**TH** — [src/models/model_factory.py](src/models/model_factory.py) เปิดให้เรียกใช้
6 สถาปัตยกรรมผ่านฟังก์ชันเดียวคือ `build_model(config)` โดยเลือกจาก `model.name`

| `model.name` | Backbone / Encoder | Source | Params | Notes / หมายเหตุ |
| --- | --- | --- | --- | --- |
| `unet` | EfficientNet-B4 | `segmentation_models_pytorch` | 20.2M | CNN baseline |
| `deeplabv3` | any smp encoder | smp | — | available, not in the main comparison |
| `upernet` | Swin-B (`tu-` timm name) | smp | 96.9M | needs `encoder_params.img_size` |
| `unetformer` | ResNet-101 + GLTB decoder | custom, [unetformer.py](src/models/unetformer.py) | 43.2M | returns `(main, aux)` in training |
| `ftunetformer` | Swin-B (fixed) + GLTB decoder | custom, [ftunetformer.py](src/models/ftunetformer.py) | 96.0M | returns `(main, aux)` in training |
| `segformer` | MiT-B5 | smp | 82.0M | transformer, no size constraint |

### 5.1 Two things that trip people up / สองเรื่องที่มักทำให้สับสน

**Auxiliary heads / หัวเสริม**

**EN** — UNetFormer and FT-UNetFormer are *deeply supervised*: during training they return
a **tuple** `(main, aux)`. `src/engine/trainer.py:compute_loss()` is the one place that
branches on tuple-vs-tensor and adds `aux_weight × criterion(aux, masks)` (0.4, the
paper's value). Every other model returns a plain tensor and takes the simple branch.

**TH** — UNetFormer และ FT-UNetFormer ใช้ *deep supervision* คือขณะเทรนจะคืนค่าเป็น
**tuple** `(main, aux)` ฟังก์ชัน `src/engine/trainer.py:compute_loss()` เป็นที่เดียว
ที่ตรวจว่าเป็น tuple หรือ tensor แล้วบวก `aux_weight × criterion(aux, masks)` เข้าไป
(ค่า 0.4 ตาม paper) โมเดลอื่นคืน tensor ธรรมดาและใช้เส้นทางปกติ

**Swin needs a fixed input size / Swin ต้องระบุขนาด input**

**EN** — Swin backbones are built for 224×224 and raise *"Input height doesn't match
model"* at 512 unless `encoder_params.img_size` is overridden. The Swin models therefore
set `img_size: 512` for training and `img_size: 1024` for the CEI test configs (CEI tiles
are 1024×1024). The **weights themselves are resolution-independent**, so a 512-trained
checkpoint loads unchanged — but it is an extra moving part.

**TH** — Backbone ตระกูล Swin ถูกออกแบบมาสำหรับ 224×224 และจะโยน error
*"Input height doesn't match model"* ที่ขนาด 512 ถ้าไม่กำหนด `encoder_params.img_size`
โมเดล Swin จึงตั้ง `img_size: 512` ตอนเทรน และ `img_size: 1024` ใน config ทดสอบ CEI
(ไทล์ CEI ขนาด 1024×1024) **ตัว weight เองไม่ผูกกับความละเอียด** ดังนั้น checkpoint ที่เทรน
ที่ 512 โหลดได้ตามปกติ — แต่ก็เป็นอีกจุดที่ต้องระวัง

### 5.2 `LeadingChannelDrop`

**EN** — A thin wrapper that exists for exactly one purpose: consuming the external
**OpenEarthMap-SAR** pretrained U-Net, whose head predicts 9 classes where index 0 is a
`background`/`unknown` class. It slices off the leading channel(s) so the rest lines up
1:1 with this project's 0-based class order. Loading that checkpoint also requires
`dataset.normalization: zero_one` and a bare `state_dict` file — both handled by
[src/models/checkpoint.py](src/models/checkpoint.py) and the
`unet_effb4_oem_pretrained.yml` / `unet_effb4_oem_finetune.yml` configs.

**TH** — Wrapper บาง ๆ ที่มีไว้เพื่อจุดประสงค์เดียว คือใช้ weight U-Net จาก
**OpenEarthMap-SAR** ที่หัวโมเดลทำนาย 9 คลาส โดย index 0 เป็นคลาส `background`/`unknown`
มันจะตัด channel นำหน้าออก เพื่อให้ที่เหลือเรียงตรงกับลำดับคลาสฐาน 0 ของโปรเจกต์นี้
การโหลด checkpoint นั้นยังต้องใช้ `dataset.normalization: zero_one` และไฟล์ที่เป็น
`state_dict` เปล่า ๆ ซึ่งจัดการโดย [src/models/checkpoint.py](src/models/checkpoint.py)
และ config `unet_effb4_oem_pretrained.yml` / `unet_effb4_oem_finetune.yml`

---

## 6. Losses / ฟังก์ชัน loss

[src/losses/loss_factory.py](src/losses/loss_factory.py), selected by `training.loss`:

| `training.loss` | What it is / คืออะไร |
| --- | --- |
| `cross_entropy` | standard CE with `ignore_index`, optional `class_weights` / CE มาตรฐาน รองรับน้ำหนักคลาส |
| `dice` | smp multiclass Dice / Dice แบบหลายคลาส |
| `ce_dice` | `CEDiceLoss` = CE + Dice / รวมทั้งสอง |
| `focal` | smp Focal, `focal_gamma` (default 2.0) / ลดน้ำหนักพิกเซลที่ง่าย |
| `focal_dice` | `FocalDiceLoss` = Focal + Dice |

**EN** — Focal down-weights easy pixels, so it helps rare or frequently-misclassified
classes without needing explicit class weights. `training.class_weights` (a list whose
length must equal `num_classes`) works with `cross_entropy` and `ce_dice`; the weight
tensor is created on CPU and moved with `criterion.to(device)` in `train.py`.

**TH** — Focal loss ลดน้ำหนักของพิกเซลที่ทำนายง่าย จึงช่วยคลาสที่หายากหรือมักถูกทำนายผิด
โดยไม่ต้องกำหนดน้ำหนักคลาสเอง ส่วน `training.class_weights` (list ที่ความยาวต้องเท่ากับ
`num_classes`) ใช้ได้กับ `cross_entropy` และ `ce_dice` โดย tensor ถูกสร้างบน CPU แล้วย้าย
ไปยัง device ด้วย `criterion.to(device)` ใน `train.py`

---

## 7. Metrics / การวัดผล

**EN** — [src/metrics/segmentation_metrics.py](src/metrics/segmentation_metrics.py)
accumulates **one confusion matrix across the whole split** (not averaged per batch),
using a single vectorized `np.bincount` per update:

```python
indices = num_classes * targets + preds      # flatten (true, pred) into one index
cm += np.bincount(indices, minlength=num_classes**2).reshape(C, C)
```

Ignored and out-of-range pixels are dropped **before** the update. `compute()` then
derives:

| Metric | Meaning | ความหมาย |
| --- | --- | --- |
| `OA` | overall accuracy = correct / total pixels | ความถูกต้องรวมทุกพิกเซล |
| `mIoU` | mean Intersection-over-Union across classes | ค่าเฉลี่ย IoU ทุกคลาส |
| `mF1` | mean F1 across classes | ค่าเฉลี่ย F1 ทุกคลาส |
| `per_class_iou` / `per_class_f1` | one value per class | ค่าแยกรายคลาส |
| `class_support` | pixel count per true class | จำนวนพิกเซลจริงของแต่ละคลาส |
| `confusion_matrix` | full C×C matrix | เมทริกซ์ความสับสนเต็ม |

⚠️ **mIoU — not OA — is the model-selection metric.** OA can look excellent while a rare
class fails completely, because OA is dominated by whichever class covers the most pixels.

⚠️ **ใช้ mIoU ไม่ใช่ OA เป็นเกณฑ์เลือกโมเดล** เพราะ OA อาจดูดีมากทั้งที่คลาสหายากล้มเหลว
โดยสิ้นเชิง เนื่องจาก OA ถูกครอบงำโดยคลาสที่มีพิกเซลมากที่สุด

**EN** — `evaluate.py` additionally prints a **row-normalized** confusion matrix (each
row = one true class, sums to 100%, so the diagonal is *recall* and every off-diagonal
cell is "% of this true class that landed on that wrong prediction"), lists the worst
confusions, and saves a heatmap PNG next to the metrics JSON. That visualization step is
best-effort — if it fails, the numbers are already safe in the JSON and the evaluation
still succeeds.

**TH** — `evaluate.py` ยังพิมพ์ confusion matrix แบบ **normalize ตามแถว** (แต่ละแถวคือคลาสจริง
รวมเป็น 100% ดังนั้นแนวทแยงคือ *recall* และช่องนอกแนวทแยงคือ "กี่ % ของคลาสจริงนี้ที่ถูกทำนายผิด
ไปเป็นคลาสนั้น") พร้อมแสดงคู่ที่สับสนมากที่สุด และบันทึกภาพ heatmap PNG ไว้ข้าง ๆ ไฟล์ metrics JSON
ขั้นตอนวาดภาพนี้เป็นแบบ best-effort ถ้าล้มเหลวตัวเลขก็ยังปลอดภัยอยู่ใน JSON และการประเมินยังสำเร็จ

---

## 8. Training — `train.py` step by step / การเทรนทีละขั้น

```
python train.py --config configs/cei_oem/unet_effb4_oem2cei.yml
python train.py --config <config.yml> --init-weights <checkpoint_or_state_dict>
```

| # | Step | ขั้นตอน |
| --- | --- | --- |
| 1 | `load_config()` reads the YAML | อ่านไฟล์ YAML |
| 2 | `validate_config()` — requires `experiment`/`dataset`/`model`/`training`, and enforces `dataset.num_classes == model.num_classes` | ตรวจว่ามีครบทั้ง 4 ส่วน และจำนวนคลาสของ dataset กับ model ตรงกัน |
| 3 | `create_output_dir()` + **copy the config into the output folder** | สร้างโฟลเดอร์ผลลัพธ์ และ **คัดลอก config เข้าไปด้วย** |
| 4 | `select_device()` — CUDA → MPS → CPU | เลือกอุปกรณ์: CUDA → MPS → CPU |
| 5 | build train/val datasets + DataLoaders | สร้าง dataset และ DataLoader |
| 6 | `build_model()`, optional `--init-weights` warm start | สร้างโมเดล และโหลด weight เริ่มต้นถ้ามี |
| 7 | `build_loss().to(device)`, `build_optimizer()`, `build_scheduler()` | สร้าง loss / optimizer / scheduler |
| 8 | epoch loop → `train_one_epoch` → `validate_one_epoch` | วน epoch: เทรนแล้ววัดผล |
| 9 | write `logs/training_logs.json` + `checkpoints/last_checkpoint.pth` **every epoch** | เขียน log และ checkpoint **ทุก epoch** |
| 10 | if `val_mIoU > best_miou`: also write `best_checkpoint.pth` | ถ้า mIoU ดีขึ้น บันทึก `best_checkpoint.pth` เพิ่ม |
| 11 | optional early stopping via `early_stopping_patience` | หยุดก่อนกำหนดได้ถ้าตั้ง `early_stopping_patience` |

### 8.1 Things worth knowing / เรื่องที่ควรรู้

**EN**

- **Checkpoints embed the full config** (`checkpoint["config"]`). This is what lets
  `evaluate.py` detect a config/checkpoint mismatch later.
- **Everything is written every epoch**, so a crash loses at most the current epoch.
- **There is no resume-from-checkpoint flag.** Re-running a config starts fresh and
  overwrites its output folder. `--init-weights` is warm-start, not resume — it loads
  weights only, not the optimizer state or epoch counter.
- `--init-weights` also **forces `model.encoder_weights = None`**, so no ImageNet encoder
  download is triggered.
- **Mixed precision only engages on CUDA** — it is silently skipped on CPU/MPS.
- **Validation uses `batch_size = 1` whenever `eval_mode: full`**, because full-resolution
  tiles are not uniformly sized and cannot be stacked into a batch.
- `num_workers: 0` in most configs is deliberate: on Windows, workers each re-import torch
  and load its CUDA DLLs, which can exhaust the paging file (`WinError 1455`).
- `save_checkpoint()` re-runs `os.makedirs` on every write, because on OneDrive/antivirus-
  scanned drives a folder created at startup can be moved out from under a long run.

**TH**

- **Checkpoint ฝัง config ทั้งไฟล์ไว้ข้างใน** (`checkpoint["config"]`) นี่คือสิ่งที่ทำให้
  `evaluate.py` ตรวจจับความไม่ตรงกันของ config กับ checkpoint ได้ในภายหลัง
- **เขียนไฟล์ทุก epoch** ถ้าโปรแกรมล่ม จะเสียอย่างมากแค่ epoch ปัจจุบัน
- **ไม่มีคำสั่งเทรนต่อจาก checkpoint** การรัน config เดิมซ้ำจะเริ่มใหม่และทับโฟลเดอร์ผลลัพธ์เดิม
  ส่วน `--init-weights` คือการ warm start ไม่ใช่การเทรนต่อ — มันโหลดแค่ weight ไม่โหลด
  สถานะ optimizer หรือหมายเลข epoch
- `--init-weights` จะ **บังคับให้ `model.encoder_weights = None`** จึงไม่ดาวน์โหลด
  encoder ของ ImageNet
- **Mixed precision ทำงานเฉพาะบน CUDA** บน CPU/MPS จะถูกข้ามไปเงียบ ๆ
- **การ validate ใช้ `batch_size = 1` เสมอเมื่อ `eval_mode: full`** เพราะไทล์ความละเอียดเต็ม
  มีขนาดไม่เท่ากัน จึงรวมเป็น batch ไม่ได้
- `num_workers: 0` ใน config ส่วนใหญ่เป็นความตั้งใจ: บน Windows แต่ละ worker จะ import
  torch และโหลด CUDA DLL ใหม่ ซึ่งอาจทำให้ paging file เต็ม (`WinError 1455`)
- `save_checkpoint()` เรียก `os.makedirs` ซ้ำทุกครั้งที่เขียน เพราะบนไดรฟ์ที่ถูก OneDrive
  หรือโปรแกรมแอนตี้ไวรัสสแกน โฟลเดอร์ที่สร้างตอนเริ่มอาจถูกย้ายระหว่างที่รันยาว ๆ

---

## 9. Evaluation — `evaluate.py` / การประเมินผล

```bash
python evaluate.py --config <test_config.yml> --checkpoint <ckpt.pth> --split test [--tta]
python test.py     --config <test_config.yml> --checkpoint <ckpt.pth>   # = --split test
```

**EN** — Two features matter here:

**1. `check_architecture_matches()` — the guard against the most common real mistake.**
Because every checkpoint embeds the config that produced it, `evaluate.py` compares
`name` / `encoder_name` / `num_classes` and fails with a readable **diff table** instead
of hundreds of lines of `state_dict` key mismatches. Checkpoints without a stored config
(external weights) are left alone — `load_state_dict` still validates them.

**2. `--tta` — 4-way flip test-time augmentation.** `FlipTTA` runs the model on the
identity, horizontal flip, vertical flip and both, flips each result back, and averages
**in logit space** — so the output is still a valid logit map and argmax/loss downstream
behave exactly as for a single forward pass. Slower, but usually a small metric gain.
Available in `predict.py` too.

Output goes to `--output`, or by default
`<experiment.output_dir>/logs/<split>_metrics.json`, plus a `_confusion.png` beside it.

**TH** — มีสองสิ่งที่สำคัญ:

**1. `check_architecture_matches()` — ตัวป้องกันความผิดพลาดที่พบบ่อยที่สุด**
เนื่องจาก checkpoint ทุกไฟล์ฝัง config ที่สร้างมันไว้ `evaluate.py` จึงเทียบ
`name` / `encoder_name` / `num_classes` และหยุดพร้อม **ตารางเปรียบเทียบ** ที่อ่านเข้าใจได้
แทนที่จะแสดง error ของ `state_dict` เป็นร้อยบรรทัด ส่วน checkpoint ที่ไม่ได้เก็บ config
(weight จากภายนอก) จะถูกข้ามการตรวจนี้ แต่ `load_state_dict` ยังตรวจสอบให้อยู่ดี

**2. `--tta` — test-time augmentation แบบพลิกภาพ 4 ทาง** `FlipTTA` รันโมเดลกับภาพต้นฉบับ,
พลิกแนวนอน, พลิกแนวตั้ง และพลิกทั้งสองแกน แล้วพลิกผลลัพธ์กลับ จากนั้นเฉลี่ย **ในปริภูมิ logit**
ผลลัพธ์จึงยังเป็น logit ที่ถูกต้อง และ argmax/loss ที่ตามมาทำงานเหมือนการ forward ครั้งเดียว
ช้ากว่าแต่มักได้คะแนนดีขึ้นเล็กน้อย ใช้ได้ใน `predict.py` ด้วย

ผลลัพธ์เขียนไปที่ `--output` หรือค่าเริ่มต้น
`<experiment.output_dir>/logs/<split>_metrics.json` พร้อมไฟล์ `_confusion.png` ข้าง ๆ

---

## 10. Prediction — `predict.py` / การทำนาย

```bash
python predict.py --config <cfg.yml> --checkpoint <ckpt.pth> --input <file_or_dir> \
    [--panel] [--tile_size 1024 --overlap 128] [--format tiff] [--tta]
```

**EN** — Two inference modes:

- **Full-image** (default): normalize → pad to a multiple of 32 (reflect padding) → one
  forward pass → argmax → crop back to the original size.
- **Sliding-window** (`--tile_size N`): for imagery too large to fit in memory. Overlapping
  tiles are run independently and their **softmax probabilities** are accumulated into a
  full-size buffer, then divided by an overlap count — so seams are *averaged*, not
  hard-cut. Anchor positions always include a final tile flush against the bottom/right
  edge, so the whole image is covered even when its size is not a multiple of the stride.

`--panel` also saves an image+prediction side-by-side PNG. `--format tiff` writes a
lossless mask, which is what you want if you intend to hand-correct it in GIMP.

**TH** — มีโหมดการอนุมาน 2 แบบ:

- **ทั้งภาพ** (ค่าเริ่มต้น): normalize → เติมขอบให้หารด้วย 32 ลงตัว (แบบสะท้อน) → forward
  ครั้งเดียว → argmax → ตัดกลับเป็นขนาดเดิม
- **หน้าต่างเลื่อน** (`--tile_size N`): สำหรับภาพใหญ่เกินหน่วยความจำ ไทล์ที่ซ้อนกันจะถูกรัน
  แยกกัน แล้วนำ **ค่าความน่าจะเป็นจาก softmax** มาสะสมในบัฟเฟอร์ขนาดเต็มภาพ จากนั้นหารด้วย
  จำนวนครั้งที่ซ้อนทับ — รอยต่อจึงถูก *เฉลี่ย* ไม่ใช่ตัดขาด ตำแหน่งไทล์จะรวมไทล์สุดท้ายที่ชิดขอบ
  ล่าง/ขวาเสมอ เพื่อให้ครอบคลุมทั้งภาพแม้ขนาดจะหารด้วย stride ไม่ลงตัว

`--panel` จะบันทึกภาพเปรียบเทียบ (ภาพต้นฉบับคู่กับผลทำนาย) เพิ่มด้วย ส่วน `--format tiff`
เขียน mask แบบไม่สูญเสียคุณภาพ ซึ่งจำเป็นถ้าจะนำไปแก้ด้วยมือใน GIMP

---

## 11. The CEI hand-labeling workflow / ขั้นตอนการทำ label CEI ด้วยมือ

**EN** — CEI imagery has no ground truth, so labels are **bootstrapped from model
predictions and hand-corrected**:

```
tools/cei/predict_for_gimp.py   →   (edit *_mask.tif in GIMP)   →   tools/cei/import_from_gimp.py
                                                                            ↓
                                                                  data/CEI_data/labels/
```

| Step | Tool | What it does / ทำอะไร |
| --- | --- | --- |
| 1 | `predict_for_gimp.py` | predicts a tile and lays out a GIMP workspace. **Never overwrites an existing `_mask.tif` unless `--overwrite` is passed**, so re-running to add a new batch is always safe. |
| 2 | *(manual)* | correct the mask in GIMP using the CEI palette |
| 3 | `import_from_gimp.py` | snaps each pixel to the **nearest palette color** (tolerates GIMP anti-aliasing), writes raw CEI-encoded (`1–7`, `0` = unlabeled) single-channel PNGs. Warns if too few pixels are exactly on-palette (→ re-export as TIFF, not JPEG) or if a mask's size drifted from its image. |
| 4 | `make_test_split.py` | regenerates the CEI test split from whichever tiles currently have labels |

**TH** — ภาพ CEI ไม่มี ground truth จึงต้อง **สร้าง label ตั้งต้นจากผลทำนายของโมเดล แล้วแก้ด้วยมือ**
ตามขั้นตอนในตารางด้านบน จุดสำคัญคือ `predict_for_gimp.py` จะ **ไม่เขียนทับไฟล์ `_mask.tif`
ที่มีอยู่แล้ว เว้นแต่ใส่ `--overwrite`** ดังนั้นการรันซ้ำเพื่อเพิ่มไทล์ชุดใหม่จึงปลอดภัยเสมอ
และ `import_from_gimp.py` จะจับสีแต่ละพิกเซลเข้ากับสีในพาเลตที่ใกล้ที่สุด (รองรับ anti-alias
ของ GIMP) แล้วเขียนเป็น PNG ช่องเดียวรหัส `1–7` (`0` = ยังไม่ label)

Labeling happens **incrementally in batches of ~100 tiles**. Currently 112 tiles are
labeled: 100 form the test set (`test_split.txt`), and the fine-tune splits carve the
same pool into train/val/test differently.

การทำ label ดำเนินการ **ทีละชุดประมาณ 100 ไทล์** ปัจจุบันมี 112 ไทล์ที่ label แล้ว
โดย 100 ไทล์เป็นชุดทดสอบ (`test_split.txt`) ส่วน finetune splits แบ่งข้อมูลชุดเดียวกันนี้
เป็น train/val/test คนละแบบ

---

## 12. The experiments / การทดลอง

### 12.1 The main comparison / การเปรียบเทียบหลัก

**EN** — Five architectures × two source datasets = **10 checkpoints**, all scored on the
**same** 100-tile CEI test set. Data, splits, classes and schedule are identical across
models within a source, so the comparison isolates architecture.

| | OEM → CEI | IRSA → CEI |
| --- | --- | --- |
| Train configs | `configs/cei_oem/*.yml` | `configs/cei_irsa/*.yml` |
| Test configs | `configs/cei_oem/test/*.yml` | `configs/cei_irsa/test/*.yml` |
| Train / val | OEM, 2100 / 350 tiles, `oem_to_cei` | IRSAMap, `irsa_to_cei`, `nodata_to_ignore: 8` |
| Test | CEI, 100 tiles, `label_map: cei` | same / เหมือนกัน |

⚠️ One honest caveat, stated in `configs/cei_oem/Script.md`: batch size and learning rate
**differ by architecture family** — CNNs use batch 8 / lr 1e-4, the three transformer
models use batch 4 / lr 6e-5 / weight decay 0.01 (the standard transformer recipe). That
is deliberate — matching them would handicap one family — but it does mean the comparison
is not a pure architecture-only ablation.

⚠️ ข้อจำกัดที่ระบุไว้ตรง ๆ ใน `configs/cei_oem/Script.md`: batch size และ learning rate
**ต่างกันตามตระกูลสถาปัตยกรรม** — CNN ใช้ batch 8 / lr 1e-4 ส่วนโมเดล transformer ทั้งสาม
ใช้ batch 4 / lr 6e-5 / weight decay 0.01 (สูตรมาตรฐานของ transformer) ทำเช่นนี้โดยตั้งใจ
เพราะถ้าบังคับให้เท่ากันจะเสียเปรียบฝ่ายใดฝ่ายหนึ่ง แต่ก็หมายความว่าการเปรียบเทียบนี้
ไม่ใช่การ ablation ที่แยกผลของสถาปัตยกรรมล้วน ๆ

### 12.2 Running all 10 tests at once / รันทดสอบทั้ง 10 ตัวรวดเดียว

```bash
python pipeline/run_all_tests.py                      # all 10, next auto round
python pipeline/run_all_tests.py --round after_cei_v2 # name the round yourself
python pipeline/run_all_tests.py --models m2 m5       # subset
python pipeline/run_all_tests.py --sources oem        # OEM-trained only
python pipeline/run_all_tests.py --no-tta             # quick check
python pipeline/run_all_tests.py --dry-run            # print the plan, run nothing
python pipeline/collect_round_results.py --round round_7
```

**EN** — [pipeline/run_all_tests.py](pipeline/run_all_tests.py) runs `evaluate.py` ten
times and writes one consolidated comparison table. Design decisions worth knowing:

- A **failed run is recorded and the suite continues**, so one broken pairing doesn't hide
  the rest.
- A run whose output already exists is **reused, not re-evaluated**, unless `--force` —
  so an interrupted suite can be resumed safely.
- **Rounds**: every invocation writes into its own `logs/<round>/` subfolder, so re-testing
  against an updated dataset never clobbers a previous round's numbers. `--round`
  auto-increments by scanning existing round folders.

[pipeline/collect_round_results.py](pipeline/collect_round_results.py) then copies one
round's ten scattered result folders into a single flat `outputs/test_results/<round>/`
with a `summary.md`, so a round can be reviewed or shared without digging through
`experiments/`.

**TH** — [pipeline/run_all_tests.py](pipeline/run_all_tests.py) รัน `evaluate.py` สิบครั้ง
แล้วสรุปเป็นตารางเปรียบเทียบเดียว จุดออกแบบที่ควรรู้:

- **รันที่ล้มเหลวจะถูกบันทึกไว้แล้วรันต่อ** ปัญหาของโมเดลตัวเดียวจึงไม่บดบังผลของตัวอื่น
- รันที่มีไฟล์ผลลัพธ์อยู่แล้วจะ **ถูกนำมาใช้ซ้ำ ไม่รันใหม่** เว้นแต่ใส่ `--force`
  ดังนั้นถ้าถูกขัดจังหวะก็รันต่อได้อย่างปลอดภัย
- **ระบบ round**: การรันแต่ละครั้งเขียนลงโฟลเดอร์ย่อย `logs/<round>/` ของตัวเอง
  การทดสอบซ้ำกับข้อมูลที่อัปเดตแล้วจึงไม่ทับตัวเลขของรอบก่อน `--round` จะเพิ่มเลขอัตโนมัติ

จากนั้น [pipeline/collect_round_results.py](pipeline/collect_round_results.py) จะรวบรวม
ผลลัพธ์ที่กระจายอยู่ 10 ที่ของรอบนั้นมาไว้ในโฟลเดอร์แบนเดียวคือ
`outputs/test_results/<round>/` พร้อม `summary.md` เพื่อให้ทบทวนหรือแชร์ได้โดยไม่ต้อง
ไปไล่หาในโครงสร้าง `experiments/`

### 12.3 Latest results / ผลลัพธ์ล่าสุด

From [outputs/cei_test_report/round_7/summary.md](outputs/cei_test_report/round_7/summary.md)
(mIoU on the 100-tile CEI test set, with TTA):

| Key | Model | OEM → CEI | IRSA → CEI |
| --- | --- | --- | --- |
| m1 | FT-UNetFormer Swin-B | 0.5638 | **0.5690** |
| m2 | U-Net EfficientNet-B4 | 0.4529 | 0.5554 |
| m3 | UNetFormer ResNet-101 | 0.5179 | 0.5452 |
| m4 | UPerNet Swin-B | 0.5659 | 0.5677 |
| m5 | SegFormer MiT-B5 | 0.4269 | 0.5640 |

**EN** — Two patterns stand out: **IRSA-trained models generalize to CEI better than
OEM-trained ones across every architecture**, and the gap is largest for the two models
that do worst on OEM (m2, m5). Numbers are a snapshot of round 7/8 — re-run
`pipeline/run_all_tests.py` for current values.

**TH** — เห็นรูปแบบสองอย่างชัดเจน: **โมเดลที่เทรนด้วย IRSA generalize ไปยัง CEI ได้ดีกว่า
โมเดลที่เทรนด้วย OEM ในทุกสถาปัตยกรรม** และช่องว่างกว้างที่สุดในสองโมเดลที่ทำได้แย่ที่สุด
บน OEM (m2, m5) ตัวเลขนี้เป็นภาพ ณ รอบที่ 7/8 — รัน `pipeline/run_all_tests.py`
เพื่อดูค่าปัจจุบัน

### 12.4 Fine-tuning experiments / การทดลอง fine-tune

`configs/cei_oem_finetune/` contains three follow-up experiments that warm-start from the
OEM baseline checkpoint via `--init-weights`:

| Config | Idea / แนวคิด |
| --- | --- |
| `finetune.yml` | fine-tune on 90 CEI training tiles, lr 1e-5, `augment: true`, early stopping patience 15 |
| `finetune_agri_aware.yml` | same, but with class-aware sampling aimed at Agriculture |
| `finetune_oem_cei_mix.yml` | joint training: OEM's 2100 tiles + 10 CEI tiles oversampled 25× (~10.6% CEI per epoch), via `dataset.name: OEM_CEI_Mix` |

⚠️ **EN** — Note the comment in `finetune_oem_cei_mix.yml`: it *trains* on `maesuai_1..10`,
which are part of the fixed test set every other experiment was scored on. Its results
are therefore **not comparable** to the others, which is why that config points val/test
at completely different tiles.

⚠️ **TH** — สังเกตคอมเมนต์ใน `finetune_oem_cei_mix.yml`: มัน *เทรน* บน `maesuai_1..10`
ซึ่งเป็นส่วนหนึ่งของชุดทดสอบคงที่ที่การทดลองอื่นใช้ให้คะแนน ผลของมันจึง **เทียบกับตัวอื่นไม่ได้**
เพราะเหตุนี้ config ดังกล่าวจึงชี้ val/test ไปยังไทล์คนละชุดโดยสิ้นเชิง

---

## 13. Repository map / แผนผังโครงสร้างโปรเจกต์

```
lulc_cei/
├── train.py                    entry point: train + validate / จุดเริ่มต้น: เทรนและวัดผล
├── evaluate.py                 score a checkpoint (+ FlipTTA, arch guard)
├── test.py                     thin shim = evaluate.py --split test
├── predict.py                  inference: full-image or sliding-window
│
├── src/
│   ├── datasets/
│   │   ├── taxonomy.py           ★ the 3 label spaces + LUT builder
│   │   ├── dataset_factory.py      build_dataset() dispatch + OEM/CEI mix
│   │   ├── openearthmap_dataset.py OEM + CEI loader (+ class-aware sampling)
│   │   ├── irsa_dataset.py         IRSAMap loader with injected collaborators
│   │   ├── label_mapping.py        LabelMapper / MaskRefiner abstractions
│   │   └── transforms.py           normalization presets + aug pipelines
│   ├── models/
│   │   ├── model_factory.py        build_model() dispatch, LeadingChannelDrop
│   │   ├── unetformer.py           custom UNetFormer (GLTB decoder)
│   │   ├── ftunetformer.py         custom FT-UNetFormer (Swin-B)
│   │   └── checkpoint.py           load weights from either on-disk format
│   ├── engine/
│   │   ├── trainer.py              train_one_epoch + compute_loss (aux head)
│   │   └── validator.py            validate_one_epoch + metric accumulation
│   ├── losses/loss_factory.py      CE / Dice / CE+Dice / Focal / Focal+Dice
│   ├── metrics/segmentation_metrics.py  confusion-matrix OA / mIoU / mF1
│   └── utils/
│       ├── config.py               YAML loader
│       ├── confusion.py            confusion formatting + heatmap PNG
│       └── visualization.py        palette decode/encode, panels
│
├── configs/
│   ├── oem/                    native 8-class OEM training
│   ├── cei_oem/                OEM → CEI (train/, test/, smoke/)
│   ├── cei_irsa/               IRSA → CEI (train/, test/)
│   └── cei_oem_finetune/       fine-tune + OEM/CEI mix experiments
│
├── tools/
│   ├── cei/                    CEI labeling workflow, smoke tests, reports
│   ├── oem/                    OEM splits + dataset diagnostics
│   ├── irsa/make_splits.py     IRSAMap split generation
│   ├── plot_training_curves.py
│   └── show_confusion.py
│
├── pipeline/
│   ├── run_all_tests.py        score all 10 checkpoints on CEI, by round
│   └── collect_round_results.py  flatten one round into outputs/test_results/
│
├── data/                       OpenEarthMap/ · IRSAMap/ · CEI_data/ · xbd/
├── experiments/                per-run checkpoints/ + logs/
├── outputs/                    reports, visualizations, collected results
└── pretrain_weight/            external OpenEarthMap-SAR U-Net weights
```

---

## 14. Anatomy of a config / กายวิภาคของไฟล์ config

```yaml
experiment:
  name: cei_oem_m2_unet_effb4                       # ชื่อ experiment
  output_dir: ./experiments/exp_02_cei_oem_m2_unet_effb4   # ทุกอย่างเขียนลงที่นี่

dataset:
  name: OpenEarthMap        # เลือก loader / picks the loader
  root: data/OpenEarthMap/OpenEarthMap_wo_xBD
  image_dir: <region>/images     # <region> ถูกแทนจากชื่อไฟล์ / substituted from the filename
  mask_dir: <region>/labels
  train_split: train_split_60.txt
  val_split: val_split_10.txt
  test_split: test_split_30.txt
  label_map: oem_to_cei     # ★ ระบบแปลง label / the label scheme
  crop_size: 512
  num_classes: 7            # ต้องเท่ากับ model.num_classes
  ignore_index: 255
  normalization: imagenet   # ★ ต้องตรงกับ weight / must match the weights
  augment: false            # สูตร paper: crop อย่างเดียว
  eval_mode: full           # full = ทั้งภาพ (paper) / crop = เร็วกว่า
  # nodata_to_ignore: 8     # IRSA เท่านั้น / IRSA only
  # mask_suffix: _label     # ถ้า mask ชื่อไม่ตรงกับ image
  # class_aware_sampling: {...}   # opt-in

model:
  name: unet                # unet | deeplabv3 | upernet | unetformer | ftunetformer | segformer
  encoder_name: efficientnet-b4
  encoder_weights: imagenet # ถูกบังคับเป็น None ตอน evaluate/predict/--init-weights
  in_channels: 3
  num_classes: 7            # ต้องเท่ากับ dataset.num_classes
  # encoder_params: {img_size: 512}   # จำเป็นสำหรับ Swin

training:
  batch_size: 8
  epochs: 60
  learning_rate: 0.0001
  optimizer: AdamW          # adam | adamw
  weight_decay: 0.000001
  loss: cross_entropy       # cross_entropy | dice | ce_dice | focal | focal_dice
  scheduler: none           # none | cosine  (+ min_learning_rate)
  mix_precision: true       # CUDA เท่านั้น / CUDA only
  num_workers: 0            # 0 บน Windows เพื่อเลี่ยง WinError 1455
  # early_stopping_patience: 15
  # class_weights: [...]    # ความยาวต้องเท่ากับ num_classes
  # aux_weight: 0.4         # เฉพาะ UNetFormer / FT-UNetFormer

metrics:                    # descriptive only / เป็นเอกสารประกอบเท่านั้น
  - OA
  - mIoU
  - mF1
```

**EN** — Note: `metrics:` is purely descriptive — the metrics computed are fixed by
`SegmentationMetrics`. Test configs keep only the keys `evaluate.py` actually reads; they
deliberately omit `encoder_weights` (forced to `None`), `batch_size` (ignored under
`eval_mode: full`), and the training-only keys.

**TH** — หมายเหตุ: `metrics:` เป็นเพียงคำอธิบาย — metric ที่คำนวณจริงถูกกำหนดตายตัวโดย
`SegmentationMetrics` ส่วน config สำหรับทดสอบจะเก็บเฉพาะคีย์ที่ `evaluate.py` อ่านจริง
โดยตั้งใจตัด `encoder_weights` (ถูกบังคับเป็น `None`), `batch_size` (ไม่ใช้เมื่อ
`eval_mode: full`) และคีย์ที่ใช้เฉพาะตอนเทรนออก

---

## 15. Common commands / คำสั่งที่ใช้บ่อย

```bash
# ── Setup / ติดตั้ง ────────────────────────────────────────────────────────
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows;  source .venv/bin/activate on Linux/macOS
python -m pip install -r requirements.txt

# ── Prepare data (once per dataset) / เตรียมข้อมูล (ทำครั้งเดียวต่อชุด) ──────
python tools/oem/create_labeled_splits.py --config configs/oem/unet_effb4_oem.yml
python tools/irsa/make_splits.py --stratify

# ── Smoke test first / ทดสอบสั้น ๆ ก่อน (3 epochs, ข้อมูลจริง) ───────────────
python tools/cei/make_smoke_configs.py --epochs 3
python tools/cei/run_smoke_tests.py [--models m2 m5] [--skip-test]

# ── Train / เทรน ─────────────────────────────────────────────────────────────
python train.py --config configs/cei_oem/unet_effb4_oem2cei.yml
python train.py --config configs/cei_irsa/unet_effb4_irsa2cei.yml
python train.py --config configs/cei_oem_finetune/finetune.yml \
    --init-weights experiments/cei_oem_m2_unet_effb4/checkpoints/best_checkpoint.pth

# ── Evaluate / ประเมินผล ─────────────────────────────────────────────────────
python evaluate.py --config configs/cei_oem/test/test_m2_uneteffb4_cei.yml \
    --checkpoint experiments/cei_oem_m2_unet_effb4/checkpoints/best_checkpoint.pth \
    --split test --tta
python pipeline/run_all_tests.py
python pipeline/collect_round_results.py --round round_7

# ── Predict / ทำนาย ──────────────────────────────────────────────────────────
python predict.py --config <cfg.yml> --checkpoint <ckpt.pth> \
    --input data/CEI_data/images --panel --tta

# ── CEI labeling workflow / ขั้นตอนทำ label CEI ──────────────────────────────
python tools/cei/predict_for_gimp.py ...     # 1. สร้าง mask ตั้งต้น
#    (แก้ *_mask.tif ใน GIMP)                # 2. แก้ด้วยมือ
python tools/cei/import_from_gimp.py ...     # 3. นำเข้ากลับ
python tools/cei/make_test_split.py --start 1 --end 100   # 4. สร้าง split ใหม่

# ── Diagnostics / เครื่องมือตรวจสอบ ──────────────────────────────────────────
python tools/oem/check_dataset.py --config <cfg.yml> --split train --num_samples 5
python tools/oem/check_missing_files.py --config <cfg.yml> --split train
python tools/oem/count_pixels.py --config <cfg.yml> --split train
python tools/oem/analyze_errors.py ...
python tools/cei/compare_predictions.py --config <cfg.yml> --checkpoint <ckpt.pth> \
    --tta --num 12 --out <out.png>
python tools/cei/visualize_cei_dataset.py
python tools/plot_training_curves.py
python tools/show_confusion.py <metrics.json>
```

> ⚠️ There is **no unit test suite** in this repo. `tools/cei/run_smoke_tests.py` is an
> *integration* smoke test — it actually trains each of the 5 architectures for a few
> epochs on real data and evaluates on CEI. It is not a fast synthetic-data check, and a
> pass does **not** rank the architectures (3 epochs is far too few — model 2's Water IoU
> went 0.00 → 0.72 between epochs 2 and 5).
>
> ⚠️ โปรเจกต์นี้ **ไม่มี unit test** `tools/cei/run_smoke_tests.py` เป็น smoke test แบบ
> *integration* — มันเทรนทั้ง 5 สถาปัตยกรรมจริง ๆ ไม่กี่ epoch บนข้อมูลจริงแล้วประเมินบน CEI
> ไม่ใช่การทดสอบเร็ว ๆ ด้วยข้อมูลสังเคราะห์ และการผ่านไม่ได้แปลว่าจัดอันดับสถาปัตยกรรมได้
> (3 epoch น้อยเกินไปมาก — Water IoU ของโมเดล 2 ขยับจาก 0.00 → 0.72 ระหว่าง epoch 2 ถึง 5)

---

## 16. Gotchas checklist / รายการข้อควรระวัง

| # | Gotcha | ข้อควรระวัง |
| --- | --- | --- |
| 1 | `normalization` must match the checkpoint's training. Wrong = incoherent output. | `normalization` ต้องตรงกับตอนเทรน ถ้าผิดผลลัพธ์จะมั่วไปเลย |
| 2 | `label_map` must match the data on disk — otherwise the loader raises (by design). | `label_map` ต้องตรงกับข้อมูลจริง ไม่งั้น loader จะโยน error (ตั้งใจให้เป็นแบบนั้น) |
| 3 | `dataset.num_classes` must equal `model.num_classes`; `validate_config` enforces it. | จำนวนคลาสของ dataset กับ model ต้องเท่ากัน |
| 4 | `nodata_to_ignore` must match between IRSA train and test configs. | `nodata_to_ignore` ต้องตรงกันระหว่าง config เทรนกับทดสอบของ IRSA |
| 5 | Swin models need `encoder_params.img_size` matching the input size. | โมเดล Swin ต้องตั้ง `encoder_params.img_size` ให้ตรงกับขนาด input |
| 6 | Re-running a config **overwrites** its output folder — there is no resume. | การรัน config ซ้ำจะ **ทับ** โฟลเดอร์ผลลัพธ์ — ไม่มีการเทรนต่อ |
| 7 | Pairing model A's test config with model B's checkpoint → caught by `check_architecture_matches`. | ใช้ config ของโมเดล A กับ checkpoint ของโมเดล B → ถูกจับโดย `check_architecture_matches` |
| 8 | `--init-weights` loads weights only; optimizer state and epoch count are not restored. | `--init-weights` โหลดแค่ weight ไม่คืนสถานะ optimizer หรือหมายเลข epoch |
| 9 | mIoU (not OA) selects the best checkpoint. | ใช้ mIoU (ไม่ใช่ OA) เลือก checkpoint ที่ดีที่สุด |
| 10 | `num_workers > 0` on Windows can trigger `WinError 1455`; most configs use `0`. | `num_workers > 0` บน Windows อาจทำให้เกิด `WinError 1455` config ส่วนใหญ่จึงใช้ `0` |
| 11 | `README.md` is partially stale (old `configs/unet/...` paths, `tests/smoke_test.py`). Trust `CLAUDE.md` and this file. | `README.md` ล้าสมัยบางส่วน (path `configs/unet/...` เก่า, `tests/smoke_test.py`) ให้เชื่อ `CLAUDE.md` และไฟล์นี้ |
