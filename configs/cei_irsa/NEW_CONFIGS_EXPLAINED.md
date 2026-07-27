# Config ใหม่ฝั่ง IRSA คืออะไร ทำอะไร มาจากไหน

เอกสารนี้อธิบายไฟล์ config ที่เพิ่งสร้างใหม่ในโฟลเดอร์ `configs/cei_irsa/`
เขียนให้คนที่ไม่ได้อยู่ด้วยตอนสร้างอ่านแล้วเข้าใจได้

---

## สรุปสั้นสุด

โปรเจกต์นี้เทรนโมเดล segmentation (แยกประเภทที่ดิน/สิ่งปกคลุมดิน LULC เป็น 7 คลาส
ตามระบบ "CEI") แล้วเทียบว่า **ข้อมูลเทรนชุดไหน / สถาปัตยกรรมไหน** ทำงานกับภาพ CEI
ได้ดีที่สุด

- ฝั่ง **OEM** (ข้อมูล OpenEarthMap) มี config ครบ 5 สถาปัตยกรรมอยู่แล้ว
- ฝั่ง **IRSA** (ข้อมูล IRSAMap) เดิมมีแค่ 1 ตัว (UNet)

ไฟล์ใหม่ = **สร้าง config ฝั่ง IRSA เพิ่มอีก 4 ตัว** ให้ครบ 5 สถาปัตยกรรมเท่าฝั่ง OEM
เพื่อให้จับคู่เทียบกันได้ตรงๆ ว่า "ข้อมูล IRSA vs OEM ต่างกันแค่ข้อมูล ไม่ใช่โมเดล"

---

## ไฟล์ใหม่ 4 ไฟล์

| ไฟล์ | สถาปัตยกรรมโมเดล | encoder หลัก |
|---|---|---|
| `segformer_mitb5_irsa2cei.yml` | SegFormer | MiT-B5 (transformer) |
| `unetformer_r101_irsa2cei.yml` | UNetFormer | ResNet-101 (CNN) |
| `upernet_swinb_irsa2cei.yml` | UPerNet | Swin-B (transformer) |
| `ftunetformer_swinb_irsa2cei.yml` | FT-UNetFormer | Swin-B (transformer) |

(ตัวที่ 5 คือ `unet_effb4_irsa2cei.yml` — UNet + EfficientNet-B4 — มีอยู่ก่อนแล้ว)

---

## แต่ละไฟล์ทำหน้าที่อะไร

ไฟล์ `.yml` พวกนี้คือ **ไฟล์ตั้งค่า (config)** ไม่ใช่โค้ด — มันบอกสคริปต์เทรน
(`train.py`) ว่า:

1. **ใช้ข้อมูลอะไร** — IRSAMap จากโฟลเดอร์ `data/IRSAMap/`
2. **ใช้โมเดลอะไร** — เช่น SegFormer, UPerNet ฯลฯ
3. **เทรนยังไง** — กี่ epoch, batch size เท่าไหร่, learning rate เท่าไหร่
4. **วัดผลด้วยอะไร** — OA, mIoU, mF1 และค่าแยกรายคลาส

เวลาใช้งานก็แค่สั่ง เช่น:
```bash
python train.py --config configs/cei_irsa/segformer_mitb5_irsa2cei.yml
```
สคริปต์จะอ่านไฟล์นี้แล้วเทรนตามที่เขียนไว้ทุกอย่าง

---

## มันมาจากไหน (ที่มา)

ไฟล์ใหม่แต่ละตัว **ก๊อปโครงมาจาก 2 ที่ผสมกัน**:

### 1. ส่วน "โมเดล + วิธีเทรน" → มาจาก config OEM ตัวเดียวกัน
เช่น `segformer_mitb5_irsa2cei.yml` เอาส่วน `model:` และ `training:` มาจาก
`configs/cei_oem/segformer_mitb5_oem2cei.yml` แบบเป๊ะๆ (encoder, batch size,
learning rate, weight decay เหมือนเดิมทั้งหมด)

**เหตุผล:** เพื่อให้เทียบ IRSA กับ OEM ได้อย่างยุติธรรม — โมเดลและสูตรเทรน
ต้องเหมือนกัน ต่างกันแค่ข้อมูลเท่านั้น

### 2. ส่วน "ข้อมูล" → มาจาก config IRSA ที่มีอยู่เดิม
ส่วน `dataset:` ก๊อปมาจาก `configs/cei_irsa/unet_effb4_irsa2cei.yml`
ซึ่งมีการตั้งค่าเฉพาะของ IRSA อยู่ 3 จุดสำคัญ:

- `name: IRSA_Map` + `root: data/IRSAMap` — ชี้ไปที่ข้อมูล IRSA
- `label_map: irsa_to_cei` — แปลงรหัสคลาสของ IRSA → 7 คลาส CEI
- `nodata_to_ignore: 8` — กัน "ขอบดำ" ของภาพ (border padding) ไม่ให้ถูกนับ
  เป็นพื้นดินจริง ( IRSA ไม่ label bareland ทำให้พื้นหลังปนกับขอบดำ)
- `mask_dir: train/SegLabel_vwsbr` — เลือก mask แบบที่ถนนทับอาคารตรงที่ซ้อนกัน

### สิ่งเดียวที่แก้เพิ่มเอง
ตั้ง `epochs: 200` ให้ทั้ง 4 ตัว (ของ OEM บางตัวเป็น 100 แต่เจ้าของโปรเจกต์
ขอให้ IRSA ใช้ 200 ทุกตัว)

---

## test_m1 – m5 คืออะไร (config สำหรับ "เทสต์")

นอกจาก config เทรนแล้ว ในโฟลเดอร์ `configs/cei_irsa/test/` ยังมี config อีกชุด
ชื่อขึ้นต้นด้วย `test_m1` ถึง `test_m5` — พวกนี้คือ **config สำหรับวัดผล (evaluate)**
ไม่ใช่เทรน ใช้ตอนเทรนโมเดลเสร็จแล้ว เอา checkpoint มาวัดว่าแม่นแค่ไหน สั่งด้วย
`evaluate.py` ไม่ใช่ `train.py`

### ทำไมต้องแยก config เทรน กับ config เทสต์
เพราะ "ข้อมูลที่เทรน" กับ "ข้อมูลที่เอาไปวัดผล" คนละชุดกัน:
- เทรน → ใช้ IRSAMap (`train/`)
- เทสต์ → ใช้ภาพแม่สาย (CEI) หรือ IRSA test split (`test/`)

ถ้าใช้ config เดียวกันไม่ได้ เพราะ path ข้อมูล, `label_map`, ขนาดภาพ ต่างกันหมด

### เลข m1–m5 หมายถึงอะไร
เป็น **เลขประจำโมเดล** ที่ยกมาจากฝั่ง OEM ตรงๆ เพื่อให้จับคู่เทียบกันง่าย:

| เลข | โมเดล |
|---|---|
| m1 | FT-UNetFormer + Swin-B |
| m2 | U-Net + EfficientNet-B4 (ไฟล์ชื่อ `test_irsa2cei_*` ไม่ได้ใส่ m2) |
| m3 | UNetFormer + ResNet-101 |
| m4 | UPerNet + Swin-B |
| m5 | SegFormer + MiT-B5 |

### ทำไมแต่ละโมเดลต้องมี test config ของตัวเอง (ใช้ร่วมกันไม่ได้)
เพราะ `evaluate.py` ต้อง **ประกอบโมเดลขึ้นมาใหม่ให้ตรงกับ checkpoint** ก่อนโหลด
weight — ดังนั้น test config ต้องมีส่วน `model:` ที่ตรงกับตอนเทรนเป๊ะ (เช่น
`name: segformer`, `encoder_name: mit_b5`) ถ้าเอา test config ของ UNet ไปจับกับ
checkpoint ของ SegFormer โค้ดจะฟ้อง error ทันที (มีตัวเช็ก `check_architecture_matches`
กันพลาดตรงนี้อยู่) — นี่คือเหตุผลหลักที่ต้องแยกไฟล์ตัวละโมเดล

### แต่ละโมเดลมี test 2 แบบ (รวมเป็น 10 ไฟล์)
| แบบ | เทสต์บนอะไร | เอาไว้ทำอะไร |
|---|---|---|
| `_on_cei` | ภาพแม่สาย (CEI) | **เลขหลัก** — เอาไปเทียบกับโมเดลฝั่ง OEM บนภาพชุดเดียวกัน |
| `_on_irsa` | IRSA test split | เทสต์ในโดเมนตัวเอง — ส่วนต่างของสองแบบคือ "generalization gap" |

### จุดที่ต่างกันต่อโมเดล (ที่ควรรู้)
- **UPerNet (m4)** ต้องใส่ `img_size: 1024` ในส่วน model ตอนเทสต์ เพราะ Swin ถูก
  สร้างมาสำหรับขนาด input คงที่ ภาพเทสต์เป็น 1024 (ไม่ใช่ 512 ตอนเทรน) — weight
  ของ Swin ไม่ผูกกับความละเอียด จึงโหลด checkpoint 512 มาใช้กับ 1024 ได้
- **FT-UNetFormer (m1) / SegFormer (m5)** ไม่ต้องใส่ `img_size` (ทำงานกับ 1024 ได้เลย)
- ทุกไฟล์ `_on_irsa` มี `nodata_to_ignore: 8` และ **ต้องตรงกับตอนเทรน** ไม่งั้น
  สัดส่วนคลาสจะเพี้ยน

### ตัวอย่างการใช้
```bash
# วัดผล SegFormer ที่เทรนบน IRSA บนภาพแม่สาย (เลขที่เอาไปเทียบ OEM)
python evaluate.py --config configs/cei_irsa/test/test_m5_segformer_on_cei.yml \
  --checkpoint experiments/cei_irsa_m5_segformer_mitb5/checkpoints/best_checkpoint.pth \
  --split test --tta
```

---

## สถานะตอนนี้ / ต้องทำอะไรต่อ

Config พร้อมแล้ว (ตรวจแล้วว่า YAML valid ทุกไฟล์) แต่ **ยังเทรนไม่ได้ทันที** เพราะ:

1. ยังไม่มีข้อมูล — ต้องดาวน์โหลด IRSAMap มาวางที่ `data/IRSAMap/` ก่อน
   (จาก https://github.com/ucas-dlg/IRSAMap)
2. ยังไม่มีไฟล์ split — ต้องสร้างด้วย `python tools/irsa/make_splits.py --stratify`
   (รันได้หลังมีข้อมูลแล้ว)

จากนั้นค่อยเริ่มเทรน โดยแผนคือ **เทรน UNet ตัวแรกให้ผ่านก่อน** แล้วค่อยไล่เทรน
อีก 4 ตัว
