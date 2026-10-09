# Test commands: 5 models × 2 datasets on the CEI test set (100 tiles)
# คำสั่งเทสทั้ง 5 โมเดล × 2 dataset บนชุด CEI (100 tiles)

Every command evaluates each model's best checkpoint on the same CEI test set
(`data/CEI_data`, `maesuai_1..100`), so results are directly comparable across
models and across training datasets.

- Results are written to `experiments/<model>/test/test_metrics.json`, with a
  confusion-matrix image `test_metrics_confusion.png` next to it.
- The commands below do **not** use TTA — to enable it, append `--tta` and
  change the output filename to `test_metrics_tta.json` so files don't collide.
- Run one command at a time (single GPU; running in parallel causes contention
  and risks out-of-memory).
- The primary comparison metric is **mIoU**.

ทุกคำสั่งประเมิน checkpoint ตัว best ของแต่ละโมเดลบนชุดเทส CEI เดียวกัน
(`data/CEI_data`, `maesuai_1..100`) ผลจึงเทียบข้ามโมเดล/ข้าม dataset ได้ตรงๆ

- ผลลัพธ์เขียนไปที่ `experiments/<โมเดล>/test/test_metrics.json`
  พร้อมรูป confusion matrix `test_metrics_confusion.png` ข้างกัน
- คำสั่งด้านล่าง**ไม่ใช้ TTA** — ถ้าต้องการ TTA ให้เติม `--tta`
  และเปลี่ยนชื่อไฟล์ output เป็น `test_metrics_tta.json` เพื่อไม่ให้ทับกัน
- ควรรันทีละคำสั่ง (GPU ตัวเดียว รันพร้อมกันจะแย่งกันเองและเสี่ยง out-of-memory)
- ตัวเลขที่ใช้เปรียบเทียบหลักคือ **mIoU**

New terminal / เปิด terminal ใหม่: `Terminal > New Terminal` (Ctrl+Shift+`)
Rename / เปลี่ยนชื่อ: right-click the terminal name in the side list > Rename
(คลิกขวาที่ชื่อ terminal ในแถบขวา > Rename)

---

## OEM → CEI (trained on OpenEarthMap / เทรนบน OpenEarthMap)

### Terminal: `Test OEM m1 ftunetformer`

```powershell
python evaluate.py --config configs/cei_oem/test/test_m1_ftunetformer_cei.yml --checkpoint experiments/cei_oem_m1_ftunetformer_swinb/checkpoints/best_checkpoint.pth --split test --output experiments/cei_oem_m1_ftunetformer_swinb/test/test_metrics.json
```

### Terminal: `Test OEM m2 unet_effb4`

```powershell
python evaluate.py --config configs/cei_oem/test/test_m2_uneteffb4_cei.yml --checkpoint experiments/cei_oem_m2_unet_effb4/checkpoints/best_checkpoint.pth --split test --output experiments/cei_oem_m2_unet_effb4/test/test_metrics.json
```

### Terminal: `Test OEM m3 unetformer`

```powershell
python evaluate.py --config configs/cei_oem/test/test_m3_unetformer_cei.yml --checkpoint experiments/cei_oem_m3_unetformer_r101/checkpoints/best_checkpoint.pth --split test --output experiments/cei_oem_m3_unetformer_r101/test/test_metrics.json
```

### Terminal: `Test OEM m4 upernet`

```powershell
python evaluate.py --config configs/cei_oem/test/test_m4_upernet_cei.yml --checkpoint experiments/cei_oem_m4_upernet_swinb/checkpoints/best_checkpoint.pth --split test --output experiments/cei_oem_m4_upernet_swinb/test/test_metrics.json
```

### Terminal: `Test OEM m5 segformer`

```powershell
python evaluate.py --config configs/cei_oem/test/test_m5_segformer_cei.yml --checkpoint experiments/cei_oem_m5_segformer_mitb5/checkpoints/best_checkpoint.pth --split test --output experiments/cei_oem_m5_segformer_mitb5/test/test_metrics.json
```

---

## IRSA → CEI (trained on IRSAMap / เทรนบน IRSAMap)

### Terminal: `Test IRSA m1 ftunetformer`

```powershell
python evaluate.py --config configs/cei_irsa/test/test_m1_ftunetformer_on_cei.yml --checkpoint experiments/cei_irsa_m1_ftunetformer_swinb/checkpoints/best_checkpoint.pth --split test --output experiments/cei_irsa_m1_ftunetformer_swinb/test/test_metrics.json
```

### Terminal: `Test IRSA m2 unet_effb4`

```powershell
python evaluate.py --config configs/cei_irsa/test/test_m2_uneteffb4_on_cei.yml --checkpoint experiments/cei_irsa_m2_unet_effb4/checkpoints/best_checkpoint.pth --split test --output experiments/cei_irsa_m2_unet_effb4/test/test_metrics.json
```

### Terminal: `Test IRSA m3 unetformer`

```powershell
python evaluate.py --config configs/cei_irsa/test/test_m3_unetformer_on_cei.yml --checkpoint experiments/cei_irsa_m3_unetformer_r101/checkpoints/best_checkpoint.pth --split test --output experiments/cei_irsa_m3_unetformer_r101/test/test_metrics.json
```

### Terminal: `Test IRSA m4 upernet`

```powershell
python evaluate.py --config configs/cei_irsa/test/test_m4_upernet_on_cei.yml --checkpoint experiments/cei_irsa_m4_upernet_swinb/checkpoints/best_checkpoint.pth --split test --output experiments/cei_irsa_m4_upernet_swinb/test/test_metrics.json
```

### Terminal: `Test IRSA m5 segformer`

```powershell
python evaluate.py --config configs/cei_irsa/test/test_m5_segformer_on_cei.yml --checkpoint experiments/cei_irsa_m5_segformer_mitb5/checkpoints/best_checkpoint.pth --split test --output experiments/cei_irsa_m5_segformer_mitb5/test/test_metrics.json
```

---

## After testing: inspect results / เทสเสร็จแล้วดูผล

```powershell
# View the confusion matrix + metrics of any result file
# ดู confusion matrix + metrics ของไฟล์ใดไฟล์หนึ่ง
python tools/show_confusion.py experiments/cei_oem_m1_ftunetformer_swinb/test/test_metrics.json
```

| Dataset | Model / โมเดล | Result file / ไฟล์ผล |
| --- | --- | --- |
| OEM | m1 FT-UNetFormer + Swin-B | `experiments/cei_oem_m1_ftunetformer_swinb/test/test_metrics.json` |
| OEM | m2 U-Net + EfficientNet-B4 | `experiments/cei_oem_m2_unet_effb4/test/test_metrics.json` |
| OEM | m3 UNetFormer + ResNet-101 | `experiments/cei_oem_m3_unetformer_r101/test/test_metrics.json` |
| OEM | m4 UPerNet + Swin-B | `experiments/cei_oem_m4_upernet_swinb/test/test_metrics.json` |
| OEM | m5 SegFormer + MiT-B5 | `experiments/cei_oem_m5_segformer_mitb5/test/test_metrics.json` |
| IRSA | m1 FT-UNetFormer + Swin-B | `experiments/cei_irsa_m1_ftunetformer_swinb/test/test_metrics.json` |
| IRSA | m2 U-Net + EfficientNet-B4 | `experiments/cei_irsa_m2_unet_effb4/test/test_metrics.json` |
| IRSA | m3 UNetFormer + ResNet-101 | `experiments/cei_irsa_m3_unetformer_r101/test/test_metrics.json` |
| IRSA | m4 UPerNet + Swin-B | `experiments/cei_irsa_m4_upernet_swinb/test/test_metrics.json` |
| IRSA | m5 SegFormer + MiT-B5 | `experiments/cei_irsa_m5_segformer_mitb5/test/test_metrics.json` |
