"""Before/after report for the OEM -> CEI fine-tune, with an Agriculture focus.

Evaluates, on the SAME fixed CEI test set (test_split.txt, via the existing
configs/cei_oem/test/test_m2_uneteffb4_cei.yml -- no new test config needed,
architecture/preprocessing/normalization/class mapping are identical across
all three):

  1. the OEM baseline checkpoint (no CEI fine-tuning)
  2. the first CEI fine-tune (normal random-crop sampling)
  3. the Agriculture-aware CEI fine-tune (class-targeted crop sampling)

then writes: one table -- OA, mIoU, mF1, per-class IoU, per-class F1 --
across all three (whichever have a checkpoint yet); a row-normalized
confusion-matrix heatmap per experiment; an Agriculture-focus summary
(Agriculture IoU, and the Agriculture->Rangeland / Agriculture->Tree
confusion shares that motivated the class-aware sampling change); and
before/after panels -- satellite | ground truth | one column per completed
experiment -- on the CEI test tiles with the most Agriculture ground truth
(i.e. the tiles that actually contain perennial-crop imagery), not arbitrary
ones.

Run configs/cei_oem_finetune/finetune.yml and finetune_agri_aware.yml through
train.py first. Missing checkpoints are reported as pending, not fatal.

Usage
-----
python tools/cei/finetune_report.py
python tools/cei/finetune_report.py --no-tta
python tools/cei/finetune_report.py --force
python tools/cei/finetune_report.py --before-after-tiles maesuai_3,maesuai_7
python tools/cei/finetune_report.py --before-after-n 6
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.dataset_factory import build_dataset               # noqa: E402
from src.datasets.taxonomy import (                                   # noqa: E402
    CEI_CLASS_COLORS, CEI_CLASS_NAMES, CEI_IGNORE_COLOR,
)
from src.datasets.transforms import NORMALIZATIONS, get_normalization_name  # noqa: E402
from src.models.checkpoint import load_model_weights                  # noqa: E402
from src.models.model_factory import build_model                     # noqa: E402
from src.utils.config import load_config                              # noqa: E402
from src.utils.confusion import save_confusion_heatmap               # noqa: E402
from train import select_device                                       # noqa: E402

TEST_CONFIG = "configs/cei_oem/test/test_m2_uneteffb4_cei.yml"

EXPERIMENTS = [
    ("baseline", "OEM baseline", "experiments/cei_oem_m2_unet_effb4"),
    ("finetuned", "CEI fine-tuned (normal sampling)", "experiments/cei_finetune"),
    ("agri_aware", "CEI fine-tuned (Agriculture-aware sampling)", "experiments/cei_finetune_agri_aware"),
]

AGRICULTURE = CEI_CLASS_NAMES.index("Agriculture")
RANGELAND = CEI_CLASS_NAMES.index("Rangeland")
TREE = CEI_CLASS_NAMES.index("Tree")


def tail_output(result, num_lines=12):
    output = (result.stdout or "") + (result.stderr or "")
    lines = [line for line in output.replace("\r", "\n").splitlines() if line.strip()]
    return "\n".join(lines[-num_lines:])


def extract_metrics(metrics):
    return {
        "OA": metrics.get("OA"), "mIoU": metrics.get("mIoU"), "mF1": metrics.get("mF1"),
        "per_class_iou": metrics.get("per_class_iou"),
        "per_class_f1": metrics.get("per_class_f1"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }


def agriculture_confusion(confusion_matrix):
    """Agriculture's row, normalized to % of its true pixels (recall view)."""
    row = confusion_matrix[AGRICULTURE]
    total = sum(row)
    if not total:
        return None
    return {
        "correct_pct": 100.0 * row[AGRICULTURE] / total,
        "to_rangeland_pct": 100.0 * row[RANGELAND] / total,
        "to_tree_pct": 100.0 * row[TREE] / total,
    }


def run_eval(label, exp_dir, args):
    checkpoint = f"{exp_dir}/checkpoints/best_checkpoint.pth"
    checkpoint_full = os.path.join(REPO_ROOT, checkpoint)
    if not os.path.isfile(checkpoint_full):
        return {"label": label, "checkpoint": checkpoint, "status": "pending (not trained yet)"}

    output = f"{exp_dir}/logs/finetune_report_test.json"
    output_full = os.path.join(REPO_ROOT, output)
    if os.path.isfile(output_full) and not args.force:
        with open(output_full, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        return {"label": label, "checkpoint": checkpoint, "output": output,
               "status": "cached", **extract_metrics(metrics)}

    os.makedirs(os.path.dirname(output_full), exist_ok=True)
    command = [sys.executable, "evaluate.py", "--config", TEST_CONFIG,
              "--checkpoint", checkpoint, "--split", "test", "--output", output]
    if args.tta:
        command.append("--tta")
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.isfile(output_full):
        return {"label": label, "checkpoint": checkpoint,
               "status": "failed", "detail": tail_output(result)}

    with open(output_full, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {"label": label, "checkpoint": checkpoint, "output": output,
           "status": "ok", **extract_metrics(metrics)}


def write_metrics_table(results, titles, out_dir):
    names = CEI_CLASS_NAMES
    lines = ["# OEM baseline vs CEI fine-tuning (normal vs Agriculture-aware sampling)", "",
             "## OA / mIoU / mF1", "",
             "| Model | OA | mIoU | mF1 | Status |",
             "| --- | --- | --- | --- | --- |"]
    for r in results:
        oa = f"{r['OA']:.4f}" if r.get("OA") is not None else "-"
        miou = f"{r['mIoU']:.4f}" if r.get("mIoU") is not None else "-"
        mf1 = f"{r['mF1']:.4f}" if r.get("mF1") is not None else "-"
        lines.append(f"| {titles[r['label']]} | {oa} | {miou} | {mf1} | {r['status']} |")

    lines += ["", "## Agriculture focus", "",
             "| Model | Agriculture IoU | Correctly Agriculture | ->Rangeland | ->Tree |",
             "| --- | --- | --- | --- | --- |"]
    for r in results:
        if r.get("confusion_matrix") is None:
            continue
        agri_iou = r["per_class_iou"][AGRICULTURE] if r.get("per_class_iou") else None
        conf = agriculture_confusion(r["confusion_matrix"])
        iou_cell = f"{agri_iou:.3f}" if agri_iou is not None else "-"
        if conf:
            lines.append(f"| {titles[r['label']]} | {iou_cell} | {conf['correct_pct']:.1f}% | "
                         f"{conf['to_rangeland_pct']:.1f}% | {conf['to_tree_pct']:.1f}% |")
        else:
            lines.append(f"| {titles[r['label']]} | {iou_cell} | - | - | - |")

    lines += ["", "## Per-class IoU", "", "| Model | " + " | ".join(names) + " |",
             "| --- | " + " | ".join("---" for _ in names) + " |"]
    for r in results:
        if r.get("per_class_iou") is None:
            continue
        cells = [f"{v:.3f}" if v is not None else "-" for v in r["per_class_iou"]]
        lines.append(f"| {titles[r['label']]} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-class F1", "", "| Model | " + " | ".join(names) + " |",
             "| --- | " + " | ".join("---" for _ in names) + " |"]
    for r in results:
        if r.get("per_class_f1") is None:
            continue
        cells = [f"{v:.3f}" if v is not None else "-" for v in r["per_class_f1"]]
        lines.append(f"| {titles[r['label']]} | " + " | ".join(cells) + " |")

    path = os.path.join(out_dir, "metrics_table.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def write_confusion_heatmaps(results, out_dir):
    for r in results:
        if r.get("confusion_matrix") is None:
            continue
        path = os.path.join(out_dir, f"confusion_{r['label']}.png")
        save_confusion_heatmap(r["confusion_matrix"], path, CEI_CLASS_NAMES,
                               normalize="row", decimals=2, annotate_all=True)


# --------------------------------------------------------------------------- before/after panels

def denormalize(tensor, config):
    name = get_normalization_name(config)
    mean = np.array(NORMALIZATIONS[name]["mean"], dtype=np.float32)
    std = np.array(NORMALIZATIONS[name]["std"], dtype=np.float32)
    array = tensor.permute(1, 2, 0).cpu().numpy() * std + mean
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def decode(mask):
    rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
    rgb[...] = CEI_IGNORE_COLOR
    for index, color in enumerate(CEI_CLASS_COLORS):
        rgb[mask == index] = color
    return rgb


@torch.no_grad()
def predict(model, image, device):
    logits = model(image.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1)[0].cpu().numpy()


def load_eval_model(checkpoint_path, device):
    config = load_config(os.path.join(REPO_ROOT, TEST_CONFIG))
    config["model"]["encoder_weights"] = None
    model = build_model(config).to(device).eval()
    checkpoint = torch.load(os.path.join(REPO_ROOT, checkpoint_path), map_location=device)
    load_model_weights(model, checkpoint)
    return model, config


def pick_agriculture_tiles(dataset, n):
    """Test tiles ranked by Agriculture share of their ground truth -- the
    ones that actually contain the perennial-crop imagery this study targets."""
    shares = []
    for index in range(len(dataset)):
        _, target = dataset[index]
        mask = target.numpy()
        valid = mask != 255
        if not valid.any():
            shares.append(0.0)
            continue
        shares.append((mask == AGRICULTURE).sum() / valid.sum())
    ranked = sorted(range(len(dataset)), key=lambda i: shares[i], reverse=True)
    return [i for i in ranked[:n] if shares[i] > 0]


def before_after_panels(results, titles, out_dir, tile_names, before_after_n, device):
    done = [r for r in results if r["status"] in ("ok", "cached")]
    if not done:
        print("Skipping before/after panels -- no evaluated checkpoints yet.")
        return []

    models, config = [], None
    for r in done:
        model, config = load_eval_model(r["checkpoint"], device)
        models.append((titles[r["label"]], model))

    dataset = build_dataset(config, split="test")
    names_on_disk = [os.path.splitext(n)[0] for n in dataset.file_names]

    if tile_names:
        indices = [names_on_disk.index(t) for t in tile_names if t in names_on_disk]
    else:
        indices = pick_agriculture_tiles(dataset, before_after_n)
        if not indices:
            print("No test tiles contain Agriculture ground truth -- "
                  "falling back to the first tiles.")
            indices = list(range(min(before_after_n, len(dataset))))

    written = []
    for index in indices:
        image, target = dataset[index]
        satellite = denormalize(image, config)
        gt = decode(target.numpy())
        name = names_on_disk[index]

        panels = [(satellite, "Satellite Image"), (gt, "Ground Truth")]
        for title, model in models:
            panels.append((decode(predict(model, image, device)), title))

        fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 5.2))
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        fig.suptitle(name, fontsize=12, y=0.98)

        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=np.array(c) / 255.0,
                                 edgecolor="0.6", linewidth=0.8) for c in CEI_CLASS_COLORS]
        fig.legend(handles, CEI_CLASS_NAMES, loc="lower center", ncol=len(CEI_CLASS_NAMES),
                  frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))

        path = os.path.join(out_dir, f"before_after_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tta", dest="tta", action="store_true", default=True)
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--force", action="store_true",
                        help="Re-run evaluate.py even if a cached result exists.")
    parser.add_argument("--before-after-tiles", default=None,
                        help="Comma-separated test tile names, e.g. maesuai_3,maesuai_7. "
                             "Default: the tiles with the most Agriculture ground truth.")
    parser.add_argument("--before-after-n", type=int, default=4,
                        help="How many before/after tiles when not using --before-after-tiles.")
    parser.add_argument("--out", default="outputs/cei_finetune_report")
    args = parser.parse_args()

    out_dir = os.path.join(REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    titles = {label: title for label, title, _ in EXPERIMENTS}

    results = []
    for label, title, exp_dir in EXPERIMENTS:
        print(f"[{label}] ", end="", flush=True)
        record = run_eval(label, exp_dir, args)
        results.append(record)
        if record["status"] in ("ok", "cached"):
            print(f"{record['status']}: OA {record['OA']:.4f}  mIoU {record['mIoU']:.4f}  "
                  f"mF1 {record['mF1']:.4f}")
        else:
            print(record["status"])

    table_path = write_metrics_table(results, titles, out_dir)
    write_confusion_heatmaps(results, out_dir)

    tile_names = args.before_after_tiles.split(",") if args.before_after_tiles else None
    device = select_device()
    panels = before_after_panels(results, titles, out_dir, tile_names, args.before_after_n, device)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nMetrics table:       {os.path.relpath(table_path, REPO_ROOT)}")
    print(f"Before/after panels: {len(panels)} written to {os.path.relpath(out_dir, REPO_ROOT)}")
    pending = [r["label"] for r in results if r["status"].startswith("pending")]
    if pending:
        print(f"\nStill pending training: {', '.join(pending)}")


if __name__ == "__main__":
    main()
