"""Plot validation-mIoU-per-epoch training curves from training_logs.json.

Reads every ``experiments/cei_{oem,irsa}_m*/logs/training_logs.json`` (smoke
runs excluded) and writes two PNGs per model under ``--out``:

* ``<experiment>.png``      -- val-mIoU curve, marker on every epoch, best epoch
  highlighted;
* ``<experiment>_loss.png`` -- train and val loss curves (train/val keep the
  same two colors on every chart), lowest val loss highlighted.

Curves are capped at ``--max-epochs`` (default 100) so runs of different length
stay comparable; the best epoch is picked within the capped range.

Each model keeps one fixed color everywhere (m1..m5 -> slots 1..5), so the same
architecture is the same hue in every figure.

Usage:
    python tools/plot_training_curves.py [--out outputs/training_curves]
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-5 (validated: CVD-safe adjacent order, light surface).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

ARCH_NAMES = {
    "ftunetformer": "FT-UNetFormer",
    "unet": "U-Net",
    "unetformer": "UNetFormer",
    "upernet": "UPerNet",
    "segformer": "SegFormer",
}
DATASET_TITLES = {"oem": "OEM", "irsa": "IRSA"}


def load_runs(max_epochs):
    """Return {dataset: [run, ...]} sorted m1..m5; each run carries its curve."""
    runs = {"oem": [], "irsa": []}
    for path in sorted(glob.glob("experiments/cei_*_m*/logs/training_logs.json")):
        folder = path.replace("\\", "/").split("/")[1]
        match = re.match(r"cei_(oem|irsa)_m(\d)_([a-z]+)", folder)
        if not match or folder.endswith("_smoke"):
            continue
        dataset, index, arch = match.group(1), int(match.group(2)), match.group(3)
        with open(path, encoding="utf-8") as handle:
            logs = json.load(handle)
        total_epochs = len(logs)
        logs = [entry for entry in logs if entry["epoch"] <= max_epochs]
        best = max(logs, key=lambda entry: entry["val_mIoU"])

        runs[dataset].append({
            "index": index,
            "label": f"m{index} {ARCH_NAMES[arch]}",
            "color": SERIES_COLORS[index - 1],
            "epochs": [entry["epoch"] for entry in logs],
            "mious": [entry["val_mIoU"] for entry in logs],
            "train_losses": [entry["train_loss"] for entry in logs],
            "val_losses": [entry["val_loss"] for entry in logs],
            "best_epoch": best["epoch"],
            "best_miou": best["val_mIoU"],
            "total_epochs": total_epochs,
            "folder": folder,
        })
    for dataset in runs:
        runs[dataset].sort(key=lambda run: run["index"])
    return runs


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_xlabel("Epoch", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("mIoU", color=INK_SECONDARY, fontsize=10)


def plot_single(run, dataset, max_epochs, out_dir):
    figure, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    figure.patch.set_facecolor(SURFACE)
    style_axes(ax)

    ax.plot(run["epochs"], run["mious"], color=run["color"], linewidth=2,
            marker="o", markersize=3.5)
    # Best epoch: a larger dot with a surface ring so it reads over the curve.
    ax.plot(run["best_epoch"], run["best_miou"], "o", markersize=9,
            color=run["color"], markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=5)
    ax.annotate(f"best {run['best_miou']:.4f} @ epoch {run['best_epoch']}",
                (run["best_epoch"], run["best_miou"]),
                textcoords="offset points", xytext=(-8, 10), ha="right",
                fontsize=9, color=INK_PRIMARY)

    title = f"{run['label']}  -  trained on {DATASET_TITLES[dataset]}"
    if run["total_epochs"] > max_epochs:
        title += f"  (first {max_epochs} of {run['total_epochs']} epochs)"
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=12)

    figure.tight_layout()
    path = os.path.join(out_dir, f"{run['folder']}.png")
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_loss(run, dataset, max_epochs, out_dir):
    figure, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    figure.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.set_ylabel("Loss", color=INK_SECONDARY, fontsize=10)

    # Train/val wear the same two hues on every model's chart, so the pair
    # reads identically across all ten figures (color follows the entity).
    ax.plot(run["epochs"], run["train_losses"], color=SERIES_COLORS[0],
            linewidth=2, marker="o", markersize=3.5, label="train loss")

    # Several runs logged val_loss as NaN every epoch; plot what exists.
    valid = [(loss, epoch) for loss, epoch
             in zip(run["val_losses"], run["epochs"]) if loss == loss]
    if valid:
        ax.plot([epoch for _, epoch in valid], [loss for loss, _ in valid],
                color=SERIES_COLORS[1], linewidth=2, marker="o",
                markersize=3.5, label="val loss")
        lowest = min(valid)
        ax.plot(lowest[1], lowest[0], "o", markersize=9, color=SERIES_COLORS[1],
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
        # Fixed spot under the legend: the descending curves never reach it.
        ax.text(0.98, 0.80, f"lowest val loss {lowest[0]:.4f} @ epoch {lowest[1]}",
                transform=ax.transAxes, ha="right", fontsize=9,
                color=INK_PRIMARY)
    else:
        ax.text(0.98, 0.80, "val loss was logged as NaN for this run",
                transform=ax.transAxes, ha="right", fontsize=9,
                color=INK_MUTED)

    title = f"{run['label']}  -  trained on {DATASET_TITLES[dataset]}"
    if run["total_epochs"] > max_epochs:
        title += f"  (first {max_epochs} of {run['total_epochs']} epochs)"
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    # Loss descends left-to-right, so the upper right corner is always clear.
    legend = ax.legend(loc="upper right", fontsize=9, frameon=True,
                       facecolor=SURFACE, edgecolor="none", framealpha=0.9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    figure.tight_layout()
    path = os.path.join(out_dir, f"{run['folder']}_loss.png")
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs/training_curves")
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = load_runs(args.max_epochs)
    written = []
    for dataset, dataset_runs in runs.items():
        for run in dataset_runs:
            written.append(plot_single(run, dataset, args.max_epochs, args.out))
            written.append(plot_loss(run, dataset, args.max_epochs, args.out))
    print("\n".join(written) or "No training logs found.")


if __name__ == "__main__":
    main()
