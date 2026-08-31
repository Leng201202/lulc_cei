"""Qualitative CEI report for the best-mIoU model per training source.

For OEM-trained and IRSA-trained CEI-comparison models independently, this
picks whichever of the 5 architectures had the best VALIDATION mIoU during
training. Their ``best_checkpoint.pth`` already *is* that checkpoint --
train.py never keeps a better one, only ``last`` (most recent epoch) and
``best`` (highest val mIoU) -- so no epoch-by-epoch search is needed, just
never point at ``last_checkpoint.pth``. Architecture, preprocessing,
normalization, class mapping and image size all come from the checkpoint's
own CEI test config (configs/cei_oem/test/ or configs/cei_irsa/test/), the
same config evaluate.py uses, so inference matches training exactly.

Inference runs on the CEI test set only (data/CEI_data, test_split.txt --
100 hand-labeled tiles, the common ground for comparing OEM-trained vs
IRSA-trained models). For each selected tile it writes one five-panel PNG:

    Satellite Image | Ground Truth | Prediction | Prediction Overlay | Ground Truth Overlay

plus a row-normalized confusion-matrix heatmap and a report (JSON + Markdown)
naming the model's biggest confusions -- "find most mistake confusion".

Caveat worth knowing: "best" here is *native* validation mIoU (OEM's own val
split for OEM-trained models, IRSA's own val split for IRSA-trained) -- NOT
CEI test performance, and the two can disagree (an architecture can validate
best in-domain yet generalize worse to CEI than a runner-up). Use --select
cei to instead pick whichever architecture scored highest on the CEI test set
in an already-run round from pipeline/run_all_tests.py.

Usage
-----
python tools/cei/qualitative_report.py                        # both sources, worst-12 CEI tiles each
python tools/cei/qualitative_report.py --source oem            # OEM-trained only
python tools/cei/qualitative_report.py --model m2              # force U-Net regardless of val mIoU
python tools/cei/qualitative_report.py --select cei             # pick by CEI mIoU instead of native val mIoU
python tools/cei/qualitative_report.py --select cei --round round_2
python tools/cei/qualitative_report.py --num 20 --order best   # 20 strongest tiles instead of weakest
python tools/cei/qualitative_report.py --tiles maesuai_3,maesuai_42
python tools/cei/qualitative_report.py --no-tta                # faster, rougher
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "pipeline")):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_all_tests import ARCH_LABELS, IRSA_MODELS, OEM_MODELS         # noqa: E402

from src.datasets.dataset_factory import build_dataset                 # noqa: E402
from src.datasets.taxonomy import (                                    # noqa: E402
    CEI_CLASS_COLORS, CEI_CLASS_NAMES, CEI_IGNORE_COLOR,
)
from src.datasets.transforms import NORMALIZATIONS, get_normalization_name  # noqa: E402
from src.metrics.segmentation_metrics import SegmentationMetrics       # noqa: E402
from src.models.checkpoint import load_model_weights                   # noqa: E402
from src.models.model_factory import build_model                      # noqa: E402
from src.utils.config import load_config                               # noqa: E402
from src.utils.confusion import save_confusion_heatmap, top_confusions  # noqa: E402
from train import select_device                                        # noqa: E402

IGNORE_INDEX = 255
MODEL_TABLES = {"oem": OEM_MODELS, "irsa": IRSA_MODELS}
OVERLAY_ALPHA = 0.45


# --------------------------------------------------------------------------- selection

def read_training_log(experiment):
    path = os.path.join(REPO_ROOT, "experiments", experiment, "logs", "training_logs.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        logs = json.load(handle)
    scored = [entry for entry in logs if entry.get("val_mIoU") is not None]
    if not scored:
        return None
    best = max(scored, key=lambda entry: entry["val_mIoU"])
    return {"epoch": best["epoch"], "val_mIoU": best["val_mIoU"],
           "final_epoch": logs[-1]["epoch"]}


def latest_round():
    pattern = os.path.join(REPO_ROOT, "experiments", "*", "logs", "round_*")
    found = [int(m.group(1)) for path in glob.glob(pattern)
            if (m := re.match(r"round_(\d+)$", os.path.basename(path)))]
    if not found:
        raise SystemExit("No round_N folders under experiments/ -- run "
                         "pipeline/run_all_tests.py first, or pass --round.")
    return f"round_{max(found)}"


def best_by_cei_miou(source, round_name):
    summary_path = os.path.join(REPO_ROOT, "experiments",
                                f"full_test_results_{round_name}.json")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"{summary_path} not found -- run pipeline/run_all_tests.py "
                         f"for {round_name} first.")
    with open(summary_path, "r", encoding="utf-8") as handle:
        results = json.load(handle)
    by_key = {r["key"]: r.get("miou") for r in results
             if r.get("source") == source and r.get("miou") is not None}
    if not by_key:
        raise SystemExit(f"No CEI mIoU results for source={source} in {round_name}.")
    return max(by_key, key=by_key.get)


def select_architecture(source, args):
    """Return (key, note) -- note is a human-readable line explaining why."""
    models = MODEL_TABLES[source]
    if args.model:
        if args.model not in models:
            raise SystemExit(f"{args.model!r} is not a valid key for source={source} "
                             f"(expected one of {sorted(models)}).")
        return args.model, "forced via --model"

    if args.select == "cei":
        round_name = args.round or latest_round()
        key = best_by_cei_miou(source, round_name)
        return key, f"best CEI test mIoU in {round_name}"

    scored = {key: read_training_log(info["experiment"]) for key, info in models.items()}
    scored = {key: entry for key, entry in scored.items() if entry is not None}
    if not scored:
        raise SystemExit(f"No training_logs.json found for any {source} model.")
    winner = max(scored, key=lambda key: scored[key]["val_mIoU"])
    entry = scored[winner]
    flag = "" if entry["epoch"] == entry["final_epoch"] else (
        f" (not the final epoch, {entry['final_epoch']})")
    return winner, f"best native val mIoU {entry['val_mIoU']:.4f} @ epoch {entry['epoch']}{flag}"


# --------------------------------------------------------------------------- inference

def denormalize(tensor, config):
    name = get_normalization_name(config)
    mean = np.array(NORMALIZATIONS[name]["mean"], dtype=np.float32)
    std = np.array(NORMALIZATIONS[name]["std"], dtype=np.float32)
    array = tensor.permute(1, 2, 0).cpu().numpy() * std + mean
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def decode(mask):
    """Class-index mask -> RGB via the CEI palette; anything out of range is ignore."""
    rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
    rgb[...] = CEI_IGNORE_COLOR
    for index, color in enumerate(CEI_CLASS_COLORS):
        rgb[mask == index] = color
    return rgb


def blend(satellite, color_mask, alpha=OVERLAY_ALPHA):
    mixed = (1.0 - alpha) * satellite.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def per_tile_miou(prediction, target, num_classes):
    valid = target != IGNORE_INDEX
    prediction, target = prediction[valid], target[valid]
    ious = []
    for class_index in np.unique(target):
        if class_index >= num_classes:
            continue
        p, t = prediction == class_index, target == class_index
        union = np.logical_or(p, t).sum()
        if union:
            ious.append(np.logical_and(p, t).sum() / union)
    return float(np.mean(ious)) if ious else float("nan")


@torch.no_grad()
def predict(model, image, device, tta):
    logits = model(image.unsqueeze(0).to(device))
    if tta:
        for dims in ([3], [2], [2, 3]):
            flipped_input = torch.flip(image.unsqueeze(0).to(device), dims=dims)
            logits = logits + torch.flip(model(flipped_input), dims=dims)
    return torch.argmax(logits, dim=1)[0].cpu().numpy()


# --------------------------------------------------------------------------- figure

def save_panel_figure(path, name, miou, satellite, gt_mask, pred_mask):
    gt_color = decode(gt_mask)
    pred_color = decode(pred_mask)
    panels = [
        (satellite, "Satellite Image"),
        (gt_color, "Ground Truth"),
        (pred_color, "Prediction"),
        (blend(satellite, pred_color), "Prediction Overlay"),
        (blend(satellite, gt_color), "Ground Truth Overlay"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(23, 5.2))
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    miou_text = f"{miou:.3f}" if miou == miou else "n/a"
    fig.suptitle(f"{name}   (per-tile mIoU {miou_text})", fontsize=12, y=0.98)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=np.array(c) / 255.0,
                             edgecolor="0.6", linewidth=0.8) for c in CEI_CLASS_COLORS]
    fig.legend(handles, CEI_CLASS_NAMES, loc="lower center", ncol=len(CEI_CLASS_NAMES),
              frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- per-source run

def run_source(source, args):
    models = MODEL_TABLES[source]
    if args.checkpoint:
        # Direct override: skip training_logs.json auto-selection and point at
        # an exact checkpoint (e.g. one that just finished training under a
        # folder name the registry above doesn't know about yet). Still needs
        # --model to pick the matching architecture/CEI test config.
        if not args.model:
            raise SystemExit("--checkpoint requires --model to pick the matching architecture/config.")
        key = args.model
        info = models[key]
        checkpoint_path = args.checkpoint
        experiment = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))
        note = f"explicit --checkpoint override ({args.checkpoint})"
    else:
        key, note = select_architecture(source, args)
        info = models[key]
        experiment = info["experiment"]
        checkpoint_path = os.path.join("experiments", experiment, "checkpoints", "best_checkpoint.pth")
    config_path = info["cei_config"]

    print(f"\n{'=' * 78}\n{source.upper()} -- selected: {key} {ARCH_LABELS[key]}\n{'=' * 78}")
    print(f"  reason:     {note}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  config:     {config_path}")

    if not os.path.isfile(os.path.join(REPO_ROOT, checkpoint_path)):
        print(f"  SKIPPED -- no checkpoint at {checkpoint_path}")
        return None

    config = load_config(os.path.join(REPO_ROOT, config_path))
    config["model"]["encoder_weights"] = None
    device = select_device()

    dataset = build_dataset(config, split="test")
    num_classes = config["dataset"]["num_classes"]

    model = build_model(config).to(device).eval()
    checkpoint = torch.load(os.path.join(REPO_ROOT, checkpoint_path), map_location=device)
    load_model_weights(model, checkpoint)

    out_dir = os.path.join(REPO_ROOT, args.out, args.round, f"{source}_{key}_{experiment}")
    os.makedirs(out_dir, exist_ok=True)

    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=IGNORE_INDEX)
    predictions, scored = {}, []
    print(f"  scoring {len(dataset)} CEI test tiles (TTA {'on' if args.tta else 'off'}) ...",
          flush=True)
    for index in range(len(dataset)):
        image, target = dataset[index]
        prediction = predict(model, image, device, args.tta)
        target_np = target.numpy()
        metrics.update(prediction, target_np)
        predictions[index] = prediction
        scored.append((index, per_tile_miou(prediction, target_np, num_classes)))
        if (index + 1) % 25 == 0:
            print(f"    {index + 1}/{len(dataset)}", flush=True)

    result = metrics.compute()
    names = CEI_CLASS_NAMES[:num_classes]
    print(f"  CEI test set: OA {result['OA']:.4f}  mIoU {result['mIoU']:.4f}  "
          f"mF1 {result['mF1']:.4f}")

    worst = top_confusions(result["confusion_matrix"], names, k=args.top_confusions)
    print(f"  worst confusions:")
    for share, true_name, predicted in worst:
        print(f"    {true_name:<15} -> {predicted:<15} {share:5.1f}%")

    heatmap_path = os.path.join(out_dir, "confusion_row.png")
    save_confusion_heatmap(result["confusion_matrix"], heatmap_path, names,
                           normalize="row", decimals=2, annotate_all=True)

    tile_names = [os.path.splitext(n)[0] for n in dataset.file_names]
    if args.tiles:
        wanted = set(args.tiles.split(","))
        chosen = [index for index in range(len(dataset)) if tile_names[index] in wanted]
        missing = wanted - {tile_names[i] for i in chosen}
        if missing:
            print(f"  WARNING -- tiles not found: {sorted(missing)}")
    else:
        ranked = sorted(scored, key=lambda pair: (pair[1] if pair[1] == pair[1] else 1e9),
                        reverse=(args.order == "best"))
        chosen = [index for index, _ in ranked[:args.num]]

    score_by_index = dict(scored)
    written = []
    for index in chosen:
        image, target = dataset[index]
        satellite = denormalize(image, config)
        name = tile_names[index]
        path = os.path.join(out_dir, f"{name}.png")
        save_panel_figure(path, name, score_by_index[index], satellite,
                          target.numpy(), predictions[index])
        written.append(path)
    print(f"  wrote {len(written)} tile figure(s) to {os.path.relpath(out_dir, REPO_ROOT)}")

    report = {
        "source": source, "model_key": key, "architecture": ARCH_LABELS[key],
        "selection_reason": note, "experiment": experiment,
        "checkpoint": checkpoint_path, "config": config_path, "tta": args.tta,
        "cei_test": {"OA": result["OA"], "mIoU": result["mIoU"], "mF1": result["mF1"]},
        "worst_confusions": [{"true": t, "predicted": p, "percent_of_true_class": share}
                             for share, t, p in worst],
        "per_tile_mIoU": {tile_names[i]: (None if s != s else s) for i, s in scored},
        "figures_written": [os.path.relpath(p, REPO_ROOT) for p in written],
        "confusion_heatmap": os.path.relpath(heatmap_path, REPO_ROOT),
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_report_md(out_dir, report)
    return report


def write_report_md(out_dir, report):
    lines = [
        f"# CEI qualitative report -- {report['source'].upper()}-trained "
        f"({report['architecture']})",
        "",
        f"- Selected: `{report['model_key']}` -- {report['selection_reason']}",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Config: `{report['config']}`",
        f"- TTA: {report['tta']}",
        "",
        "## CEI test set",
        "",
        f"OA **{report['cei_test']['OA']:.4f}**  "
        f"mIoU **{report['cei_test']['mIoU']:.4f}**  "
        f"mF1 **{report['cei_test']['mF1']:.4f}**",
        "",
        "## Worst confusions (% of the true class)",
        "",
        "| True | Predicted as | Share |",
        "| --- | --- | --- |",
    ]
    for row in report["worst_confusions"]:
        lines.append(f"| {row['true']} | {row['predicted']} | "
                     f"{row['percent_of_true_class']:.1f}% |")
    lines += ["", f"Confusion heatmap: `{os.path.basename(report['confusion_heatmap'])}`",
             "", f"{len(report['figures_written'])} tile figure(s) in this folder."]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", nargs="*", default=["oem", "irsa"],
                        choices=["oem", "irsa"], help="Default: both.")
    parser.add_argument("--select", choices=["val", "cei"], default="val",
                        help="'val' (default) = best native validation mIoU; "
                             "'cei' = best CEI test mIoU from an existing round.")
    parser.add_argument("--round", default=None,
                        help="Round this report belongs to -- figures are written to "
                             "<out>/<round>/, matching pipeline/run_all_tests.py's round "
                             "folders (so re-running never overwrites an earlier round). "
                             "Also which round's CEI mIoU is read when --select cei. "
                             "Default: latest round found under experiments/.")
    parser.add_argument("--model", default=None,
                        help="Force a specific architecture key (m1..m5), "
                             "skipping auto-selection.")
    parser.add_argument("--checkpoint", default=None,
                        help="Exact checkpoint path, bypassing training_logs.json "
                             "auto-selection entirely (e.g. a checkpoint under a "
                             "folder name not yet in OEM_MODELS/IRSA_MODELS). "
                             "Requires --model.")
    parser.add_argument("--num", type=int, default=12, help="Tiles to render per source.")
    parser.add_argument("--order", choices=["worst", "best"], default="worst")
    parser.add_argument("--tiles", default=None,
                        help="Comma-separated tile names, e.g. maesuai_3,maesuai_42. "
                             "Overrides --num/--order.")
    parser.add_argument("--top-confusions", type=int, default=8)
    parser.add_argument("--tta", dest="tta", action="store_true", default=True)
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--out", default="outputs/cei_qualitative")
    args = parser.parse_args()
    args.round = args.round or latest_round()
    print(f"Round: {args.round}")

    for source in args.source:
        run_source(source, args)


if __name__ == "__main__":
    main()
