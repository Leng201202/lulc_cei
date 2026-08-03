"""Plot training curves and find the "best epoch" from a run's training_logs.json.

Why a best epoch exists
-----------------------
Training loss almost always keeps falling the longer you train, but the model's
performance on *held-out* validation data stops improving at some point and can
even get worse -- the model starts memorising the training set (overfitting).
The best epoch is where the validation metric is highest; training past it mostly
burns time for no gain. This is the standard justification for early stopping
(Prechelt 1998; Goodfellow et al. 2016, ch. 7.8).

What "best" means here
----------------------
This repo selects models by **validation mIoU** (train.py saves
best_checkpoint.pth at the epoch with the highest val mIoU), so this script uses
the same criterion. It reports:

* best epoch      -- the epoch with the highest val mIoU (the checkpoint you keep)
* early-stop epoch -- the earliest epoch after which val mIoU did not improve for
                      `--patience` epochs; i.e. where you could have stopped and
                      trained ~this many fewer epochs with essentially the same
                      result. Use it to justify a shorter schedule next time.

Note: in some runs Validation Loss is logged as nan (a known quirk of full-image
validation), so the mIoU curve -- not the loss curve -- is the reliable signal.
The metric is the right thing to select on anyway (val loss can look fine while a
rare class collapses).

Usage
-----
python tools/plot_loss_curve.py --experiment cei_irsa01_irsa2cei
python tools/plot_loss_curve.py --log experiments/cei_irsa_m4_upernet_swinb/logs/training_logs.json
python tools/plot_loss_curve.py --experiment cei_irsa01_irsa2cei --patience 10 --out curve.png
"""

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")  # save to file without needing a display (headless-safe)
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_log_path(args):
    """Return the training_logs.json path from --log or --experiment."""
    if args.log:
        return args.log
    return os.path.join(
        REPO_ROOT, "experiments", args.experiment, "logs", "training_logs.json"
    )


def is_number(x):
    return isinstance(x, (int, float)) and not math.isnan(x)


def find_best_epoch(epochs, mious):
    """Epoch with the highest val mIoU (ignoring missing/nan values)."""
    best_epoch, best_val = None, -1.0
    for e, m in zip(epochs, mious):
        if is_number(m) and m > best_val:
            best_epoch, best_val = e, m
    return best_epoch, best_val


def find_early_stop_epoch(epochs, mious, patience):
    """Earliest epoch after which val mIoU did not beat its running best for
    `patience` consecutive epochs -- the epoch early stopping would have picked."""
    best_val, best_epoch, since_improved = -1.0, None, 0
    for e, m in zip(epochs, mious):
        if not is_number(m):
            continue
        if m > best_val:
            best_val, best_epoch, since_improved = m, e, 0
        else:
            since_improved += 1
            if since_improved >= patience:
                return best_epoch, best_val
    return best_epoch, best_val  # never triggered -- still improving at the end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", help="experiment name under experiments/")
    group.add_argument("--log", help="direct path to a training_logs.json")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="epochs of no val-mIoU improvement before early stopping (default 10)",
    )
    parser.add_argument("--out", help="output PNG path (default: next to the log)")
    args = parser.parse_args()

    log_path = resolve_log_path(args)
    if not os.path.isfile(log_path):
        sys.exit(f"Log not found: {log_path}")

    with open(log_path, encoding="utf-8") as f:
        logs = json.load(f)
    if not logs:
        sys.exit(f"Log is empty: {log_path}")

    epochs = [row["epoch"] for row in logs]
    train_loss = [row.get("train_loss") for row in logs]
    val_loss = [row.get("val_loss") for row in logs]
    val_miou = [row.get("val_mIoU") for row in logs]

    best_epoch, best_miou = find_best_epoch(epochs, val_miou)
    stop_epoch, stop_miou = find_early_stop_epoch(epochs, val_miou, args.patience)

    print(f"Run:            {log_path}")
    print(f"Epochs trained: {len(epochs)}")
    print(f"Best epoch:     {best_epoch}  (val mIoU = {best_miou:.4f})")
    if stop_epoch is not None:
        saved = epochs[-1] - stop_epoch
        print(
            f"Early stop:     epoch {stop_epoch} (val mIoU = {stop_miou:.4f}) with "
            f"patience {args.patience} -- ~{saved} epochs could have been skipped"
        )

    # Two stacked panels sharing the epoch axis: loss on top, val mIoU below.
    fig, (ax_loss, ax_miou) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_loss.plot(epochs, train_loss, label="train loss", color="tab:blue")
    if any(is_number(v) for v in val_loss):
        ax_loss.plot(epochs, val_loss, label="val loss", color="tab:orange")
    else:
        ax_loss.text(
            0.5, 0.5, "val loss is nan in this run\n(use the val mIoU panel below)",
            transform=ax_loss.transAxes, ha="center", va="center", color="gray",
        )
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(os.path.basename(os.path.dirname(os.path.dirname(log_path))))
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    ax_miou.plot(epochs, val_miou, label="val mIoU", color="tab:green")
    if best_epoch is not None:
        ax_miou.axvline(best_epoch, color="red", ls="--", alpha=0.8)
        ax_miou.annotate(
            f"best epoch {best_epoch}\nmIoU {best_miou:.4f}",
            xy=(best_epoch, best_miou),
            xytext=(6, -12), textcoords="offset points", color="red",
        )
    if stop_epoch is not None and stop_epoch != best_epoch:
        ax_miou.axvline(stop_epoch, color="purple", ls=":", alpha=0.7)
    ax_miou.set_xlabel("epoch")
    ax_miou.set_ylabel("val mIoU")
    ax_miou.legend(loc="lower right")
    ax_miou.grid(True, alpha=0.3)

    out_path = args.out or os.path.join(
        os.path.dirname(log_path), "loss_curve.png"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot:     {out_path}")


if __name__ == "__main__":
    main()
