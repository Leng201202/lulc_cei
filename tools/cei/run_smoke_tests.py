"""Run the short train+test smoke suite across every model and summarise it.

For each model this runs the generated smoke config through ``train.py``, then
scores the resulting checkpoint on the 100 CEI tiles with ``evaluate.py``, and
prints one table at the end.

A model that fails is recorded and the suite continues, so one broken
architecture does not hide the status of the other four. The exit code is
non-zero if anything failed, which makes this usable as a pre-flight check
before committing GPU time to the real 200-epoch runs.

What a pass means: the model builds, trains without NaN or OOM at its configured
batch size, produces in-range predictions, and scores on CEI end to end. It says
nothing about final accuracy -- three epochs is far too few to judge that.

Usage
-----
python tools/cei/run_smoke_tests.py
python tools/cei/run_smoke_tests.py --models m2 m5      # subset
python tools/cei/run_smoke_tests.py --skip-test         # training only
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMOKE_DIR = os.path.join("configs", "unet", "cei", "smoke")

# (key, label, train config, test config, experiment name)
MODELS = [
    ("m1", "FT-UNetFormer Swin-B",
     "ftunetformer_swinb_oem2cei_smoke.yml",
     "test_ftunetformer_swinb_oem2cei_smoke.yml",
     "cei_m1_ftunetformer_swinb_smoke"),
    ("m2", "U-Net EfficientNet-B4",
     "unet_effb4_oem2cei_smoke.yml",
     "test_unet_effb4_oem2cei_smoke.yml",
     "cei_m2_unet_effb4_smoke"),
    ("m3", "UNetFormer ResNet-101",
     "unetformer_r101_oem2cei_smoke.yml",
     "test_unetformer_r101_oem2cei_smoke.yml",
     "cei_m3_unetformer_r101_smoke"),
    ("m4", "UPerNet Swin-B",
     "upernet_swinb_oem2cei_smoke.yml",
     "test_upernet_swinb_oem2cei_smoke.yml",
     "cei_m4_upernet_swinb_smoke"),
    ("m5", "SegFormer MiT-B5",
     "segformer_mitb5_oem2cei_smoke.yml",
     "test_segformer_mitb5_oem2cei_smoke.yml",
     "cei_m5_segformer_mitb5_smoke"),
]


def run(command):
    """Run a command, streaming nothing, returning (ok, tail_of_output)."""
    result = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, errors="replace"
    )
    output = (result.stdout or "") + (result.stderr or "")
    # tqdm writes carriage returns; keep only the last few real lines.
    lines = [line for line in output.replace("\r", "\n").splitlines() if line.strip()]
    return result.returncode == 0, "\n".join(lines[-12:])


def train_one(train_config):
    return run([sys.executable, "train.py", "--config",
                os.path.join(SMOKE_DIR, train_config)])


def test_one(test_config, experiment, tta):
    checkpoint = os.path.join("experiments", experiment, "checkpoints",
                              "best_checkpoint.pth")
    if not os.path.isfile(os.path.join(REPO_ROOT, checkpoint)):
        return False, f"no checkpoint at {checkpoint}", None

    output = os.path.join("experiments", experiment, "logs", "cei_smoke_metrics.json")
    command = [sys.executable, "evaluate.py",
               "--config", os.path.join(SMOKE_DIR, test_config),
               "--checkpoint", checkpoint,
               "--split", "test",
               "--output", output]
    if tta:
        command.append("--tta")

    ok, tail = run(command)
    metrics = None
    metrics_path = os.path.join(REPO_ROOT, output)
    if ok and os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    return ok, tail, metrics


def training_summary(experiment):
    """Best OEM validation mIoU and final train loss from the run's log."""
    path = os.path.join(REPO_ROOT, "experiments", experiment, "logs",
                        "training_logs.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        logs = json.load(handle)
    if not logs:
        return None
    scored = [entry for entry in logs if entry.get("val_mIoU") is not None]
    best = max(scored, key=lambda e: e["val_mIoU"]) if scored else None
    return {
        "epochs": len(logs),
        "first_loss": logs[0].get("train_loss"),
        "last_loss": logs[-1].get("train_loss"),
        "best_val_miou": best["val_mIoU"] if best else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model keys, e.g. m2 m5. Default: all.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Train only; do not score on CEI.")
    parser.add_argument("--tta", action="store_true",
                        help="Use 4-way TTA when scoring on CEI.")
    args = parser.parse_args()

    selected = [m for m in MODELS if args.models is None or m[0] in args.models]
    if not selected:
        raise SystemExit(f"No models matched {args.models}")

    results = []

    for key, label, train_config, test_config, experiment in selected:
        print(f"\n{'=' * 70}\n{key.upper()}  {label}\n{'=' * 70}")

        started = time.time()
        print("  training ...", flush=True)
        trained, tail = train_one(train_config)
        train_minutes = (time.time() - started) / 60.0

        if not trained:
            print(f"  TRAIN FAILED\n{tail}")
            results.append({"key": key, "label": label, "status": "train failed",
                            "detail": tail.splitlines()[-1] if tail else "",
                            "minutes": train_minutes})
            continue

        summary = training_summary(experiment) or {}
        print(f"  trained {summary.get('epochs', '?')} epochs in "
              f"{train_minutes:.1f} min, "
              f"loss {summary.get('first_loss', float('nan')):.3f} -> "
              f"{summary.get('last_loss', float('nan')):.3f}, "
              f"best OEM val mIoU {summary.get('best_val_miou', float('nan')):.4f}")

        record = {"key": key, "label": label, "status": "ok",
                  "minutes": train_minutes, **summary}

        if not args.skip_test:
            print("  testing on CEI ...", flush=True)
            tested, tail, metrics = test_one(test_config, experiment, args.tta)
            if not tested or metrics is None:
                print(f"  TEST FAILED\n{tail}")
                record["status"] = "test failed"
                record["detail"] = tail.splitlines()[-1] if tail else ""
            else:
                record["cei_miou"] = metrics.get("mIoU")
                record["cei_oa"] = metrics.get("OA")
                record["per_class_iou"] = metrics.get("per_class_iou")
                print(f"  CEI mIoU {record['cei_miou']:.4f}  OA {record['cei_oa']:.4f}")

        results.append(record)

    # ------------------------------------------------------------------ summary
    print(f"\n\n{'=' * 78}\nSMOKE SUITE SUMMARY\n{'=' * 78}")
    print("%-4s %-24s %8s %10s %10s %9s  %s"
          % ("KEY", "MODEL", "MIN", "TRAIN LOSS", "OEM mIoU", "CEI mIoU", "STATUS"))
    for record in results:
        loss = ("%.3f->%.3f" % (record["first_loss"], record["last_loss"])
                if record.get("first_loss") is not None else "-")
        oem = ("%.4f" % record["best_val_miou"]
               if record.get("best_val_miou") is not None else "-")
        cei = "%.4f" % record["cei_miou"] if record.get("cei_miou") is not None else "-"
        print("%-4s %-24s %8.1f %10s %10s %9s  %s"
              % (record["key"], record["label"][:24], record.get("minutes", 0.0),
                 loss, oem, cei, record["status"]))

    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        print(f"\n{len(failed)} model(s) failed:")
        for record in failed:
            print(f"  {record['key']}: {record['status']} -- {record.get('detail', '')}")

    print("\nThree epochs only -- this checks the pipeline runs, not how good "
          "each model is.")

    out = os.path.join(REPO_ROOT, "experiments", "smoke_suite_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Full results: {os.path.relpath(out, REPO_ROOT)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
