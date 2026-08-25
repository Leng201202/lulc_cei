"""Test every trained CEI-comparison checkpoint on the CEI test set.

Five architectures were each trained twice (see configs/cei_oem/Script.md and
configs/cei_irsa/Script.md) -- once on OpenEarthMap, once on IRSAMap, both
remapped into the 7-class CEI taxonomy. This script scores all ten checkpoints
on the one target that matters for comparing them: the 100-tile hand-labeled
CEI test set (``data/CEI_data``, ``test_split.txt``).

That is 10 evaluate.py runs. Each writes its metrics JSON (OA/mIoU/mF1,
per-class IoU, confusion matrix + heatmap) next to the checkpoint that
produced it, using the ``cei_test_tta.json`` filename configs/cei_oem/Script.md
and configs/cei_irsa/Script.md establish. This script additionally prints one
consolidated comparison table and writes
``experiments/full_test_results_<round>.json``.

A run that fails (missing checkpoint, evaluate.py error) is recorded and the
suite continues, so one broken pairing does not hide the rest. A run whose
output file already exists is reused (not re-evaluated) unless --force is
passed, so an interrupted suite can be safely resumed.

Rounds: every invocation writes into its own ``logs/<round>/`` subfolder
instead of overwriting the plain ``logs/cei_test_tta.json`` etc. from a
previous run, so re-testing against an updated dataset never clobbers the
prior round's numbers. --round auto-increments (round_1, round_2, ...) by
scanning existing round folders across all experiments; pass --round
explicitly (e.g. --round after_cei_v2) to name one yourself. The very first
pre-round results some checkpoints already have (plain ``logs/*_test_tta.json``,
from before this feature existed) are left alone either way.

Usage
-----
python pipeline/run_all_tests.py                       # all 10 checkpoints, next auto round
python pipeline/run_all_tests.py --round after_cei_v2   # name this round explicitly
python pipeline/run_all_tests.py --models m2 m5         # subset of architectures
python pipeline/run_all_tests.py --sources oem          # OEM-trained models only
python pipeline/run_all_tests.py --no-tta               # quick check, skip 4-way TTA
python pipeline/run_all_tests.py --dry-run              # print the run plan, execute nothing
python pipeline/run_all_tests.py --force                # re-run even if a metrics file exists
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARCH_LABELS = {
    "m1": "FT-UNetFormer Swin-B",
    "m2": "U-Net EfficientNet-B4",
    "m3": "UNetFormer ResNet-101",
    "m4": "UPerNet Swin-B",
    "m5": "SegFormer MiT-B5",
}

# OEM-trained models, scored on the CEI test set.
OEM_MODELS = {
    "m1": dict(experiment="cei_oem_m1_ftunetformer_swinb",
               cei_config="configs/cei_oem/test/test_m1_ftunetformer_cei.yml"),
    "m2": dict(experiment="cei_oem_m2_unet_effb4",
               cei_config="configs/cei_oem/test/test_m2_uneteffb4_cei.yml"),
    "m3": dict(experiment="cei_oem_m3_unetformer_r101",
               cei_config="configs/cei_oem/test/test_m3_unetformer_cei.yml"),
    "m4": dict(experiment="cei_oem_m4_upernet_swinb",
               cei_config="configs/cei_oem/test/test_m4_upernet_cei.yml"),
    "m5": dict(experiment="cei_oem_m5_segformer_mitb5",
               cei_config="configs/cei_oem/test/test_m5_segformer_cei.yml"),
}

# IRSA-trained models, scored on the same CEI test set.
IRSA_MODELS = {
    "m1": dict(experiment="cei_irsa_m1_ftunetformer_swinb",
               cei_config="configs/cei_irsa/test/test_m1_ftunetformer_on_cei.yml"),
    "m2": dict(experiment="cei_irsa_m2_unet_effb4",
               cei_config="configs/cei_irsa/test/test_m2_uneteffb4_on_cei.yml"),
    "m3": dict(experiment="cei_irsa_m3_unetformer_r101",
               cei_config="configs/cei_irsa/test/test_m3_unetformer_on_cei.yml"),
    "m4": dict(experiment="cei_irsa_m4_upernet_swinb",
               cei_config="configs/cei_irsa/test/test_m4_upernet_on_cei.yml"),
    "m5": dict(experiment="cei_irsa_m5_segformer_mitb5",
               cei_config="configs/cei_irsa/test/test_m5_segformer_on_cei.yml"),
}


def resolve_round(explicit):
    """Auto-increment round_N by scanning every experiment's logs/ folder.

    Explicit --round names bypass this entirely. Otherwise, the highest
    round_N found anywhere under experiments/*/logs/ plus one -- so a fresh
    checkout with no round folders yet starts at round_1, and every later
    invocation lands in its own new folder without needing to track a
    counter anywhere.
    """
    if explicit:
        return explicit
    pattern = os.path.join(REPO_ROOT, "experiments", "*", "logs", "round_*")
    found = [int(m.group(1)) for path in glob.glob(pattern)
            if (m := re.match(r"round_(\d+)$", os.path.basename(path)))]
    return f"round_{max(found) + 1 if found else 1}"


def build_runs(args):
    """Expand the model tables into one CEI-test run dict per model."""
    runs = []

    def add(key, source, experiment, config):
        suffix = "_tta" if args.tta else ""
        runs.append({
            "key": key,
            "label": ARCH_LABELS[key],
            "source": source,
            "target": "cei",
            "config": config,
            "checkpoint": os.path.join("experiments", experiment, "checkpoints",
                                       "best_checkpoint.pth"),
            "output": os.path.join("experiments", experiment, "logs", args.round,
                                   f"cei_test{suffix}.json"),
        })

    if "oem" in args.sources:
        for key, info in OEM_MODELS.items():
            if args.models and key not in args.models:
                continue
            add(key, "oem", info["experiment"], info["cei_config"])

    if "irsa" in args.sources:
        for key, info in IRSA_MODELS.items():
            if args.models and key not in args.models:
                continue
            add(key, "irsa", info["experiment"], info["cei_config"])

    return runs


def tail_output(result, num_lines=12):
    output = (result.stdout or "") + (result.stderr or "")
    lines = [line for line in output.replace("\r", "\n").splitlines() if line.strip()]
    return "\n".join(lines[-num_lines:])


def extract_metrics(metrics):
    return {
        "oa": metrics.get("OA"),
        "miou": metrics.get("mIoU"),
        "mf1": metrics.get("mF1"),
        "per_class_iou": metrics.get("per_class_iou"),
    }


def run_one(run, args):
    checkpoint_path = os.path.join(REPO_ROOT, run["checkpoint"])
    if not os.path.isfile(checkpoint_path):
        return {**run, "status": "no checkpoint", "minutes": 0.0}

    output_path = os.path.join(REPO_ROOT, run["output"])
    if os.path.isfile(output_path) and not args.force:
        with open(output_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        return {**run, "status": "cached", "minutes": 0.0, **extract_metrics(metrics)}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    command = [sys.executable, "evaluate.py",
               "--config", run["config"],
               "--checkpoint", run["checkpoint"],
               "--split", "test",
               "--output", run["output"]]
    if args.tta:
        command.append("--tta")

    started = time.time()
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True,
                            text=True, errors="replace")
    minutes = (time.time() - started) / 60.0

    if result.returncode != 0 or not os.path.isfile(output_path):
        return {**run, "status": "failed", "minutes": minutes,
               "detail": tail_output(result)}

    with open(output_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {**run, "status": "ok", "minutes": minutes, **extract_metrics(metrics)}


def print_run_table(results):
    print(f"\n{'=' * 92}\nPER-RUN RESULTS\n{'=' * 92}")
    print("%-4s %-24s %-6s %-6s %8s %8s %8s %7s  %s"
          % ("KEY", "MODEL", "SRC", "ON", "OA", "mIoU", "mF1", "MIN", "STATUS"))
    for r in results:
        oa = f"{r['oa']:.4f}" if r.get("oa") is not None else "-"
        miou = f"{r['miou']:.4f}" if r.get("miou") is not None else "-"
        mf1 = f"{r['mf1']:.4f}" if r.get("mf1") is not None else "-"
        print("%-4s %-24s %-6s %-6s %8s %8s %8s %7.1f  %s"
              % (r["key"], r["label"][:24], r["source"], r["target"],
                 oa, miou, mf1, r.get("minutes", 0.0), r["status"]))
        if r["status"] == "failed":
            print(f"       {r.get('detail', '').splitlines()[-1] if r.get('detail') else ''}")


def print_comparison_table(results):
    """The headline table: mIoU on CEI per model, OEM-trained vs IRSA-trained."""
    lookup = {(r["key"], r["source"]): r.get("miou") for r in results}
    keys = sorted({r["key"] for r in results})
    if not keys:
        return

    print(f"\n{'=' * 92}\nGENERALIZATION COMPARISON (mIoU on CEI)\n{'=' * 92}")
    print("%-4s %-24s %12s %12s" % ("KEY", "MODEL", "OEM->CEI", "IRSA->CEI"))
    for key in keys:
        oem = lookup.get((key, "oem"))
        irsa = lookup.get((key, "irsa"))
        print("%-4s %-24s %12s %12s"
              % (key, ARCH_LABELS.get(key, key),
                 f"{oem:.4f}" if oem is not None else "-",
                 f"{irsa:.4f}" if irsa is not None else "-"))
    print("\nOEM->CEI vs IRSA->CEI is the cross-dataset comparison the whole "
          "suite exists to answer: which training source generalizes better "
          "to CEI imagery, per architecture.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model keys, e.g. m2 m5. Default: all.")
    parser.add_argument("--sources", nargs="*", default=["oem", "irsa"],
                        choices=["oem", "irsa"],
                        help="Which training sources to test. Default: both.")
    parser.add_argument("--tta", dest="tta", action="store_true", default=True,
                        help="4-way flip TTA (default: on -- Script.md's "
                             "recommendation for any number you report).")
    parser.add_argument("--no-tta", dest="tta", action="store_false",
                        help="Disable TTA for a faster, rougher check.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run evaluate.py even if the output metrics "
                             "file already exists (within this round).")
    parser.add_argument("--round", default=None,
                        help="Name for this round's results, written to "
                             "logs/<round>/. Default: auto-increment "
                             "round_1, round_2, ... across all experiments.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run plan and exit without evaluating.")
    args = parser.parse_args()
    args.round = resolve_round(args.round)

    runs = build_runs(args)
    if not runs:
        raise SystemExit("No runs selected -- check --models/--sources.")

    print(f"Round: {args.round}")
    print(f"{len(runs)} evaluate.py run(s) planned "
          f"(TTA {'on' if args.tta else 'off'}).")
    if args.dry_run:
        for r in runs:
            print(f"  {r['key']} {r['source']}->{r['target']:<4} "
                  f"config={r['config']}  checkpoint={r['checkpoint']}  "
                  f"output={r['output']}")
        return 0

    results = []
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {run['key']} {run['label']} "
              f"({run['source']} -> {run['target']}) ...", flush=True)
        record = run_one(run, args)
        results.append(record)
        if record["status"] in ("ok", "cached"):
            print(f"  {record['status']}: OA {record.get('oa', float('nan')):.4f}  "
                  f"mIoU {record.get('miou', float('nan')):.4f}  "
                  f"mF1 {record.get('mf1', float('nan')):.4f}"
                  + (f"  ({record['minutes']:.1f} min)" if record["status"] == "ok" else ""))
        else:
            print(f"  {record['status']}"
                  + (f": {record.get('detail', '').splitlines()[-1]}" if record.get("detail") else ""))

    print_run_table(results)
    print_comparison_table(results)

    failed = [r for r in results if r["status"] == "failed"]
    missing = [r for r in results if r["status"] == "no checkpoint"]
    if failed:
        print(f"\n{len(failed)} run(s) failed:")
        for r in failed:
            print(f"  {r['key']} {r['source']}->{r['target']}: "
                  f"{r.get('detail', '').splitlines()[-1] if r.get('detail') else ''}")
    if missing:
        print(f"\n{len(missing)} run(s) skipped, no checkpoint found:")
        for r in missing:
            print(f"  {r['key']} {r['source']}->{r['target']}: {r['checkpoint']}")

    out_path = os.path.join(REPO_ROOT, "experiments",
                            f"full_test_results_{args.round}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nFull results: {os.path.relpath(out_path, REPO_ROOT)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
