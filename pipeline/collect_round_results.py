"""Collect one test round's scattered per-model results into one flat folder.

run_all_tests.py writes each model's metrics under its own
experiments/<experiment>/logs/<round>/ -- ten different places to look for a
single round's confusion matrices. This copies everything for one round into
outputs/test_results/<round>/, flat and clearly named
(<model>_<source>2cei_confusion.png / _metrics.json), plus one summary.md
with the same comparison table run_all_tests.py prints, so a round can be
reviewed (or shared) without digging through the experiments/ tree.

Usage
-----
python pipeline/collect_round_results.py --round round_2
python pipeline/collect_round_results.py --round round_2 --out outputs/test_results
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all_tests import ARCH_LABELS, IRSA_MODELS, OEM_MODELS  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_experiments():
    """Yield (key, source, experiment) for every CEI-tested model this project defines."""
    for key, info in OEM_MODELS.items():
        yield key, "oem", info["experiment"]
    for key, info in IRSA_MODELS.items():
        yield key, "irsa", info["experiment"]


def find_run_files(experiment, round_name):
    """Locate this run's CEI metrics JSON and confusion PNG, tta or not."""
    log_dir = os.path.join(REPO_ROOT, "experiments", experiment, "logs", round_name)
    for suffix in ("_tta", ""):
        json_path = os.path.join(log_dir, f"cei_test{suffix}.json")
        if os.path.isfile(json_path):
            png_path = os.path.splitext(json_path)[0] + "_confusion.png"
            return json_path, (png_path if os.path.isfile(png_path) else None)
    return None, None


def load_round_summary(round_name):
    """Prefer the consolidated file run_all_tests.py writes; fall back to
    scanning round folders directly if that run never completed / wasn't full."""
    summary_path = os.path.join(REPO_ROOT, "experiments",
                                f"full_test_results_{round_name}.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    results = []
    for key, source, experiment in model_experiments():
        json_path, _ = find_run_files(experiment, round_name)
        if json_path is None:
            continue
        with open(json_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        results.append({"key": key, "label": ARCH_LABELS[key], "source": source,
                        "target": "cei", "status": "ok", "oa": metrics.get("OA"),
                        "miou": metrics.get("mIoU"), "mf1": metrics.get("mF1")})
    return results


def write_summary_md(results, round_name, out_dir):
    lines = [f"# Test results -- {round_name}", "",
             "## Per-run results (CEI test set)", "",
             "| Key | Model | Source | OA | mIoU | mF1 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        oa = f"{r['oa']:.4f}" if r.get("oa") is not None else "-"
        miou = f"{r['miou']:.4f}" if r.get("miou") is not None else "-"
        mf1 = f"{r['mf1']:.4f}" if r.get("mf1") is not None else "-"
        lines.append(f"| {r['key']} | {r['label']} | {r['source']} | "
                     f"{oa} | {miou} | {mf1} |")

    lookup = {(r["key"], r["source"]): r.get("miou") for r in results}
    keys = sorted({r["key"] for r in results})
    if keys:
        lines += ["", "## Generalization comparison (mIoU on CEI)", "",
                 "| Key | Model | OEM->CEI | IRSA->CEI |",
                 "| --- | --- | --- | --- |"]
        for key in keys:
            oem = lookup.get((key, "oem"))
            irsa = lookup.get((key, "irsa"))
            lines.append(f"| {key} | {ARCH_LABELS.get(key, key)} | "
                         f"{f'{oem:.4f}' if oem is not None else '-'} | "
                         f"{f'{irsa:.4f}' if irsa is not None else '-'} |")

    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", required=True,
                        help="Round name to collect, e.g. round_2 (matches "
                             "--round from run_all_tests.py).")
    parser.add_argument("--out", default="outputs/test_results",
                        help="Base output directory. Default: outputs/test_results.")
    args = parser.parse_args()

    out_dir = os.path.join(REPO_ROOT, args.out, args.round)
    os.makedirs(out_dir, exist_ok=True)

    copied, missing = 0, []
    for key, source, experiment in model_experiments():
        json_path, png_path = find_run_files(experiment, args.round)
        arch_slug = experiment.replace("cei_oem_", "").replace("cei_irsa_", "")
        stem = f"{arch_slug}_{source}2cei"
        if json_path is None:
            missing.append(f"{key} ({source})")
            continue
        shutil.copy2(json_path, os.path.join(out_dir, f"{stem}_metrics.json"))
        if png_path:
            shutil.copy2(png_path, os.path.join(out_dir, f"{stem}_confusion.png"))
        copied += 1

    results = load_round_summary(args.round)
    summary_path = write_summary_md(results, args.round, out_dir)

    print(f"Collected {copied} run(s) into {os.path.relpath(out_dir, REPO_ROOT)}")
    if missing:
        print(f"{len(missing)} run(s) not found for {args.round}: {', '.join(missing)}")
    print(f"Summary: {os.path.relpath(summary_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
