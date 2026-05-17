#!/usr/bin/env python3
"""Run FedLEASE across multiple random seeds and aggregate the results.

Usage:
    python experiments/run_seeds.py                   # default seeds [42, 7, 123]
    python experiments/run_seeds.py --seeds 42 7 123 999
    python experiments/run_seeds.py --preset medium   # smaller scale per-seed

Total wall-clock = len(seeds) × single-run time.
On an RTX A5000 with --preset full: ~50 min × 3 ≈ 2.5 hours.

After all seeds finish, prints:
  - per-seed results (headline acc, macro-F1)
  - mean ± std (95% CI) across seeds
  - per-test-set in-domain breakdown with CIs
And saves everything to outputs/seed_summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci95": 0.0, "n": 1}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    # 95% CI for a small sample using t-distribution z ≈ 1.96 for large n,
    # ≈ 2.776 for n=5, ≈ 4.303 for n=3, ≈ 12.706 for n=2. Use 4.303 for n=3.
    t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
               8: 2.365, 9: 2.306, 10: 2.262}
    t = t_table.get(n, 1.96)
    ci95 = t * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci95": ci95, "n": n}


def _format_metric(stats: Dict[str, float]) -> str:
    if stats["n"] <= 1:
        return f"{stats['mean']:.4f}"
    return f"{stats['mean']:.4f} ± {stats['ci95']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 7, 123],
                        help="List of random seeds (default: 42 7 123)")
    parser.add_argument("--preset", default="full",
                        choices=["debug", "quick", "medium", "full"],
                        help="Training preset (default: full)")
    parser.add_argument("--output_root", default=os.path.join(REPO_ROOT, "outputs"),
                        help="Parent directory for per-seed output folders")
    parser.add_argument("--skip_existing", action="store_true",
                        help="If a seed's results.json already exists, skip its run")
    parser.add_argument("--summary_only", action="store_true",
                        help="Don't run anything — just aggregate existing per-seed results")
    args = parser.parse_args()

    seeds = args.seeds
    per_seed_dirs = {
        s: os.path.join(args.output_root, f"seed_{s}_{args.preset}")
        for s in seeds
    }

    # ------------------------------------------------------------------
    # 1. Run each seed (unless --summary_only)
    # ------------------------------------------------------------------
    if not args.summary_only:
        t_start = time.time()
        for i, seed in enumerate(seeds, 1):
            out_dir = per_seed_dirs[seed]
            results_path = os.path.join(out_dir, "results", "fedlease_results.json")

            if args.skip_existing and os.path.exists(results_path):
                print(f"\n[{i}/{len(seeds)}] seed={seed} — results exist, skipping")
                continue

            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(seeds)}] Running FedLEASE seed={seed} preset={args.preset}")
            print(f"  → output: {out_dir}")
            print(f"{'=' * 70}\n", flush=True)

            cmd = [
                sys.executable, os.path.join(REPO_ROOT, "run.py"),
                "--preset", args.preset,
                "--seed", str(seed),
                "--output_dir", out_dir,
            ]
            # Stream subprocess output live to user
            result = subprocess.run(cmd, cwd=REPO_ROOT)
            if result.returncode != 0:
                print(f"  ✗ seed {seed} failed with exit code {result.returncode}",
                      flush=True)

        total_mins = (time.time() - t_start) / 60
        print(f"\nAll seed runs complete in {total_mins:.1f} min")

    # ------------------------------------------------------------------
    # 2. Aggregate
    # ------------------------------------------------------------------
    per_seed_results: Dict[int, Dict] = {}
    for seed in seeds:
        results_path = os.path.join(per_seed_dirs[seed],
                                    "results", "fedlease_results.json")
        if not os.path.exists(results_path):
            print(f"  ⚠ Missing results for seed {seed}: {results_path}")
            continue
        with open(results_path, "r", encoding="utf-8") as fh:
            per_seed_results[seed] = json.load(fh)

    if not per_seed_results:
        print("\nNo per-seed results found — nothing to aggregate.")
        return

    # ------------------------------------------------------------------
    # 3. Collect headline + per-test-set metrics
    # ------------------------------------------------------------------
    headline_acc:  List[float] = []
    headline_f1:   List[float] = []
    per_test: Dict[str, Dict[str, List[float]]] = {}

    for seed, data in per_seed_results.items():
        fm = data.get("final_metrics", {})
        headline = fm.get("_headline", {})
        if headline:
            headline_acc.append(headline["in_domain_mean_accuracy"])
            headline_f1.append(headline["in_domain_mean_macro_f1"])

        for test_name, test_data in fm.items():
            if test_name.startswith("_"):
                continue
            for split_name in ("in_domain", "cross_domain", "aggregate"):
                agg = test_data.get(split_name, {})
                if not agg:
                    continue
                key = f"{test_name}|{split_name}"
                per_test.setdefault(key, {"acc": [], "f1": []})
                per_test[key]["acc"].append(agg.get("mean_accuracy", 0.0))
                per_test[key]["f1"].append(agg.get("mean_macro_f1", 0.0))

    # ------------------------------------------------------------------
    # 4. Print report
    # ------------------------------------------------------------------
    n_seeds = len(per_seed_results)
    print("\n" + "=" * 70)
    print(f"FedLEASE — {n_seeds}-seed summary ({sorted(per_seed_results.keys())})")
    print("=" * 70)

    print("\nPer-seed headline (in-domain mean across both test sets):")
    print(f"  {'seed':>6} | {'acc':>8} | {'macro-F1':>8}")
    print(f"  {'-' * 6} | {'-' * 8} | {'-' * 8}")
    for seed in sorted(per_seed_results.keys()):
        h = per_seed_results[seed].get("final_metrics", {}).get("_headline", {})
        print(f"  {seed:>6} | "
              f"{h.get('in_domain_mean_accuracy', 0):>8.4f} | "
              f"{h.get('in_domain_mean_macro_f1', 0):>8.4f}")

    print(f"\nHeadline aggregate (mean ± 95% CI):")
    print(f"  in-domain accuracy : {_format_metric(_mean_std(headline_acc))}")
    print(f"  in-domain macro-F1 : {_format_metric(_mean_std(headline_f1))}")

    print(f"\nPer test-set breakdown (mean ± 95% CI):")
    print(f"  {'test set | split':<40} | {'acc':>18} | {'macro-F1':>18}")
    print(f"  {'-' * 40} | {'-' * 18} | {'-' * 18}")
    for key in sorted(per_test.keys()):
        acc_stats = _mean_std(per_test[key]["acc"])
        f1_stats  = _mean_std(per_test[key]["f1"])
        print(f"  {key:<40} | "
              f"{_format_metric(acc_stats):>18} | "
              f"{_format_metric(f1_stats):>18}")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    summary = {
        "seeds": sorted(per_seed_results.keys()),
        "n_seeds": n_seeds,
        "preset": args.preset,
        "headline": {
            "in_domain_mean_accuracy": _mean_std(headline_acc),
            "in_domain_mean_macro_f1": _mean_std(headline_f1),
        },
        "per_test_set": {
            key: {
                "accuracy": _mean_std(v["acc"]),
                "macro_f1": _mean_std(v["f1"]),
            }
            for key, v in per_test.items()
        },
        "per_seed_headline": {
            str(seed): per_seed_results[seed].get("final_metrics", {}).get("_headline", {})
            for seed in sorted(per_seed_results.keys())
        },
    }
    summary_path = os.path.join(args.output_root, "seed_summary.json")
    os.makedirs(args.output_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
