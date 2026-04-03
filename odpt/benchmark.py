#!/usr/bin/env python3
# benchmark.py
# Multi-seed benchmark runner for the DARP simulation.
#
# Runs every policy combination across multiple seeds, collects per-run
# summary.json files, then computes mean ± std for every metric and writes
# a single aggregated results table.
#
# Usage
# -----
#   python benchmark.py                        # all policies, 5 seeds
#   python benchmark.py --seeds 42 43 44       # specific seeds
#   python benchmark.py --n-seeds 10           # 10 seeds starting from 42
#   python benchmark.py --policies greedy greedy+ts greedy+ga
#   python benchmark.py --no-rl                # skip RL policies (no model needed)
#   python benchmark.py --out benchmark_results
#   python benchmark.py --workers 4            # parallel runs (experimental)
#
# Outputs
# -------
#   benchmark_results/
#     runs/            one summary.json per (policy, seed) run
#     aggregated.json  mean ± std for every metric, every policy
#     aggregated.csv   same, in spreadsheet-friendly format
#     report.txt       human-readable table for thesis appendix

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Optional


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------

# Each entry: (policy_name, model_key_or_None)
# model_key matches MODEL_REGISTRY in main.py
_GREEDY_POLICIES = [
    ("greedy",      None),
    ("greedy+sa",   None),
    ("greedy+ts",   None),
    ("greedy+ga",   None),
    ("greedy+alns", None),
]

_RL_POLICIES = [
    ("rl",        "rl_tuned"),
    ("rl",        "rl_base"),
    ("rl+sa",     "rl_tuned"),
    ("rl+sa",     "rl_base"),
    ("rl+ts",     "rl_tuned"),
    ("rl+ts",     "rl_base"),
    ("rl+ga",     "rl_tuned"),
    ("rl+ga",     "rl_base"),
    ("rl+alns",   "rl_tuned"),
    ("rl+alns",   "rl_base"),
]

MODEL_REGISTRY = {
    "rl_tuned": "rl_outputs/run_008/model.zip",
    "rl_base":  "rl_outputs/run_006/model.zip",
}

# Metrics to aggregate (must match keys in summary["metrics"])
METRIC_KEYS = [
    "total_requests",
    "served",
    "rejected",
    "service_rate",
    "mean_wait",
    "p95_wait",
    "mean_ride",
    "p95_ride",
    "mean_detour_ratio",
    "p95_detour_ratio",
    "total_distance",
    "improvements",
    "mean_latency_ms",
    "p95_latency_ms",
    "violations_total",
    "violations_wait",
    "violations_ride",
    "mean_wait_excess",
    "mean_ride_excess",
]


# ---------------------------------------------------------------------------
# Run descriptor
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    policy:     str
    model_key:  Optional[str]   # None for greedy policies
    seed:       int

    @property
    def label(self) -> str:
        """Human-readable identifier used in output filenames and tables."""
        if self.model_key:
            return f"{self.policy} ({self.model_key})"
        return self.policy

    @property
    def filename_stem(self) -> str:
        """Safe filename stem: policy__modelkey__seed."""
        pol  = self.policy.replace("+", "_plus_")
        mkey = self.model_key or "none"
        return f"{pol}__{mkey}__seed{self.seed}"


# ---------------------------------------------------------------------------
# Single run executor
# ---------------------------------------------------------------------------

def execute_run(spec: RunSpec, out_dir: Path, verbose: bool = False) -> dict:
    """
    Execute one simulation run and return its summary dict.

    Imports main lazily to avoid loading heavy dependencies at module level.
    Each call is fully isolated — SimPy environment, RNGs, and vehicle
    state are all created fresh inside main().
    """
    from config import SimulationConfig
    from main import main as sim_main

    model_path = MODEL_REGISTRY.get(spec.model_key) if spec.model_key else None

    # Validate model file exists before burning time on a run
    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train a model first or skip RL policies with --no-rl"
        )

    cfg = SimulationConfig(
        seed       = spec.seed,
        policy     = spec.policy,
        n_requests = 400,
    )

    t0 = time.time()
    metrics = sim_main(cfg=cfg, model_path=model_path, verbose=verbose, visualize=False)
    elapsed = time.time() - t0

    summary = metrics.summary()
    summary["_run_wall_seconds"] = round(elapsed, 2)
    summary["_policy_label"]     = spec.label
    summary["_seed"]             = spec.seed

    # Save individual run JSON
    run_path = out_dir / f"{spec.filename_stem}.json"
    with open(run_path, "w") as f:
        json.dump({"spec": {"policy": spec.policy,
                             "model_key": spec.model_key,
                             "seed": spec.seed,
                             "label": spec.label},
                   "metrics": summary}, f, indent=2, default=str)

    return summary


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def aggregate(results: dict[str, list[dict]]) -> dict[str, dict]:
    """
    Aggregate per-seed metric dicts into mean ± std per policy label.

    results: { policy_label: [metrics_dict_seed1, metrics_dict_seed2, ...] }
    returns: { policy_label: { metric: {mean, std, n, values} } }
    """
    aggregated = {}

    for label, runs in results.items():
        agg = {}
        for key in METRIC_KEYS:
            values = [r[key] for r in runs if r.get(key) is not None]
            if not values:
                agg[key] = {"mean": None, "std": None, "n": 0, "values": []}
                continue
            m  = mean(values)
            sd = stdev(values) if len(values) > 1 else 0.0
            agg[key] = {
                "mean":   round(m,  4),
                "std":    round(sd, 4),
                "n":      len(values),
                "values": [round(v, 4) for v in values],
            }

        # Wall time (not a metric, just informational)
        wall = [r["_run_wall_seconds"] for r in runs if "_run_wall_seconds" in r]
        agg["_wall_seconds"] = {"mean": round(mean(wall), 1)} if wall else {}

        aggregated[label] = agg

    return aggregated


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(aggregated: dict, path: Path) -> None:
    """Write aggregated results to CSV — one row per policy, mean and std columns."""
    rows = []
    for label, agg in aggregated.items():
        row = {"policy": label}
        for key in METRIC_KEYS:
            entry = agg.get(key, {})
            row[f"{key}_mean"] = entry.get("mean", "")
            row[f"{key}_std"]  = entry.get("std",  "")
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Text report writer
# ---------------------------------------------------------------------------

def write_report(aggregated: dict, seeds: list[int], path: Path) -> None:
    """
    Write a human-readable comparison table for the thesis appendix.
    Shows mean (± std) for the primary thesis metrics.
    """
    PRIMARY = [
        ("service_rate",      "service rate",    ".1%"),
        ("mean_wait",         "mean wait (min)", ".2f"),
        ("p95_wait",          "p95 wait (min)",  ".2f"),
        ("mean_ride",         "mean ride (min)", ".2f"),
        ("mean_detour_ratio", "mean detour",     ".3f"),
        ("total_distance",    "total dist",      ".0f"),
        ("mean_latency_ms",   "mean lat (ms)",   ".1f"),
        ("p95_latency_ms",    "p95 lat (ms)",    ".0f"),
        ("violations_total",  "violations",      ".0f"),
    ]

    def fmt(val, fmtstr):
        if val is None:
            return "n/a"
        if fmtstr == ".1%":
            return f"{val:.1%}"
        return format(val, fmtstr)

    lines = []
    lines.append("=" * 100)
    lines.append("BENCHMARK RESULTS — DARP SIMULATION")
    lines.append(f"Seeds: {seeds}   n_seeds={len(seeds)}")
    lines.append("=" * 100)
    lines.append("")

    # Header
    col_w = 22
    metric_w = 14
    header = f"{'Policy':<{col_w}}"
    for _, col_label, _ in PRIMARY:
        header += f"  {col_label:>{metric_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    for label, agg in aggregated.items():
        row = f"{label:<{col_w}}"
        for key, _, fmtstr in PRIMARY:
            entry = agg.get(key, {})
            m  = entry.get("mean")
            sd = entry.get("std")
            if m is None:
                cell = "n/a"
            elif sd is not None and sd > 0:
                cell = f"{fmt(m, fmtstr)} ±{fmt(sd, fmtstr)}"
            else:
                cell = fmt(m, fmtstr)
            row += f"  {cell:>{metric_w}}"
        lines.append(row)

    lines.append("")
    lines.append("Values shown as mean ± std across seeds.")
    lines.append("Violations = execution-time hard constraint breaches.")
    lines.append("Latency = full decision epoch (insertion + improvement pass).")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Progress printer
# ---------------------------------------------------------------------------

def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total) if total else 0
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    policies:   list[tuple[str, Optional[str]]],
    seeds:      list[int],
    out_root:   str = "benchmark_results",
    verbose:    bool = False,
    stop_on_error: bool = False,
) -> dict:
    """
    Run all (policy, seed) combinations and return aggregated results.

    Parameters
    ----------
    policies    : list of (policy_name, model_key_or_None)
    seeds       : list of integer seeds
    out_root    : root output directory
    verbose     : pass through to sim_main
    stop_on_error : raise on first failure instead of logging and continuing

    Returns
    -------
    aggregated dict from aggregate()
    """
    out_root_path = Path(out_root)
    runs_dir      = out_root_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    specs = [RunSpec(pol, mkey, seed)
             for (pol, mkey) in policies
             for seed in seeds]

    total = len(specs)
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK RUN")
    print(f"{'=' * 60}")
    print(f"  Policies : {len(policies)}")
    print(f"  Seeds    : {seeds}")
    print(f"  Total    : {total} runs")
    print(f"  Output   : {out_root_path.resolve()}")
    print(f"{'=' * 60}\n")

    # Group results by policy label for aggregation
    results: dict[str, list[dict]] = {}
    errors:  list[tuple[RunSpec, str]] = []

    t_total_start = time.time()

    for idx, spec in enumerate(specs):
        print(f"{_bar(idx, total)}  {spec.label}  seed={spec.seed}", end="  ", flush=True)
        t0 = time.time()

        try:
            summary = execute_run(spec, runs_dir, verbose=verbose)
            elapsed = time.time() - t0

            sr   = summary.get("service_rate", 0)
            mw   = summary.get("mean_wait", 0) or 0
            viols = summary.get("violations_total", 0) or 0
            print(f"svc={sr:.1%}  wait={mw:.1f}  viols={viols}  ({elapsed:.0f}s)")

            label = spec.label
            results.setdefault(label, [])
            results[label].append(summary)

        except Exception as exc:
            elapsed = time.time() - t0
            msg = str(exc)
            print(f"ERROR ({elapsed:.0f}s): {msg[:80]}")
            errors.append((spec, traceback.format_exc()))
            if stop_on_error:
                raise

    print(f"\n{_bar(total, total)}  done\n")

    # Summary of errors
    if errors:
        print(f"  {len(errors)} run(s) failed:")
        for spec, tb in errors:
            print(f"    {spec.label} seed={spec.seed}")
            if verbose:
                print(tb)
        print()

    total_elapsed = time.time() - t_total_start
    print(f"  Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Successful runs: {total - len(errors)} / {total}")

    # Aggregate
    aggregated = aggregate(results)

    # Write outputs
    agg_json_path = out_root_path / "aggregated.json"
    with open(agg_json_path, "w") as f:
        json.dump({
            "meta": {
                "seeds":    seeds,
                "n_seeds":  len(seeds),
                "policies": [f"{p} ({m})" if m else p for p, m in policies],
                "n_runs_total":      total,
                "n_runs_successful": total - len(errors),
                "wall_seconds":      round(total_elapsed, 1),
            },
            "results": aggregated,
        }, f, indent=2, default=str)

    agg_csv_path = out_root_path / "aggregated.csv"
    write_csv(aggregated, agg_csv_path)

    report_path = out_root_path / "report.txt"
    write_report(aggregated, seeds, report_path)

    print(f"\n  aggregated.json : {agg_json_path}")
    print(f"  aggregated.csv  : {agg_csv_path}")
    print(f"  report.txt      : {report_path}")
    print(f"  individual runs : {runs_dir}/")

    return aggregated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed DARP benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python benchmark.py                            # all policies, 5 seeds
  python benchmark.py --n-seeds 10              # 10 seeds (42-51)
  python benchmark.py --seeds 42 100 200 300    # specific seeds
  python benchmark.py --no-rl                   # greedy policies only
  python benchmark.py --policies greedy greedy+ts greedy+ga
  python benchmark.py --out my_results          # custom output dir
  python benchmark.py --stop-on-error           # halt on first failure
""",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Explicit seed list (e.g. --seeds 42 43 44 45 46)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of seeds starting from 42 (default: 5)",
    )
    parser.add_argument(
        "--policies", nargs="+", default=None,
        choices=["greedy","greedy+sa","greedy+ts","greedy+ga","greedy+alns",
                 "rl","rl+sa","rl+ts","rl+ga","rl+alns"],
        help="Subset of greedy policies to run (RL variants added separately)",
    )
    parser.add_argument(
        "--no-rl", action="store_true",
        help="Skip all RL policies (no model files required)",
    )
    parser.add_argument(
        "--rl-model", default=None,
        choices=["rl_tuned", "rl_base", "both"],
        help="Which RL model(s) to use for RL policies (default: both)",
    )
    parser.add_argument(
        "--out", default="benchmark_results",
        help="Output directory (default: benchmark_results)",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop immediately on first failed run",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Pass verbose flag to each simulation run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Seeds ---
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = list(range(42, 42 + args.n_seeds))

    # --- Greedy policies ---
    if args.policies:
        greedy_policies = [(p, None) for p in args.policies]
    else:
        greedy_policies = list(_GREEDY_POLICIES)

    # --- RL policies ---
    if args.no_rl:
        rl_policies = []
    else:
        # Filter by requested model(s)
        if args.rl_model == "rl_tuned":
            rl_policies = [(p, m) for p, m in _RL_POLICIES if m == "rl_tuned"]
        elif args.rl_model == "rl_base":
            rl_policies = [(p, m) for p, m in _RL_POLICIES if m == "rl_base"]
        else:
            rl_policies = list(_RL_POLICIES)

        # Filter by requested policy names if --policies was given
        if args.policies:
            rl_base_names = set(args.policies)
            rl_policies = [(p, m) for p, m in rl_policies if p in rl_base_names]

    all_policies = greedy_policies + rl_policies

    if not all_policies:
        print("ERROR: no policies selected. Check --policies and --no-rl flags.")
        sys.exit(1)

    print(f"Policies to run ({len(all_policies)}):")
    for p, m in all_policies:
        label = f"{p} ({m})" if m else p
        model_path = MODEL_REGISTRY.get(m) if m else None
        exists = ""
        if model_path:
            exists = "  [OK]" if os.path.exists(model_path) else "  [MISSING]"
        print(f"  {label}{exists}")

    # Warn about missing models
    missing = [(p, m) for p, m in all_policies
               if m and not os.path.exists(MODEL_REGISTRY.get(m, ""))]
    if missing:
        print(f"\nWARNING: {len(missing)} RL model(s) not found.")
        if not args.stop_on_error:
            print("  Those runs will be skipped (error logged, benchmark continues).")
        print()

    run_benchmark(
        policies      = all_policies,
        seeds         = seeds,
        out_root      = args.out,
        verbose       = args.verbose,
        stop_on_error = args.stop_on_error,
    )


if __name__ == "__main__":
    main()
