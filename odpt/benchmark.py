#!/usr/bin/env python3
# benchmark.py
# Multi-seed benchmark runner for the DARP simulation.
#
# Runs every policy combination across multiple seeds, collects per-run
# summary.json files, then computes mean ± std for every metric and writes
# a structured results report.
#
# Policy matrix
# -------------
# Greedy family  : greedy, greedy+sa, greedy+ts, greedy+ga, greedy+alns
# RL base        : rl, rl+sa, rl+ts, rl+ga, rl+alns  (untuned model)
# RL v3          : rl, rl+sa, rl+ts, rl+ga, rl+alns  (standalone-objective tune)
# RL v4          : rl, rl+sa, rl+ts, rl+ga, rl+alns  (TS-initialiser tune — champion)
#
# Usage
# -----
#   python benchmark.py                          # all policies, 5 seeds
#   python benchmark.py --n-seeds 10
#   python benchmark.py --seeds 42 43 44 45 46
#   python benchmark.py --no-rl                  # greedy family only
#   python benchmark.py --no-greedy              # RL models only (all five)
#   python benchmark.py --rl-model rl_v4         # only v4 RL policies (+greedy)
#   python benchmark.py --rl-model rl_v4 --no-greedy  # only v4 RL, no greedy
#   python benchmark.py --rl-model rl_v4 rl_base # subset of RL models
#   python benchmark.py --out thesis_benchmark
#   python benchmark.py --stop-on-error
#
# Outputs
# -------
#   benchmark_results/
#     runs/            one summary.json per (policy, seed) run
#     aggregated.json  mean ± std for every metric, every policy
#     aggregated.csv   same, in spreadsheet-friendly format
#     report.txt       human-readable thesis tables (service, operational,
#                      worst-case seed, constraint compliance)

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# ---------------------------------------------------------------------------
# !! USER ACTION REQUIRED — update these paths to your actual model files !!
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # Untuned RL (run_006 from rl_train.py — no Optuna tuning)
    "rl_base": "rl_outputs/run_006/model.zip",

    # v3 tune — standalone-objective model (best v3 Optuna trial)
    # Update this path to your best v3 training run.
    "rl_v3":   "rl_outputs/run_008/model.zip",

    # v3 tune — anticipatory features
    "rl_v3ant": "rl_outputs/run_012/checkpoints/best/model.zip",

    # v4 tune — TS-initialiser model, trained via rl_train_from_tune.py.
    # Use model_final.zip — confirmed from tfevents that the best callback
    # score (8.083) was recorded at step 999,960 = the final step, meaning
    # model_final.zip and checkpoints/best/ are the same weights.
    # model_final.zip is the cleaner reference.
    # !! UPDATE THIS PATH to your v4 run directory.
    "rl_v4":   "rl_outputs/run_009/model_final.zip",

    # v5 tune
    "rl_v5":   "rl_outputs/run_011/checkpoints/best/model.zip",

    # v3 new — new congestion factors v3 model
    "rl_v3new": "rl_outputs/run_013/model.zip",

    #v3ant new — new congestion factors v3ant model
    "rl_v3ant_new": "rl_outputs/run_014/model.zip",

    "rl_v4ant": "rl_outputs/run_015/checkpoints/best/model.zip",

    "rl_v5ant": "rl_outputs/run_016/checkpoints/best/model.zip",

    "rl_v6": "rl_outputs/run_017/checkpoints/best/model.zip",
}


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------

_GREEDY_POLICIES = [
    ("greedy",      None),
    ("greedy+sa",   None),
    ("greedy+ts",   None),
    ("greedy+ga",   None),
    ("greedy+alns", None),
]

# All three RL models × all five policy variants.
# Produces 15 RL policies + 5 greedy = 20 total × n_seeds runs.
_RL_POLICIES = [
    # ---- Standalone (no post-processing) ----
    ("rl",     "rl_base"),
    ("rl",     "rl_v3"),
    ("rl",     "rl_v3ant"),
    ("rl",     "rl_v4"),
    ("rl",     "rl_v5"),
    #("rl",     "rl_v3new"),
    #("rl",     "rl_v3ant_new"),
    ("rl",     "rl_v4ant"),
    ("rl",     "rl_v5ant"),
    ("rl",     "rl_v6")
    # ---- RL + Simulated Annealing ----
    ("rl+sa",  "rl_base"),
    ("rl+sa",  "rl_v3"),
    ("rl+sa",  "rl_v3ant"),
    ("rl+sa",  "rl_v4"),
    ("rl+sa",  "rl_v5"),
    #("rl+sa",  "rl_v3new"),
    #("rl+sa",  "rl_v3ant_new"),
    ("rl+sa",  "rl_v4ant"),
    ("rl+sa",  "rl_v5ant"),
    ("rl+sa",  "rl_v6"),
    # ---- RL + Tabu Search (primary hybrid) ----
    ("rl+ts",  "rl_base"),
    ("rl+ts",  "rl_v3"),
    ("rl+ts",  "rl_v3ant"),
    ("rl+ts",  "rl_v4"),
    ("rl+ts",  "rl_v5"),
    #("rl+ts",  "rl_v3new"),
    #("rl+ts",  "rl_v3ant_new"),
    ("rl+ts",  "rl_v4ant"),
    ("rl+ts",  "rl_v5ant"),
    ("rl+ts",  "rl_v6"),
    # ---- RL + Genetic Algorithm ----
    ("rl+ga",  "rl_base"),
    ("rl+ga",  "rl_v3"),
    ("rl+ga",  "rl_v3ant"),
    ("rl+ga",  "rl_v4"),
    ("rl+ga",  "rl_v5"),
    #("rl+ga",  "rl_v3new"),
    #("rl+ga",  "rl_v3ant_new"),
    ("rl+ga",  "rl_v4ant"),
    ("rl+ga",  "rl_v5ant"),
    ("rl+ga",  "rl_v6"),
    # ---- RL + ALNS ----
    ("rl+alns","rl_base"),
    ("rl+alns","rl_v3"),
    ("rl+alns","rl_v3ant"),
    ("rl+alns","rl_v4"),
    ("rl+alns","rl_v5"),
    #("rl+alns","rl_v3new"),
    #("rl+alns","rl_v3ant_new"),
    ("rl+alns","rl_v4ant"),
    ("rl+alns","rl_v5ant"),
    ("rl+alns",  "rl_v6"),
]

# All metric keys produced by metrics.MetricsCollector.summary().
# Must stay in sync with that file.
METRIC_KEYS = [
    # Demand
    "total_requests",
    "served",
    "rejected",
    "service_rate",
    # Wait time
    "mean_wait",
    "p50_wait",
    "p95_wait",
    "max_wait",
    "min_wait",
    "std_wait",
    # Ride time
    "mean_ride",
    "p95_ride",
    # Detour
    "mean_detour_ratio",
    "p95_detour_ratio",
    # Fleet efficiency
    "total_distance",
    "loaded_distance",
    "empty_distance",
    "deadhead_ratio",
    # Algorithm
    "improvements",
    "mean_latency_ms",
    "p95_latency_ms",
    # Constraint compliance
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
    policy:    str
    model_key: Optional[str]
    seed:      int

    @property
    def label(self) -> str:
        if self.model_key:
            return f"{self.policy} ({self.model_key})"
        return self.policy

    @property
    def filename_stem(self) -> str:
        pol  = self.policy.replace("+", "_plus_")
        mkey = self.model_key or "none"
        return f"{pol}__{mkey}__seed{self.seed}"


# ---------------------------------------------------------------------------
# Single run executor
# ---------------------------------------------------------------------------

def execute_run(spec: RunSpec, out_dir: Path,
                verbose: bool = False,
                sim_params: dict = None) -> dict:
    """
    Execute one simulation run and return its summary dict.

    sim_params : dict of SimulationConfig overrides for sensitivity runs.
                 Any key accepted by SimulationConfig can be passed.
                 Example: {"fleet_size": 4, "n_requests": 280}
    """
    from config import SimulationConfig
    from main import main as sim_main
    import rl_env

    # Set observation flags based on model.
    # v6: 12 per-vehicle features (USE_V6_FEATURES=True), no anticipatory features.
    # ant variants: 4 extra global anticipatory features.
    rl_env.USE_V6_FEATURES = (spec.model_key == "rl_v6")
    rl_env.USE_ANTICIPATORY_FEATURES = (spec.model_key in ("rl_v3ant", "rl_v3ant_new", "rl_v4ant", "rl_v5ant"))

    model_path = MODEL_REGISTRY.get(spec.model_key) if spec.model_key else None

    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Update MODEL_REGISTRY in benchmark.py, or skip RL with --no-rl"
        )

    base = {
        "seed":       spec.seed,
        "policy":     spec.policy,
        "n_requests": 400,
    }
    if sim_params:
        base.update(sim_params)

    cfg = SimulationConfig(**base)

    t0 = time.time()
    metrics = sim_main(cfg=cfg, model_path=model_path, verbose=verbose, visualize=False)
    elapsed = time.time() - t0

    summary = metrics.summary()
    summary["_run_wall_seconds"] = round(elapsed, 2)
    summary["_policy_label"]     = spec.label
    summary["_seed"]             = spec.seed
    summary["_sim_params"]       = base   # record what was varied

    run_path = out_dir / f"{spec.filename_stem}.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump({
            "spec":       {"policy": spec.policy,
                           "model_key": spec.model_key,
                           "seed": spec.seed,
                           "label": spec.label},
            "sim_params": base,
            "metrics":    summary,
        }, f, indent=2, default=str)

    return summary


# ---------------------------------------------------------------------------
# Aggregator — mean ± std across seeds, plus per-seed values for worst-case
# ---------------------------------------------------------------------------

def aggregate(results: dict[str, list[dict]]) -> dict[str, dict]:
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
                # Worst-case for robustness analysis
                "worst":  round(max(values), 4) if values else None,
                "best":   round(min(values), 4) if values else None,
            }

        # Per-seed breakdown (for worst-case table in report)
        per_seed = {}
        for run in runs:
            seed = run.get("_seed")
            if seed is not None:
                per_seed[seed] = {
                    k: round(run[k], 4) if isinstance(run.get(k), float) else run.get(k)
                    for k in METRIC_KEYS if run.get(k) is not None
                }
        agg["_per_seed"] = per_seed

        wall = [r["_run_wall_seconds"] for r in runs if "_run_wall_seconds" in r]
        agg["_wall_seconds"] = {"mean": round(mean(wall), 1)} if wall else {}

        aggregated[label] = agg

    return aggregated


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(aggregated: dict, path: Path) -> None:
    rows = []
    for label, agg in aggregated.items():
        row = {"policy": label}
        for key in METRIC_KEYS:
            entry = agg.get(key, {})
            row[f"{key}_mean"]  = entry.get("mean",  "")
            row[f"{key}_std"]   = entry.get("std",   "")
            row[f"{key}_worst"] = entry.get("worst", "")
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Report writer — three structured tables for thesis appendix
# ---------------------------------------------------------------------------

def write_report(aggregated: dict, seeds: list[int], path: Path) -> None:
    """
    Write a structured human-readable report for the thesis appendix.

    Table 1 — Service quality    : service rate, wait (mean/p50/p95/max/σ), ride, detour
    Table 2 — Operational         : VKT, deadhead ratio, improvements, decision latency
    Table 3 — Worst-case seeds    : mean_wait per seed for every policy (robustness)
    Table 4 — Constraint compliance: violations
    """

    def fmt(val, fmtstr):
        if val is None:
            return "n/a"
        if fmtstr == ".1%":
            return f"{val:.1%}"
        return format(val, fmtstr)

    def cell(agg, key, fmtstr, show_std=True):
        entry = agg.get(key, {})
        m  = entry.get("mean")
        sd = entry.get("std")
        if m is None:
            return "n/a"
        s = fmt(m, fmtstr)
        if show_std and sd is not None and sd > 0.001:
            s += f" ±{fmt(sd, fmtstr)}"
        return s

    def cell_worst(agg, key, fmtstr):
        entry = agg.get(key, {})
        w = entry.get("worst")
        return fmt(w, fmtstr) if w is not None else "n/a"

    lines = []

    def ruler(n):
        return "=" * n

    def sub_ruler(n):
        return "-" * n

    lines += [
        ruler(110),
        "BENCHMARK RESULTS — DARP SIMULATION (Malta On Demand)",
        f"Seeds: {seeds}   n_seeds={len(seeds)}",
        "Values shown as mean ± std_cross_seed unless noted.",
        ruler(110),
        "",
    ]

    # ── Table 1: Service quality ─────────────────────────────────────────────
    T1 = [
        ("service_rate",     "svc rate",    ".1%"),
        ("mean_wait",        "wait mean",   ".2f"),
        ("p50_wait",         "wait p50",    ".2f"),
        ("p95_wait",         "wait p95",    ".2f"),
        ("max_wait",         "wait max",    ".2f"),
        ("min_wait",         "wait min",    ".2f"),
        ("std_wait",         "wait σ",      ".2f"),
        ("mean_ride",        "ride mean",   ".2f"),
        ("p95_ride",         "ride p95",    ".2f"),
        ("mean_detour_ratio","detour mean", ".3f"),
        ("p95_detour_ratio", "detour p95",  ".3f"),
        ("rejected",         "rejected",    ".1f"),
    ]
    col_w = 26; m_w = 12
    lines.append("TABLE 1 — SERVICE QUALITY  (wait/ride in minutes)")
    lines.append(sub_ruler(col_w + (m_w + 2) * len(T1)))
    hdr = f"{'Policy':<{col_w}}" + "".join(f"  {c:>{m_w}}" for _, c, _ in T1)
    lines += [hdr, sub_ruler(len(hdr))]
    for label, agg in aggregated.items():
        row = f"{label:<{col_w}}"
        for key, _, fmtstr in T1:
            show_std = key in ("mean_wait", "service_rate", "rejected")
            row += f"  {cell(agg, key, fmtstr, show_std):>{m_w}}"
        lines.append(row)
    lines += [
        "",
        "  std_cross_seed shown for mean_wait, service_rate, rejected only.",
        "  wait σ = within-run standard deviation across passengers (equity metric).",
        "  wait min = best-case passenger experience; detour p95 = tail comfort bound.",
        "",
    ]

    # ── Table 2: Operational efficiency ─────────────────────────────────────
    T2 = [
        ("total_distance",  "total VKT",  ".1f"),
        ("loaded_distance", "loaded VKT", ".1f"),
        ("empty_distance",  "empty VKT",  ".1f"),
        ("deadhead_ratio",  "deadhead %", ".1%"),
        ("improvements",    "improvements",".1f"),
        ("mean_latency_ms", "lat mean ms",".1f"),
        ("p95_latency_ms",  "lat p95 ms", ".0f"),
    ]
    lines.append("TABLE 2 — OPERATIONAL EFFICIENCY")
    lines.append(sub_ruler(col_w + (m_w + 2) * len(T2)))
    hdr = f"{'Policy':<{col_w}}" + "".join(f"  {c:>{m_w}}" for _, c, _ in T2)
    lines += [hdr, sub_ruler(len(hdr))]
    for label, agg in aggregated.items():
        row = f"{label:<{col_w}}"
        for key, _, fmtstr in T2:
            row += f"  {cell(agg, key, fmtstr, show_std=False):>{m_w}}"
        lines.append(row)
    lines += [
        "",
        "  VKT = vehicle-kilometres travelled (minutes proxy — same units as simulation).",
        "  Deadhead = fraction of total distance driven with no passengers aboard.",
        "  Improvements = successful metaheuristic route-improvement steps.",
        "  Latency = full dispatch decision epoch (insertion + improvement pass).",
        "",
    ]

    # ── Table 3: Worst-case seed robustness (mean_wait per seed) ────────────
    lines.append("TABLE 3 — WORST-CASE SEED ANALYSIS  (mean_wait per seed, minutes)")
    seed_cols = sorted(seeds)
    s_w = 10
    lines.append(sub_ruler(col_w + (s_w + 2) * len(seed_cols) + s_w + 4))
    hdr = (f"{'Policy':<{col_w}}"
           + "".join(f"  {'s'+str(s):>{s_w}}" for s in seed_cols)
           + f"  {'WORST':>{s_w}}")
    lines += [hdr, sub_ruler(len(hdr))]
    for label, agg in aggregated.items():
        per_seed = agg.get("_per_seed", {})
        mw_entry = agg.get("mean_wait", {})
        row = f"{label:<{col_w}}"
        worst_val = None
        for s in seed_cols:
            v = per_seed.get(s, {}).get("mean_wait")
            row += f"  {fmt(v, '.2f'):>{s_w}}"
            if v is not None:
                worst_val = max(worst_val, v) if worst_val is not None else v
        row += f"  {fmt(worst_val, '.2f'):>{s_w}}"
        lines.append(row)
    lines += [
        "",
        "  WORST = highest mean_wait observed across any single seed.",
        "  A robust policy shows low variance across this row.",
        "",
    ]

    # ── Table 4: Constraint compliance ──────────────────────────────────────
    T4 = [
        ("violations_total", "violations", ".0f"),
        ("violations_wait",  "wait viols", ".0f"),
        ("violations_ride",  "ride viols", ".0f"),
        ("mean_wait_excess", "wait excess",".2f"),
        ("mean_ride_excess", "ride excess",".2f"),
    ]
    lines.append("TABLE 4 — CONSTRAINT COMPLIANCE")
    lines.append(sub_ruler(col_w + (m_w + 2) * len(T4)))
    hdr = f"{'Policy':<{col_w}}" + "".join(f"  {c:>{m_w}}" for _, c, _ in T4)
    lines += [hdr, sub_ruler(len(hdr))]
    for label, agg in aggregated.items():
        row = f"{label:<{col_w}}"
        for key, _, fmtstr in T4:
            row += f"  {cell(agg, key, fmtstr, show_std=False):>{m_w}}"
        lines.append(row)
    lines += [
        "",
        "  Violations = execution-time hard constraint breaches (target: 0).",
        "  Wait/ride excess = mean overshoot in minutes when a violation occurs.",
        "",
        ruler(110),
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Progress bar (tqdm with plain fallback)
# ---------------------------------------------------------------------------

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

def _bar(done: int, total: int, width: int = 30) -> str:
    """Fallback ASCII bar (used only when tqdm is unavailable)."""
    filled = int(width * done / total) if total else 0
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    policies:      list[tuple[str, Optional[str]]],
    seeds:         list[int],
    out_root:      str  = "benchmark_results",
    verbose:       bool = False,
    stop_on_error: bool = False,
    sim_params:    dict = None,
) -> dict:
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
    if sim_params:
        print(f"  Overrides: {sim_params}")
    print(f"  Output   : {out_root_path.resolve()}")
    print(f"{'=' * 60}\n")

    results: dict[str, list[dict]] = {}
    errors:  list[tuple[RunSpec, str]] = []
    t_total_start = time.time()

    iterator = (
        _tqdm(specs, desc="Benchmark", unit="run", dynamic_ncols=True)
        if _HAS_TQDM else specs
    )

    for idx, spec in enumerate(iterator):
        if _HAS_TQDM:
            iterator.set_description(f"{spec.label}  seed={spec.seed}")
        else:
            print(f"{_bar(idx, total)}  {spec.label}  seed={spec.seed}", end="  ", flush=True)

        t0 = time.time()

        try:
            summary = execute_run(spec, runs_dir, verbose=verbose,
                                   sim_params=sim_params)
            elapsed = time.time() - t0

            sr    = summary.get("service_rate", 0)
            mw    = summary.get("mean_wait",    0) or 0
            dh    = summary.get("deadhead_ratio", 0) or 0
            viols = summary.get("violations_total", 0) or 0

            result_str = f"svc={sr:.1%}  wait={mw:.2f}  dh={dh:.1%}  viols={viols}  ({elapsed:.0f}s)"
            if _HAS_TQDM:
                iterator.set_postfix_str(result_str)
            else:
                print(result_str)

            label = spec.label
            results.setdefault(label, [])
            results[label].append(summary)

        except Exception as exc:
            elapsed = time.time() - t0
            err_str = f"ERROR ({elapsed:.0f}s): {str(exc)[:80]}"
            if _HAS_TQDM:
                iterator.write(f"  x {spec.label} seed={spec.seed} -- {err_str}")
                iterator.set_postfix_str(err_str)
            else:
                print(err_str)
            errors.append((spec, traceback.format_exc()))
            if stop_on_error:
                raise

    print(f"\n{_bar(total, total)}  done\n")

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

    aggregated = aggregate(results)

    agg_json_path = out_root_path / "aggregated.json"
    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "seeds":             seeds,
                "n_seeds":           len(seeds),
                "policies":          [f"{p} ({m})" if m else p for p, m in policies],
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
  python benchmark.py                                    # baseline, all policies, 5 seeds
  python benchmark.py --n-seeds 3 --fleet-size 4        # fleet sensitivity
  python benchmark.py --n-seeds 3 --n-requests 280      # demand intensity (low)
  python benchmark.py --n-seeds 3 --capacity 8          # vehicle capacity sensitivity
  python benchmark.py --n-seeds 3 --max-wait 15         # tighter service constraint
  python benchmark.py --n-seeds 3 --demand-profile uniform
  python benchmark.py --rl-model rl_v4 --n-seeds 5      # v4 + greedy, 5 seeds
  python benchmark.py --rl-model rl_v4 --no-greedy --n-seeds 5  # v4 only, no greedy
  python benchmark.py --no-rl --n-seeds 5               # greedy family only
  python benchmark.py --no-greedy --n-seeds 5           # all RL models, no greedy
  python benchmark.py --out results/fleet_4 --stop-on-error
""",
    )
    # ---- Seeds ----
    parser.add_argument("--seeds",   nargs="+", type=int, default=None)
    parser.add_argument("--n-seeds", type=int,  default=5,
                        help="Seeds 42..(42+n) (default: 5)")
    # ---- Policy filter ----
    parser.add_argument("--no-rl",     action="store_true",
                        help="Exclude all RL policies; run greedy family only.")
    parser.add_argument("--no-greedy", action="store_true",
                        help="Exclude all greedy policies; run RL models only.")
    parser.add_argument("--rl-model", nargs="+", default=None,
                        choices=["rl_base", "rl_v3", "rl_v3ant", "rl_v4", "rl_v5", "rl_v3new", "rl_v3ant_new", "rl_v4ant", "rl_v5ant", "rl_v6"],
                        help="Restrict to specific RL model(s). Default: all five.")
    # ---- Output ----
    parser.add_argument("--out",          default="benchmark_results")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--verbose",       action="store_true")
    # ---- Sensitivity parameters (override SimulationConfig defaults) ----
    parser.add_argument("--fleet-size",     type=int,   default=None,
                        help="Number of vehicles (default: 6)")
    parser.add_argument("--capacity",       type=int,   default=None,
                        help="Vehicle passenger capacity (default: 16)")
    parser.add_argument("--n-requests",     type=int,   default=None,
                        help="Request cap per episode. For demand sensitivity "
                             "use --inter-arrival instead and set this to 9999.")
    parser.add_argument("--inter-arrival",  type=float, default=None,
                        help="Mean inter-arrival gap in minutes (default: 3.0). "
                             "This is the correct knob for demand intensity. "
                             "Use --n-requests 9999 alongside this flag so the "
                             "cap never binds before the service window closes.")
    parser.add_argument("--max-wait",       type=float, default=None,
                        help="Max passenger wait constraint in minutes (default: 30)")
    parser.add_argument("--ride-factor",    type=float, default=None,
                        help="Max ride = factor × direct time (default: 2.5)")
    parser.add_argument("--demand-profile", type=str,   default=None,
                        choices=["malta", "uniform", "bimodal", "peak"],
                        help="Temporal demand profile (default: malta)")
    return parser.parse_args()


def main():
    args = parse_args()

    seeds = args.seeds if args.seeds else list(range(100, 100 + args.n_seeds))

    greedy_policies = [] if args.no_greedy else list(_GREEDY_POLICIES)

    if args.no_rl:
        rl_policies = []
    else:
        allowed_models = set(args.rl_model) if args.rl_model else set(MODEL_REGISTRY.keys())
        rl_policies = [(p, m) for p, m in _RL_POLICIES if m in allowed_models]

    all_policies = greedy_policies + rl_policies

    if not all_policies:
        print("ERROR: no policies selected.")
        sys.exit(1)

    # Build sensitivity overrides dict from any non-None sensitivity args
    _sensitivity_map = {
        "fleet_size":       args.fleet_size,
        "vehicle_capacity": args.capacity,
        "n_requests":       args.n_requests,
        "inter_arrival":    args.inter_arrival,
        "max_wait":         args.max_wait,
        "ride_factor":      args.ride_factor,
        "demand_profile":   args.demand_profile,
    }
    sim_params = {k: v for k, v in _sensitivity_map.items() if v is not None}

    print(f"\nPolicies to run ({len(all_policies)}):")
    for p, m in all_policies:
        label      = f"{p} ({m})" if m else p
        model_path = MODEL_REGISTRY.get(m) if m else None
        status     = ""
        if model_path:
            status = "  [OK]" if os.path.exists(model_path) else "  [MISSING — update MODEL_REGISTRY]"
        print(f"  {label}{status}")

    missing = [(p, m) for p, m in all_policies
               if m and not os.path.exists(MODEL_REGISTRY.get(m, ""))]
    if missing:
        print(f"\nWARNING: {len(missing)} model file(s) not found.")
        print("  Update MODEL_REGISTRY paths at the top of benchmark.py.")
        if not args.stop_on_error:
            print("  Those runs will be skipped (benchmark continues).\n")

    run_benchmark(
        policies      = all_policies,
        seeds         = seeds,
        out_root      = args.out,
        verbose       = args.verbose,
        stop_on_error = args.stop_on_error,
        sim_params    = sim_params if sim_params else None,
    )


if __name__ == "__main__":
    main()