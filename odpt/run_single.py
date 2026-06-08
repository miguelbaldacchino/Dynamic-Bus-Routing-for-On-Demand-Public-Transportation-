#!/usr/bin/env python3
"""
run_single.py — Extended single-run wrapper for the ODPT demo UI.
Drop into the odpt/ directory alongside main.py and config.py.

Usage:
  python run_single.py --policy rl+ts --model rl2.0 --fleet 6 --verbose
  python run_single.py --policy greedy+alns --seed 42 --requests 400
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Model registry ─────────────────────────────────────────────────────────────
# Paths are relative to the project root (one level above this script's directory)
_ROOT = Path(__file__).resolve().parent.parent

MODEL_REGISTRY: dict[str, str] = {
    # Strategy 1 — Independent reward training
    "rl1.0":   str(_ROOT / "rl_outputs/run_006/model"),        # rl_base
    "rl_base": str(_ROOT / "rl_outputs/run_006/model"),
    "rl1.1":   str(_ROOT / "rl_outputs/run_008/model"),        # rl_v3
    "rl_v3":   str(_ROOT / "rl_outputs/run_008/model"),
    "rl1.1-a": str(_ROOT / "rl_outputs/run_012/checkpoints/best/model"),  # rl_v3ant
    "rl_v3ant":str(_ROOT / "rl_outputs/run_012/checkpoints/best/model"),
    # Strategy 2 — TS-coupled training
    "rl2.0":   str(_ROOT / "rl_outputs/run_009/model_final"),  # rl_v4
    "rl_v4":   str(_ROOT / "rl_outputs/run_009/model_final"),
    "rl2.0-a": str(_ROOT / "rl_outputs/run_015/checkpoints/best/model"),  # rl_v4ant
    "rl_v4ant":str(_ROOT / "rl_outputs/run_015/checkpoints/best/model"),
    "rl2.1":   str(_ROOT / "rl_outputs/run_011/checkpoints/best/model"),  # rl_v5
    "rl_v5":   str(_ROOT / "rl_outputs/run_011/checkpoints/best/model"),
    "rl2.1-a": str(_ROOT / "rl_outputs/run_016/checkpoints/best/model"),  # rl_v5ant
    "rl_v5ant":str(_ROOT / "rl_outputs/run_016/checkpoints/best/model"),
    "rl2.2":   str(_ROOT / "rl_outputs/run_017/checkpoints/best/model"),  # rl_v6
    "rl_v6":   str(_ROOT / "rl_outputs/run_017/checkpoints/best/model"),
    # NOTE: no rl1.0-a or rl2.2-a — those training runs don't exist
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ODPT single-run wrapper with full parameter support")
    p.add_argument("--policy", default="greedy+sa",
                   choices=["greedy", "greedy+sa", "greedy+ts", "greedy+ga", "greedy+alns",
                            "rl", "rl+sa", "rl+ts", "rl+ga", "rl+alns"])
    p.add_argument("--model",           default=None,   help="RL model key (rl1.0/rl1.1/rl2.0/rl2.1/rl2.2) or direct path")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--requests",        type=int,   default=400)
    p.add_argument("--fleet",           type=int,   default=6)
    p.add_argument("--capacity",        type=int,   default=16)
    p.add_argument("--inter-arrival",   type=float, default=3.0,   dest="inter_arrival")
    p.add_argument("--max-wait",        type=float, default=30.0,  dest="max_wait")
    p.add_argument("--ride-factor",     type=float, default=2.5,   dest="ride_factor")
    p.add_argument("--demand-profile",  default="malta",
                   choices=["malta", "uniform"],           dest="demand_profile")
    p.add_argument("--verbose",         action="store_true")
    p.add_argument("--no-viz",          action="store_true",       dest="no_viz")
    return p.parse_args()


def find_latest_output_dir() -> Path | None:
    out = Path("outputs")
    if not out.exists():
        return None
    # Try run_* first (original convention)
    dirs = sorted(out.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if dirs:
        return dirs[0]
    # Fall back to any subdirectory (date/hash names etc.)
    all_dirs = sorted([d for d in out.iterdir() if d.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if all_dirs:
        return all_dirs[0]
    # Fall back to outputs/ itself for flat layout
    if (out / "summary.json").exists() or (out / "events.json").exists():
        return out
    return None


if __name__ == "__main__":
    args = parse_args()

    # Resolve model path
    model_path: str | None = args.model
    if model_path and model_path in MODEL_REGISTRY:
        model_path = MODEL_REGISTRY[model_path]

    if "rl" in args.policy and not model_path:
        print(f"ERROR: RL policy '{args.policy}' requires --model.", flush=True)
        print(f"  Available keys: {list(MODEL_REGISTRY)}", flush=True)
        sys.exit(1)

    # Check model file exists; fall back to base variant if -a model is missing
    if model_path:
        exists = Path(model_path + ".zip").exists() or Path(model_path).exists()
        if not exists and args.model and "-a" in args.model:
            base_key = args.model.replace("-a", "")
            fallback_path = MODEL_REGISTRY.get(base_key)
            if fallback_path and (Path(fallback_path + ".zip").exists() or Path(fallback_path).exists()):
                print(f"WARNING: Anticipatory model '{args.model}' not found — "
                      f"falling back to base variant '{base_key}'.", flush=True)
                model_path = fallback_path
            else:
                print(f"ERROR: Model '{args.model}' not found at {model_path}", flush=True)
                sys.exit(1)

    # Build config
    from config import SimulationConfig
    cfg = SimulationConfig(
        seed             = args.seed,
        n_requests       = args.requests,
        fleet_size       = args.fleet,
        vehicle_capacity = args.capacity,
        inter_arrival    = args.inter_arrival,
        max_wait         = args.max_wait,
        ride_factor      = args.ride_factor,
        demand_profile   = args.demand_profile,
        policy           = args.policy,
    )

    # Run simulation
    from main import main as run_sim
    result = run_sim(
        cfg        = cfg,
        model_path = model_path,
        verbose    = args.verbose,
        visualize  = not args.no_viz,
    )

    # Brief pause to let file I/O flush
    time.sleep(0.3)

    # Attempt to emit metrics for server to capture
    metrics: dict | None = None

    if isinstance(result, dict):
        # Unwrap nested structure if summary.json format: {"config": ..., "metrics": ...}
        metrics = result.get("metrics", result) if "config" in result else result

    # Always scan the latest output dir for files (map, events.json, summary.json fallback)
    latest = find_latest_output_dir()
    if latest:
        if metrics is None:
            sj = latest / "summary.json"
            if sj.exists():
                try:
                    data = json.loads(sj.read_text(encoding="utf-8"))
                    metrics = data.get("metrics", data) if isinstance(data, dict) and "config" in data else data
                except Exception:
                    pass
        maps = list(latest.glob("*.html"))
        if maps:
            print(f"\n__MAP_PATH__{maps[0]}__END__", flush=True)
        # Print events.json content directly — server reads from stdout, no path needed
        evts = latest / "events.json"
        if evts.exists():
            try:
                print("\n__EVENTS_JSON_START__", flush=True)
                print(evts.read_text(encoding="utf-8"), flush=True)
                print("__EVENTS_JSON_END__", flush=True)
            except Exception:
                pass

    if metrics:
        print("\n__METRICS_START__", flush=True)
        print(json.dumps(metrics), flush=True)
        print("__METRICS_END__", flush=True)