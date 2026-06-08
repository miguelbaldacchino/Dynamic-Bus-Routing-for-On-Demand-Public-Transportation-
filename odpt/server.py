#!/usr/bin/env python3
"""
server.py — ODPT Demo Server
Drop into the odpt/ directory and run: python server.py
UI available at: http://localhost:5000

Requires: pip install flask
"""
from __future__ import annotations
import json, os, re, sys, subprocess, threading, uuid
from pathlib import Path
from flask import Flask, jsonify, request, send_file, send_from_directory, Response

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

# ── Model shortcuts forwarded to run_single.py ────────────────────────────────
RL_MODEL_KEYS = {
    "rl1.1-a", "rl2.0-a", "rl2.1-a",          # anticipatory variants (must match before base)
    "rl1.0",   "rl1.1",   "rl2.0",   "rl2.1",  "rl2.2",
}

# ── In-memory run store ────────────────────────────────────────────────────────
RUNS: dict[str, dict] = {}

# ── Verbose line parsers ───────────────────────────────────────────────────────
_PICKUP_RE  = re.compile(
    r'\[(\d{2}:\d{2})\]\s+(Bus-\d+)\s+picked up\s+(R-?\d+)\s+\(waited\s+([\d.]+)\s+min\)'
)
_DROPOFF_RE = re.compile(
    r'\[(\d{2}:\d{2})\]\s+(Bus-\d+)\s+dropped off\s+(R-?\d+)'
)
# "252 served / 81 rejected / 333 total  (75.7%)"
_SUMMARY_RE = re.compile(
    r'(\d+)\s+served\s*/\s*(\d+)\s+rejected\s*/\s*(\d+)\s+total.*\(([\d.]+)%\)'
)


def _parse_event(line: str) -> dict | None:
    m = _PICKUP_RE.search(line)
    if m:
        return {"type": "pickup", "clock": m[1], "bus": m[2], "req": m[3], "wait": float(m[4])}
    m = _DROPOFF_RE.search(line)
    if m:
        return {"type": "dropoff", "clock": m[1], "bus": m[2], "req": m[3]}
    return None


def _build_cmd(cfg: dict) -> list[str]:
    policy = cfg["policy"]
    rl_key = cfg.get("rlModel", "rl2.0")

    # Normalise "rl2.0-a+ts" → "rl+ts", "rl1.1-a" → "rl"
    # Sort longest-first so "rl2.0-a" matches before "rl2.0"
    for key in sorted(RL_MODEL_KEYS, key=len, reverse=True):
        if policy.startswith(key):
            policy = "rl" + policy[len(key):]
            break

    cmd = [sys.executable, "-u", str(BASE_DIR / "run_single.py"), "--policy", policy]

    if "rl" in policy:
        cmd += ["--model", rl_key]

    cmd += [
        "--seed",           str(cfg.get("seed", 42)),
        "--requests",       str(cfg.get("requests", 400)),
        "--fleet",          str(cfg.get("fleet", 6)),
        "--capacity",       str(cfg.get("capacity", 16)),
        "--inter-arrival",  str(cfg.get("interArrival", 3.0)),
        "--max-wait",       str(cfg.get("maxWait", 30.0)),
        "--ride-factor",    str(cfg.get("rideFactor", 2.5)),
        "--demand-profile", cfg.get("demandProfile", "malta"),
        "--verbose",
    ]
    return cmd


def _run_worker(run_id: str, cmd: list[str]) -> None:
    run = RUNS[run_id]
    events: list[dict] = []
    stdout_lines: list[str] = []
    metrics: dict | None = None
    map_path: str | None = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(BASE_DIR),
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )

        in_metrics      = False
        in_events_json  = False
        metrics_buf:     list[str] = []
        events_json_buf: list[str] = []

        for raw in proc.stdout:
            line = raw.rstrip()
            stdout_lines.append(line)

            # ── Metrics block ──────────────────────────────────────────────────
            if line.strip() == "__METRICS_START__":
                in_metrics = True; continue
            if line.strip() == "__METRICS_END__":
                in_metrics = False
                try:
                    metrics = json.loads("\n".join(metrics_buf))
                except Exception:
                    pass
                metrics_buf = []; continue
            if in_metrics:
                metrics_buf.append(line); continue

            # ── Events JSON block ──────────────────────────────────────────────
            if line.strip() == "__EVENTS_JSON_START__":
                in_events_json = True; continue
            if line.strip() == "__EVENTS_JSON_END__":
                in_events_json = False
                try:
                    raw_events = json.loads("\n".join(events_json_buf))
                    parsed: list[dict] = []
                    for e in raw_events:
                        t = e.get("type")
                        if t == "pickup":
                            parsed.append({"type":"pickup","clock":e["clock"],"bus":e["vehicle"],"req":e["req_id"],"wait":e.get("wait_time",0)})
                        elif t == "dropoff":
                            parsed.append({"type":"dropoff","clock":e["clock"],"bus":e["vehicle"],"req":e["req_id"]})
                        elif t == "reject":
                            parsed.append({"type":"reject","clock":e["clock"],"req":e["req_id"]})
                    if parsed:
                        events = parsed
                except Exception:
                    pass
                events_json_buf = []; continue
            if in_events_json:
                events_json_buf.append(line); continue

            # ── Map path sentinel ──────────────────────────────────────────────
            mp = re.match(r"__MAP_PATH__(.+)__END__", line)
            if mp:
                map_path = mp[1].strip(); continue

            # ── Verbose-stdout fallback (pickup/dropoff only) ──────────────────
            ev = _parse_event(line)
            if ev:
                events.append(ev)
                if ev["type"] == "pickup":
                    run["served_so_far"] = run.get("served_so_far", 0) + 1

        proc.wait()
        run["returncode"] = proc.returncode

    except Exception as exc:
        run["error"] = str(exc)

    # ── Parse summary line from verbose stdout as metrics fallback ────────────
    # e.g. "252 served / 81 rejected / 333 total  (75.7%)"
    summary_metrics: dict | None = None
    for line in stdout_lines:
        m = _SUMMARY_RE.search(line)
        if m:
            served, rejected, total, rate = int(m[1]), int(m[2]), int(m[3]), float(m[4])
            summary_metrics = {
                "served": served, "rejected": rejected,
                "total_requests": total, "service_rate": rate / 100.0,
            }
            break

    # ── Last-resort: scan outputs/ directly ───────────────────────────────────
    out_base = BASE_DIR / "outputs"
    if out_base.exists():
        # Accept any subdirectory, not just run_* — some sims use date/hash names
        all_dirs = sorted(
            [d for d in out_base.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        # Also check outputs/ root itself for flat layout
        search_dirs = all_dirs + [out_base]
        for candidate in search_dirs:
            if metrics is None:
                sj = candidate / "summary.json"
                if sj.exists():
                    try:
                        data = json.loads(sj.read_text(encoding="utf-8"))
                        metrics = data.get("metrics", data) if isinstance(data, dict) and "config" in data else data
                    except Exception:
                        pass
            if not events:
                ej = candidate / "events.json"
                if ej.exists():
                    try:
                        raw_events = json.loads(ej.read_text(encoding="utf-8"))
                        parsed = []
                        for e in raw_events:
                            t = e.get("type")
                            if t == "pickup":
                                parsed.append({"type":"pickup","clock":e["clock"],"bus":e["vehicle"],"req":e["req_id"],"wait":e.get("wait_time",0)})
                            elif t == "dropoff":
                                parsed.append({"type":"dropoff","clock":e["clock"],"bus":e["vehicle"],"req":e["req_id"]})
                            elif t == "reject":
                                parsed.append({"type":"reject","clock":e["clock"],"req":e["req_id"]})
                        if parsed:
                            events = parsed
                    except Exception:
                        pass
            if not map_path:
                maps = list(candidate.glob("*.html"))
                if maps:
                    map_path = str(maps[0])
            if metrics and events:
                break  # found everything we need

    # ── Merge verbose-summary fallback into metrics if still missing fields ───
    if summary_metrics:
        if metrics is None:
            metrics = summary_metrics
        else:
            # Patch in rejected/service_rate if they weren't in the JSON
            for k, v in summary_metrics.items():
                if metrics.get(k) is None:
                    metrics[k] = v

    run.update(
        status="done",
        events=events,
        stdout=stdout_lines[-200:],
        metrics=metrics,
        map_path=map_path,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "ui.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    cfg = request.get_json(force=True, silent=True) or {}
    run_id = uuid.uuid4().hex[:8]
    cmd = _build_cmd(cfg)

    RUNS[run_id] = {
        "status":        "running",
        "served_so_far": 0,
        "total":         cfg.get("requests", 400),
        "cmd":           " ".join(cmd),
    }

    threading.Thread(
        target=_run_worker,
        args=(run_id, cmd),
        daemon=True,
    ).start()

    return jsonify({"run_id": run_id, "cmd": RUNS[run_id]["cmd"]})


@app.route("/api/status/<run_id>")
def api_status(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"error": "not_found"}), 404
    return jsonify({
        "status":        run["status"],
        "served_so_far": run.get("served_so_far", 0),
        "total":         run.get("total", 400),
        "event_count":   len(run.get("events", [])),
    })


@app.route("/api/results/<run_id>")
def api_results(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"error": "not_found"}), 404
    if run["status"] != "done":
        return jsonify({"error": "not_ready"}), 202
    return jsonify({
        "events":      run.get("events", []),
        "metrics":     run.get("metrics"),
        "has_map":     bool(run.get("map_path")),
        "cmd":         run.get("cmd", ""),
        "stdout":      run.get("stdout", []),
        "returncode":  run.get("returncode"),
    })


@app.route("/api/map/<run_id>")
def api_map(run_id: str):
    run = RUNS.get(run_id)
    if not run or not run.get("map_path"):
        return "Map not available", 404
    try:
        return send_file(run["map_path"])
    except Exception as e:
        return str(e), 500


@app.route("/api/debug/<run_id>")
def api_debug(run_id: str):
    """Diagnostic endpoint — open in browser to see exactly what the server captured."""
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"error": "not_found"}), 404
    out_base = BASE_DIR / "outputs"
    dirs_found = []
    if out_base.exists():
        dirs_found = [str(p) for p in sorted(out_base.glob("run_*"),
                      key=lambda p: p.stat().st_mtime, reverse=True)[:5]]
    return jsonify({
        "base_dir":      str(BASE_DIR),
        "outputs_path":  str(out_base),
        "outputs_dirs":  dirs_found,
        "status":        run.get("status"),
        "metrics_keys":  list(run["metrics"].keys()) if run.get("metrics") else None,
        "event_count":   len(run.get("events", [])),
        "event_types":   {t: sum(1 for e in run.get("events",[]) if e["type"]==t)
                          for t in ("pickup","dropoff","reject")},
        "returncode":    run.get("returncode"),
        "stdout_tail":   run.get("stdout", [])[-30:],
    })


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │        ODPT Demo Server                  │")
    print("  │  http://localhost:5000                   │")
    print("  │                                          │")
    print("  │  Ctrl+C to stop                         │")
    print("  └─────────────────────────────────────────┘\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)