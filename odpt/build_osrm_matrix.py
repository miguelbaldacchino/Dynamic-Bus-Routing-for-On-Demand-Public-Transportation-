# build_osrm_matrix.py
# One-off utility: queries a local OSRM server to build
# malta_travel_matrix.json from malta_stops.csv.
# Only needed if regenerating the travel matrix.
#
# python build_osrm_matrix.py --stops malta_stops.csv --out malta_travel_matrix.json


import argparse
import csv
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PBF_URL       = "https://download.geofabrik.de/europe/malta-latest.osm.pbf"
PBF_FILE      = "malta-latest.osm.pbf"
OSRM_DIR      = "osrm_data"
CONTAINER_NAME = "osrm-malta"
OSRM_PORT     = 5000
STOPS_CSV     = "malta_stops.csv"
MATRIX_OUT    = "malta_travel_matrix.json"
MODULE_OUT    = "malta_travel.py"


# ---------------------------------------------------------------------------
# Step 1: Download Malta PBF
# ---------------------------------------------------------------------------

def download_pbf():
    """Download Malta OSM data from Geofabrik."""
    if os.path.exists(PBF_FILE):
        size_mb = os.path.getsize(PBF_FILE) / 1024 / 1024
        print(f"PBF file already exists: {PBF_FILE} ({size_mb:.1f} MB)")
        return

    print(f"Downloading {PBF_URL}...")
    import requests
    resp = requests.get(PBF_URL, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(PBF_FILE, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded/1024/1024:.1f} / {total/1024/1024:.1f} MB ({pct:.0f}%)", end="")
    print(f"\n  Saved: {PBF_FILE}")


# ---------------------------------------------------------------------------
# Step 2: Setup and start OSRM via Docker
# ---------------------------------------------------------------------------

def setup_osrm_docker():
    """Prepare OSRM data and start the Docker container."""

    # Check Docker is available
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: Docker is not installed or not running.")
        print("Install Docker: https://docs.docker.com/get-docker/")
        sys.exit(1)

    # Stop any existing container
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                   capture_output=True)

    # Create data directory and copy PBF
    os.makedirs(OSRM_DIR, exist_ok=True)
    pbf_dest = os.path.join(OSRM_DIR, PBF_FILE)
    if not os.path.exists(pbf_dest):
        import shutil
        shutil.copy(PBF_FILE, pbf_dest)

    abs_dir = os.path.abspath(OSRM_DIR)

    # Extract
    print("\nOSRM: Extracting road network (this takes ~30 seconds)...")
    subprocess.run([
        "docker", "run", "--rm", "-t",
        "-v", f"{abs_dir}:/data",
        "osrm/osrm-backend",
        "osrm-extract",
        "-p", "/opt/car.lua",
        f"/data/{PBF_FILE}"
    ], check=True)

    # Partition
    print("OSRM: Partitioning...")
    subprocess.run([
        "docker", "run", "--rm", "-t",
        "-v", f"{abs_dir}:/data",
        "osrm/osrm-backend",
        "osrm-partition",
        f"/data/{PBF_FILE.replace('.osm.pbf', '.osrm')}"
    ], check=True)

    # Customize
    print("OSRM: Customizing...")
    subprocess.run([
        "docker", "run", "--rm", "-t",
        "-v", f"{abs_dir}:/data",
        "osrm/osrm-backend",
        "osrm-customize",
        f"/data/{PBF_FILE.replace('.osm.pbf', '.osrm')}"
    ], check=True)

    # Start routing server
    print(f"OSRM: Starting server on port {OSRM_PORT}...")
    subprocess.run([
        "docker", "run", "-d", "--rm",
        "--name", CONTAINER_NAME,
        "-p", f"{OSRM_PORT}:{OSRM_PORT}",
        "-v", f"{abs_dir}:/data",
        "osrm/osrm-backend",
        "osrm-routed",
        "--algorithm", "mld",
        "--port", str(OSRM_PORT),
        f"/data/{PBF_FILE.replace('.osm.pbf', '.osrm')}"
    ], check=True)

    # Wait for server to be ready
    import requests as req_lib
    print("  Waiting for OSRM to start...", end="", flush=True)
    for _ in range(30):
        try:
            r = req_lib.get(f"http://localhost:{OSRM_PORT}/health", timeout=2)
            if r.status_code == 200:
                print(" ready!")
                return
        except Exception:
            pass
        # Also try a simple route query as health check
        try:
            r = req_lib.get(
                f"http://localhost:{OSRM_PORT}/table/v1/driving/14.5,35.9;14.5,35.91",
                timeout=2
            )
            if r.status_code == 200:
                print(" ready!")
                return
        except Exception:
            pass
        time.sleep(1)
        print(".", end="", flush=True)

    print("\nWARNING: OSRM may not be ready. Proceeding anyway...")


def stop_osrm_docker():
    """Stop the OSRM Docker container."""
    print(f"\nStopping OSRM container...")
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)


# ---------------------------------------------------------------------------
# Step 3: Load stops and query OSRM
# ---------------------------------------------------------------------------

def load_stops(csv_path: str) -> list[dict]:
    stops = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stops.append({
                "id":   int(row["id"]),
                "name": row["name"],
                "lat":  float(row["lat"]),
                "lon":  float(row["lon"]),
            })
    return stops


def build_matrix(stops: list[dict], osrm_url: str) -> tuple:
    """Query OSRM /table endpoint for all-pairs driving durations and distances."""
    import requests as req_lib

    n = len(stops)
    coords_str = ";".join(f"{s['lon']},{s['lat']}" for s in stops)

    url = f"{osrm_url}/table/v1/driving/{coords_str}"
    params = {"annotations": "duration,distance"}

    print(f"\nQuerying OSRM for {n}x{n} travel matrix...")
    resp = req_lib.get(url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "Ok":
        raise RuntimeError(
            f"OSRM error: {data.get('code')} — {data.get('message', 'unknown')}\n"
            f"Check that your stops are within the Malta road network."
        )

    durations_sec = data["durations"]
    distances_m   = data.get("distances", [])

    # Convert to minutes
    durations_min = []
    for row in durations_sec:
        durations_min.append([
            round(d / 60.0, 3) if d is not None else None
            for d in row
        ])

    # Stats
    flat = [d for row in durations_min for d in row if d and d > 0]
    print(f"  Matrix size: {n}x{n}")
    print(f"  Travel times (minutes):")
    print(f"    Min:  {min(flat):.1f}")
    print(f"    Max:  {max(flat):.1f}")
    print(f"    Mean: {sum(flat)/len(flat):.1f}")

    if distances_m:
        flat_d = [d / 1000 for row in distances_m for d in row if d and d > 0]
        print(f"  Distances (km):")
        print(f"    Min:  {min(flat_d):.1f}")
        print(f"    Max:  {max(flat_d):.1f}")
        print(f"    Mean: {sum(flat_d)/len(flat_d):.1f}")

    return durations_min, distances_m


# ---------------------------------------------------------------------------
# Step 4: Save matrix
# ---------------------------------------------------------------------------

def save_matrix(stops, durations_min, distances_m, output_path):
    result = {
        "n_stops":           len(stops),
        "stops":             stops,
        "durations_minutes": durations_min,
        "distances_meters":  distances_m if distances_m else [],
        "built_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "source":            "OSRM local (malta-latest.osm.pbf from Geofabrik)",
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n  Saved: {output_path} ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Step 5: Generate malta_travel.py
# ---------------------------------------------------------------------------

def generate_travel_module(stops, matrix_path, output_path):
    n = len(stops)
    code = f'''# malta_travel.py
# Drop-in replacement for travel.py using real Malta road network.
# Auto-generated by build_osrm_matrix.py
#
# {n} stops from the tallinja On Demand service zone.
# Travel times from OSRM with Malta road network (Geofabrik).
# Base times represent typical/free-flow driving conditions.
# Congestion factors scale for morning/afternoon peaks.
#
# Usage (identical to travel.py):
#   from malta_travel import DEFAULT_COORDS, make_travel_fn
#   fn = make_travel_fn(DEFAULT_COORDS)
#   t = fn(0, 15, 60.0)  # travel from node 0 to node 15 at sim time 60 min

from __future__ import annotations
import json
from pathlib import Path


# Load precomputed matrix
_MATRIX_PATH = Path(__file__).parent / "{matrix_path}"
with open(_MATRIX_PATH) as _f:
    _DATA = json.load(_f)

_DURATIONS = _DATA["durations_minutes"]
_N = _DATA["n_stops"]
_STOPS = _DATA["stops"]

# Node coordinates: dict[int, (x, y)] where x = lon, y = lat
# Matches the interface expected by the simulation.
DEFAULT_COORDS: dict[int, tuple[float, float]] = {{}}

# For mapping / visualization (lat, lon order)
STOP_COORDS: dict[int, tuple[float, float]] = {{}}
STOP_NAMES: dict[int, str] = {{}}

for _s in _STOPS:
    _i = _s["id"]
    DEFAULT_COORDS[_i] = (_s["lon"], _s["lat"])
    STOP_COORDS[_i]    = (_s["lat"], _s["lon"])
    STOP_NAMES[_i]     = _s["name"]


def congestion_factor(t_minutes: float) -> float:
    """
    Speed multiplier at simulation time t (minutes from 07:00).

      07:00-09:00  t=0-120    morning peak      x 0.65
      09:00-15:00  t=120-480  off-peak          x 1.00
      15:00-17:00  t=480-600  afternoon peak    x 0.70
      17:00-18:00  t=600-660  evening wind-down x 0.85
    """
    if t_minutes < 120:
        return 0.65
    if t_minutes < 480:
        return 1.00
    if t_minutes < 600:
        return 0.70
    return 0.85


def make_travel_fn(coords: dict = None, speed_kmh: float = None):
    """
    Return travel_time(a, b, t) -> float (minutes).

    Uses the precomputed OSRM matrix as the off-peak baseline,
    scaled by congestion_factor(t) for time-of-day variation.

    The OSRM matrix encodes real road geometry, speed limits, and
    turn restrictions for Malta.  Congestion scaling on top provides
    realistic peak-hour slowdowns.

    Parameters coords and speed_kmh are accepted for API compatibility
    with the original travel.py but ignored (the matrix is used).
    """
    def travel_time(a: int, b: int, t: float = 0.0) -> float:
        if a == b:
            return 0.0
        if a < 0 or a >= _N or b < 0 or b >= _N:
            raise ValueError(f"Node index out of range: a={{a}}, b={{b}}, N={{_N}}")

        base = _DURATIONS[a][b]
        if base is None or base <= 0:
            return 0.01

        cf = congestion_factor(t)
        return base / cf

    return travel_time
'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  Generated: {output_path}")
    print(f"    {n} nodes (0 = depot, 1..{n-1} = stops)")
    print(f"    Depot: {stops[0]['name']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Malta OSRM travel-time matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download + Docker + matrix):
  python build_osrm_matrix.py

  # Skip download (you already have the PBF):
  python build_osrm_matrix.py --skip-download

  # Use public OSRM server (no Docker needed, slower):
  python build_osrm_matrix.py --skip-docker --skip-download \\
      --osrm-url http://router.project-osrm.org

  # OSRM already running locally:
  python build_osrm_matrix.py --skip-docker --skip-download \\
      --osrm-url http://localhost:5000
        """,
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Don't download PBF (already have it)")
    parser.add_argument("--skip-docker", action="store_true",
                        help="Don't start OSRM Docker (already running or using remote)")
    parser.add_argument("--osrm-url", default=f"http://localhost:{OSRM_PORT}",
                        help=f"OSRM server URL (default: http://localhost:{OSRM_PORT})")
    parser.add_argument("--stops", default=STOPS_CSV,
                        help=f"Stops CSV file (default: {STOPS_CSV})")
    parser.add_argument("--keep-running", action="store_true",
                        help="Don't stop OSRM container after building matrix")
    args = parser.parse_args()

    print("=" * 60)
    print("Malta On Demand — OSRM Travel Matrix Builder")
    print("=" * 60)

    # Step 1: Download
    if not args.skip_download and not args.skip_docker:
        download_pbf()

    # Step 2: Docker
    if not args.skip_docker:
        setup_osrm_docker()

    # Step 3: Load stops & build matrix
    try:
        stops = load_stops(args.stops)
        print(f"\nLoaded {len(stops)} stops from {args.stops}")

        durations, distances = build_matrix(stops, args.osrm_url)

        # Step 4: Save
        save_matrix(stops, durations, distances, MATRIX_OUT)

        # Step 5: Generate travel module
        generate_travel_module(stops, MATRIX_OUT, MODULE_OUT)

    finally:
        # Step 6: Cleanup
        if not args.skip_docker and not args.keep_running:
            stop_osrm_docker()

    print()
    print("=" * 60)
    print("DONE! Next steps:")
    print("=" * 60)
    print(f"  1. In main.py, change the import:")
    print(f"       from malta_travel import DEFAULT_COORDS, make_travel_fn")
    print(f"  2. In config.py, set:")
    print(f"       n_nodes = {len(stops) - 1}")
    print(f"       depot_node = 0")
    print(f"  3. Run: python main.py")
    print()


if __name__ == "__main__":
    main()