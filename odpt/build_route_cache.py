#!/usr/bin/env python3
"""
build_route_cache.py
====================
One-time script: query OSRM for the road geometry between every pair
of Malta On Demand stops and save to route_cache.json.

After this runs once, simulation_map.html will show real road routes
instead of straight lines.

Usage:
    # With local OSRM Docker running:
    py build_route_cache.py

    # With public OSRM server (slower but no Docker needed):
    py build_route_cache.py --osrm-url http://router.project-osrm.org

    # Start OSRM Docker first if needed:
    docker run -d --rm --name osrm-malta -p 5000:5000 ^
        -v %cd%/osrm_data:/data osrm/osrm-backend ^
        osrm-routed --algorithm mld /data/malta-latest.osrm
"""

import argparse
from visualize import build_route_cache
from malta_travel import STOP_COORDS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build OSRM route geometry cache")
    parser.add_argument("--osrm-url", default="http://localhost:5000",
                        help="OSRM server URL (default: http://localhost:5000)")
    args = parser.parse_args()

    print(f"Querying OSRM at {args.osrm_url}")
    print(f"Stops: {len(STOP_COORDS)}")
    print(f"Routes to cache: {len(STOP_COORDS) * (len(STOP_COORDS) - 1)}")
    print()

    build_route_cache(STOP_COORDS, osrm_url=args.osrm_url)
    print("\nDone! Now run: py main.py")
    print("The map will automatically use road-following routes.")
