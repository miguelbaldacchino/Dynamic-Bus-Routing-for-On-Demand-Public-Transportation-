# build_route_cache.py
# One-off pre-computation of common route segments for warm-starting.
# Run once before benchmark if caching is enabled in config.
#
# python build_route_cache.py
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
