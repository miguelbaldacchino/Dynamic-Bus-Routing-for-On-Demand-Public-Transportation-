# visualize.py
# Generates a Folium HTML map from a simulation event log.
# Called automatically by main.py unless --no-viz is passed.
# python visualize.py --events outputs/run_001/events.json

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Event Logger
# ---------------------------------------------------------------------------

@dataclass
class SimEvent:
    time:       float
    clock:      str
    event_type: str     # "depart" | "arrive" | "pickup" | "dropoff" | "request" | "reject"
    vehicle_id: Optional[str]   = None
    req_id:     Optional[str]   = None
    from_node:  Optional[int]   = None
    to_node:    Optional[int]   = None
    details:    Optional[dict]  = None


class EventLogger:
    def __init__(self):
        self.events: list[SimEvent] = []

    def log_request(self, time, clock, req_id, pu_node, do_node):
        self.events.append(SimEvent(
            time=time, clock=clock, event_type="request",
            req_id=req_id, from_node=pu_node, to_node=do_node,
        ))

    def log_reject(self, time, clock, req_id, pu_node, do_node):
        self.events.append(SimEvent(
            time=time, clock=clock, event_type="reject",
            req_id=req_id, from_node=pu_node, to_node=do_node,
        ))

    def log_depart(self, time, clock, vehicle_id, from_node, to_node, req_id, stop_kind):
        self.events.append(SimEvent(
            time=time, clock=clock, event_type="depart",
            vehicle_id=vehicle_id, from_node=from_node, to_node=to_node,
            req_id=req_id, details={"stop_kind": stop_kind},
        ))

    def log_arrive(self, time, clock, vehicle_id, node, req_id, stop_kind):
        self.events.append(SimEvent(
            time=time, clock=clock, event_type="arrive",
            vehicle_id=vehicle_id, to_node=node,
            req_id=req_id, details={"stop_kind": stop_kind},
        ))

    def log_pickup(self, time, clock, vehicle_id, node, req_id, wait_time):
        self.events.append(SimEvent(
            time=time, clock=clock, event_type="pickup",
            vehicle_id=vehicle_id, to_node=node, req_id=req_id,
            details={"wait_time": round(wait_time, 1)},
        ))

    def log_dropoff(self, time, clock, vehicle_id, node, req_id, ride_time):
        self.events.append(SimEvent(
            time=time, clock=clock, event_type="dropoff",
            vehicle_id=vehicle_id, to_node=node, req_id=req_id,
            details={"ride_time": round(ride_time, 1) if ride_time else None},
        ))

    def to_json(self, path: str):
        data = []
        for e in self.events:
            d = {"time": round(e.time, 2), "clock": e.clock, "type": e.event_type}
            if e.vehicle_id: d["vehicle"] = e.vehicle_id
            if e.req_id:     d["req_id"] = e.req_id
            if e.from_node is not None: d["from_node"] = e.from_node
            if e.to_node is not None:   d["to_node"] = e.to_node
            if e.details: d.update(e.details)
            data.append(d)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} events to {path}")


# ---------------------------------------------------------------------------
# Route Cache — OSRM road geometries
# ---------------------------------------------------------------------------

ROUTE_CACHE_PATH = "route_cache.json"


def load_route_cache() -> dict:
    """Load cached OSRM route geometries. Returns dict of "from-to" -> [[lat,lon],...]."""
    if os.path.exists(ROUTE_CACHE_PATH):
        with open(ROUTE_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded route cache: {len(cache)} routes from {ROUTE_CACHE_PATH}")
        return cache
    return {}


def build_route_cache(
    stop_coords: dict,
    osrm_url: str = "http://localhost:5000",
    output_path: str = ROUTE_CACHE_PATH,
) -> dict:
    """
    Query OSRM /route endpoint for every unique (a, b) stop pair
    and cache the road geometry. Run this once with OSRM running.
    """
    import requests as req_lib
    import time as _time

    n = len(stop_coords)
    cache = {}
    total = n * (n - 1)
    done = 0
    failed = 0
    t0 = _time.time()

    print(f"Building route cache: {total} routes via {osrm_url}...")

    # Use a session for connection pooling (much faster than individual requests)
    session = req_lib.Session()

    # Quick connectivity test
    try:
        test_a = stop_coords[0]
        test_b = stop_coords[1]
        test_url = f"{osrm_url}/route/v1/driving/{test_a[1]},{test_a[0]};{test_b[1]},{test_b[0]}"
        r = session.get(test_url, params={"overview": "full", "geometries": "geojson"}, timeout=5)
        r.raise_for_status()
        print(f"  OSRM connection OK ({r.elapsed.total_seconds()*1000:.0f}ms per query)")
    except Exception as e:
        print(f"  ERROR: Cannot reach OSRM at {osrm_url}")
        print(f"  {e}")
        print(f"  Make sure OSRM is running (check Docker)")
        return {}

    for a in range(n):
        row_start = _time.time()
        for b in range(n):
            if a == b:
                continue

            key = f"{a}-{b}"
            lat_a, lon_a = stop_coords[a]
            lat_b, lon_b = stop_coords[b]

            try:
                url = f"{osrm_url}/route/v1/driving/{lon_a},{lat_a};{lon_b},{lat_b}"
                resp = session.get(url, params={
                    "overview": "full",
                    "geometries": "geojson",
                }, timeout=5)
                resp.raise_for_status()
                data = resp.json()

                if data.get("code") == "Ok" and data.get("routes"):
                    coords = data["routes"][0]["geometry"]["coordinates"]
                    cache[key] = [[c[1], c[0]] for c in coords]
                else:
                    cache[key] = [[lat_a, lon_a], [lat_b, lon_b]]
                    failed += 1
            except Exception:
                cache[key] = [[lat_a, lon_a], [lat_b, lon_b]]
                failed += 1

            done += 1

        # Print after each source node (every 71 routes)
        elapsed = _time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        print(f"  Node {a+1}/{n} done | {done}/{total} routes | "
              f"{rate:.0f} routes/sec | ~{remaining:.0f}s remaining")

    session.close()

    with open(output_path, "w") as f:
        json.dump(cache, f)

    elapsed = _time.time() - t0
    print(f"\nRoute cache saved: {len(cache)} routes to {output_path}")
    print(f"  Time: {elapsed:.1f}s ({done/elapsed:.0f} routes/sec)")
    if failed:
        print(f"  {failed} routes fell back to straight lines")
    return cache


# ---------------------------------------------------------------------------
# Map Generator
# ---------------------------------------------------------------------------

def generate_map(
    logger:      EventLogger,
    output_path: str = "simulation_map.html",
    stop_coords: dict = None,
    stop_names:  dict = None,
):
    try:
        import folium
    except ImportError:
        print("Folium not installed. Run: pip install folium")
        logger.to_json(output_path.replace(".html", "_events.json"))
        return

    if stop_coords is None or stop_names is None:
        try:
            from malta_travel import STOP_COORDS, STOP_NAMES
            stop_coords = stop_coords or STOP_COORDS
            stop_names = stop_names or STOP_NAMES
        except ImportError:
            print("malta_travel not found.")
            return

    # Load route cache
    route_cache = load_route_cache()
    use_roads = len(route_cache) > 0
    if not use_roads:
        print("No route_cache.json found — using straight lines.")
        print("To get road-following routes, run: py build_route_cache.py")

    # Center map
    all_lats = [c[0] for c in stop_coords.values()]
    all_lons = [c[1] for c in stop_coords.values()]
    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    # --- Bus stops ---
    stop_group = folium.FeatureGroup(name="Bus Stops", show=True)
    for node_id, (lat, lon) in stop_coords.items():
        name = stop_names.get(node_id, f"Stop {node_id}")
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#333",
            fill=True,
            fill_color="#666",
            fill_opacity=0.7,
            popup=f"<b>{name}</b><br>Node {node_id}",
            tooltip=name,
        ).add_to(stop_group)
    stop_group.add_to(m)

    # --- Vehicle routes ---
    vehicle_colors = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
        "#fabed4", "#469990",
    ]

    vehicle_segments: dict[str, list] = {}
    for e in logger.events:
        if e.event_type == "depart" and e.vehicle_id:
            vid = e.vehicle_id
            if vid not in vehicle_segments:
                vehicle_segments[vid] = []
            if e.from_node is not None and e.to_node is not None:
                from_coord = stop_coords.get(e.from_node)
                to_coord = stop_coords.get(e.to_node)
                if from_coord and to_coord:
                    vehicle_segments[vid].append({
                        "from_node": e.from_node,
                        "to_node": e.to_node,
                        "from": from_coord,
                        "to": to_coord,
                        "time": e.time,
                        "clock": e.clock,
                        "req_id": e.req_id,
                        "kind": e.details.get("stop_kind", "") if e.details else "",
                    })

    for i, (vid, segments) in enumerate(sorted(vehicle_segments.items())):
        color = vehicle_colors[i % len(vehicle_colors)]
        route_group = folium.FeatureGroup(name=vid, show=True)

        for seg in segments:
            # Look up road geometry from cache
            cache_key = f"{seg['from_node']}-{seg['to_node']}"
            if use_roads and cache_key in route_cache:
                locations = route_cache[cache_key]
            else:
                # Straight line fallback
                locations = [
                    [seg["from"][0], seg["from"][1]],
                    [seg["to"][0], seg["to"][1]],
                ]

            folium.PolyLine(
                locations=locations,
                color=color,
                weight=3,
                opacity=0.7,
                tooltip=f"{vid} [{seg['clock']}] {seg['kind']} {seg['req_id']}",
            ).add_to(route_group)

        route_group.add_to(m)

    # --- Pickups ---
    pu_group = folium.FeatureGroup(name="Pickups", show=False)
    for e in logger.events:
        if e.event_type == "pickup" and e.to_node is not None:
            coord = stop_coords.get(e.to_node)
            if coord:
                wait = e.details.get("wait_time", "?") if e.details else "?"
                folium.CircleMarker(
                    location=[coord[0], coord[1]],
                    radius=6,
                    color="#2ecc71",
                    fill=True,
                    fill_color="#2ecc71",
                    fill_opacity=0.8,
                    popup=(f"<b>Pickup {e.req_id}</b><br>"
                           f"Vehicle: {e.vehicle_id}<br>"
                           f"Time: {e.clock}<br>"
                           f"Wait: {wait} min"),
                ).add_to(pu_group)
    pu_group.add_to(m)

    # --- Dropoffs ---
    do_group = folium.FeatureGroup(name="Dropoffs", show=False)
    for e in logger.events:
        if e.event_type == "dropoff" and e.to_node is not None:
            coord = stop_coords.get(e.to_node)
            if coord:
                ride = e.details.get("ride_time", "?") if e.details else "?"
                folium.CircleMarker(
                    location=[coord[0], coord[1]],
                    radius=6,
                    color="#e74c3c",
                    fill=True,
                    fill_color="#e74c3c",
                    fill_opacity=0.8,
                    popup=(f"<b>Dropoff {e.req_id}</b><br>"
                           f"Vehicle: {e.vehicle_id}<br>"
                           f"Time: {e.clock}<br>"
                           f"Ride: {ride} min"),
                ).add_to(do_group)
    do_group.add_to(m)

    # --- Rejected ---
    rej_group = folium.FeatureGroup(name="Rejected", show=False)
    for e in logger.events:
        if e.event_type == "reject" and e.from_node is not None:
            coord = stop_coords.get(e.from_node)
            if coord:
                folium.CircleMarker(
                    location=[coord[0], coord[1]],
                    radius=8,
                    color="#e74c3c",
                    fill=True,
                    fill_color="#ff0000",
                    fill_opacity=0.5,
                    popup=(f"<b>REJECTED {e.req_id}</b><br>"
                           f"Time: {e.clock}<br>"
                           f"From: {stop_names.get(e.from_node, e.from_node)}<br>"
                           f"To: {stop_names.get(e.to_node, e.to_node)}"),
                ).add_to(rej_group)
    rej_group.add_to(m)

    # --- Layer control ---
    folium.LayerControl(collapsed=False).add_to(m)

    # --- Summary box ---
    n_requests = sum(1 for e in logger.events if e.event_type == "request")
    n_pickups = sum(1 for e in logger.events if e.event_type == "pickup")
    n_dropoffs = sum(1 for e in logger.events if e.event_type == "dropoff")
    n_rejects = sum(1 for e in logger.events if e.event_type == "reject")

    summary_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3); font-size:13px;
                font-family:monospace; line-height:1.6;">
        <b>DARP Simulation &mdash; Malta On Demand</b><br>
        Requests: {n_requests} | Served: {n_dropoffs} | Rejected: {n_rejects}<br>
        Vehicles: {len(vehicle_segments)} | Route type: {"road" if use_roads else "straight"}<br>
        <i>Toggle layers in top-right control</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(summary_html))

    m.save(output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Map saved to {output_path} ({size_kb:.0f} KB)")