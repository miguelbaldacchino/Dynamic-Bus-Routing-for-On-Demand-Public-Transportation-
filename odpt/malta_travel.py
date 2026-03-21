# malta_travel.py
# Drop-in replacement for travel.py using real Malta road network.
# 72 stops from the tallinja On Demand service zone.
# Travel times from OSRM with Malta road data (Geofabrik).
#
# Usage (identical to travel.py):
#   from malta_travel import DEFAULT_COORDS, make_travel_fn
#   fn = make_travel_fn(DEFAULT_COORDS)
#   t = fn(0, 15, 60.0)  # node 0 to node 15 at sim time 60 min

from __future__ import annotations
import json
from pathlib import Path


# Load precomputed matrix
_MATRIX_PATH = Path(__file__).parent / "malta_travel_matrix.json"
with open(_MATRIX_PATH, encoding="utf-8") as _f:
    _DATA = json.load(_f)

_DURATIONS = _DATA["durations_minutes"]
_N = _DATA["n_stops"]
_STOPS = _DATA["stops"]

# Node coordinates: dict[int, (x, y)] where x = lon, y = lat
DEFAULT_COORDS: dict[int, tuple[float, float]] = {}

# For mapping / visualization (lat, lon order)
STOP_COORDS: dict[int, tuple[float, float]] = {}
STOP_NAMES: dict[int, str] = {}

for _s in _STOPS:
    _i = _s["id"]
    DEFAULT_COORDS[_i] = (_s["lon"], _s["lat"])
    STOP_COORDS[_i]    = (_s["lat"], _s["lon"])
    STOP_NAMES[_i]     = _s["name"]


def congestion_factor(t_minutes: float) -> float:
    """
    Speed multiplier at simulation time t (minutes from 07:00).
    Calibrated for Malta's On Demand zone (Sliema/St Julian's/San Gwann
    corridor), where peak-hour congestion roughly doubles travel times.

      07:00-09:00  t=0-120    morning peak      x 0.40  (2.5x slower)
      09:00-15:00  t=120-480  off-peak          x 1.00  (OSRM base)
      15:00-17:00  t=480-600  afternoon peak    x 0.45  (2.2x slower)
      17:00-18:00  t=600-660  evening wind-down x 0.70
    """
    if t_minutes < 120:
        return 0.40
    if t_minutes < 480:
        return 1.00
    if t_minutes < 600:
        return 0.45
    return 0.70


def make_travel_fn(coords: dict = None, speed_kmh: float = None):
    """
    Return travel_time(a, b, t) -> float (minutes).

    Uses the precomputed OSRM matrix as the off-peak baseline,
    scaled by congestion_factor(t) for time-of-day variation.

    Parameters coords and speed_kmh are accepted for API compatibility
    with the original travel.py but ignored (the matrix is used).
    """
    def travel_time(a: int, b: int, t: float = 0.0) -> float:
        if a == b:
            return 0.0
        if a < 0 or a >= _N or b < 0 or b >= _N:
            raise ValueError(f"Node index out of range: a={a}, b={b}, N={_N}")

        base = _DURATIONS[a][b]
        if base is None or base <= 0:
            return 0.01

        cf = congestion_factor(t)
        return base / cf

    return travel_time
