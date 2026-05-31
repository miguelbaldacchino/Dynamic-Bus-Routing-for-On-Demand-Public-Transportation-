# malta_travel.py
# Malta-specific travel function built on the OSRM matrix.
# Provides make_travel_fn() used throughout the simulation.
# Not runnable — imported by main.py and rl_env.py.

from __future__ import annotations
import json
from pathlib import Path


# Load precomputed matrix
_MATRIX_PATH = Path(__file__).parent / "malta_travel_matrix.json"
with open(_MATRIX_PATH) as _f:
    _DATA = json.load(_f)

_DURATIONS = _DATA["durations_minutes"]
_N = _DATA["n_stops"]
_STOPS = _DATA["stops"]

# Node coordinates: dict[int, (x, y)] where x = lon, y = lat
# Matches the interface expected by the simulation.
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
    Speed multiplier for Malta traffic. Simulation t=0 = 05:30.
 
      05:30-06:30  t=0-60    early morning          x 0.90
      06:30-09:30  t=60-240  morning peak (gridlock) x 0.45  (school + commute)
      09:30-15:30  t=240-600 mid-day grind           x 0.65  (no true off-peak in Malta)
      15:30-18:30  t=600-780 afternoon/evening peak  x 0.50  (school pickups + work finish)
      18:30-21:00  t=780-930 evening wind-down       x 0.75  (Sliema/St Julian's still busy; leisure)
      21:00+       t=930+    night flow              x 0.95
    """
    if t_minutes < 60:   return 0.90
    if t_minutes < 240:  return 0.45
    if t_minutes < 600:  return 0.65
    if t_minutes < 780:  return 0.50
    if t_minutes < 930:  return 0.75
    return 0.95


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
            raise ValueError(f"Node index out of range: a={a}, b={b}, N={_N}")

        base = _DURATIONS[a][b]
        if base is None or base <= 0:
            return 0.01

        cf = congestion_factor(t)
        return base / cf

    return travel_time
