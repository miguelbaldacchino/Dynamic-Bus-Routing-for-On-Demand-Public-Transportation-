# travel.py
# Base travel-time utilities and matrix lookup helpers.
# Not runnable — imported by malta_travel.py and dispatcher.

from __future__ import annotations
import math


DEFAULT_COORDS: dict[int, tuple[float, float]] = {
    0:  ( 0.0,  0.0),   # Depot / central interchange

    # Town centre
    1:  ( 0.5,  0.8),
    2:  (-0.6,  0.7),
    3:  ( 0.3, -0.9),
    4:  (-0.4, -0.6),
    5:  ( 0.9,  0.2),

    # Inner suburbs
    6:  ( 2.1,  2.4),
    7:  (-2.0,  2.1),
    8:  ( 2.3, -1.8),
    9:  (-1.9, -2.2),
    10: ( 3.0,  0.5),

    # Outer suburbs
    11: ( 4.2,  3.8),
    12: (-3.8,  3.5),
    13: ( 4.5, -3.0),
    14: (-4.0, -3.6),
    15: ( 5.1,  0.3)
}

_BASE_SPEED_KMH: float = 30.0


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


def euclidean_km(a: int, b: int, coords: dict) -> float:
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.hypot(ax - bx, ay - by)


def make_travel_fn(coords: dict, speed_kmh: float = _BASE_SPEED_KMH):
    """
    Return travel_time(a, b, t) -> float (minutes).
    t is simulation time in minutes (used for congestion factor).
    """
    def travel_time(a: int, b: int, t: float = 0.0) -> float:
        if a == b:
            return 0.0
        dist_km   = euclidean_km(a, b, coords)
        eff_speed = speed_kmh * congestion_factor(t)
        return (dist_km / eff_speed) * 60.0

    return travel_time