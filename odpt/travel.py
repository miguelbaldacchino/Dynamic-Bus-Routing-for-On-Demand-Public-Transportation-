# travel.py
# Map initialisation and travel-time model.
#
# Version 1: static Euclidean distance as a proxy for travel time.
# Future: replace travel_time() with a time-dependent lookup or
#         road-network shortest-path while keeping the same interface.

import math


def euclidean_travel_time(a: int, b: int, coords: dict) -> float:
    """Straight-line distance between node a and node b."""
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.hypot(ax - bx, ay - by)


def make_travel_fn(coords: dict):
    """
    Return a travel_time(a, b, t) callable bound to *coords*.

    The extra argument *t* (current simulation time) is reserved for
    future time-dependent extensions and is ignored in Version 1.
    """
    def travel_time(a: int, b: int, t: float = 0.0) -> float:
        return euclidean_travel_time(a, b, coords)

    return travel_time


# ---------------------------------------------------------------------------
# Default small test map (7 nodes including depot at 0)
# ---------------------------------------------------------------------------

DEFAULT_COORDS = {
    0: (0, 0),
    1: (1, 2),
    2: (4, 1),
    3: (2, 6),
    4: (7, 5),
    5: (8, 1),
    6: (5, 8),
}