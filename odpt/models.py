# models.py
# Shared data structures used across the simulation.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Stop:
    """A single pickup or dropoff waypoint in a vehicle's plan."""
    node: int
    kind: str           # "PU" or "DO"
    req_id: str
    earliest: float     # Earliest service time (PU only; None for DO)
    service: float      # Dwell / service time at this stop
    request_time: float # Epoch at which the originating request arrived


@dataclass
class Request:
    """An on-demand passenger request."""
    id: str
    pickup_node: int
    dropoff_node: int
    earliest: float     # Earliest pickup time
    request_time: float # Wall-clock time the request entered the system


@dataclass
class Vehicle:
    """Fleet vehicle with mutable runtime state."""
    id: str
    capacity: int
    location: int       # Current node (updated as vehicle moves)
    plan: list = field(default_factory=list)  # Ordered list of Stop objects

    def to_state_dict(self, current_time: float) -> dict:
        """Snapshot used by the dispatcher and feasibility checker."""
        return {
            "capacity": self.capacity,
            "location": self.location,
            "time": current_time,
        }