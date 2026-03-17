# models.py
# Shared data structures used across the simulation.
#
# Changes from previous version
# ------------------------------
# Request  — added status, assignment_time, pickup_time, dropoff_time
# Vehicle  — added onboard set and committed_stops counter
# Stop     — added latest (upper time window bound)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Request status enum (string constants — no import needed)
# ---------------------------------------------------------------------------

class RequestStatus:
    PENDING   = "PENDING"    # Arrived, not yet assigned
    ASSIGNED  = "ASSIGNED"   # Inserted into a vehicle plan
    ONBOARD   = "ONBOARD"    # Passenger has been picked up
    COMPLETED = "COMPLETED"  # Passenger has been dropped off
    REJECTED  = "REJECTED"   # No feasible insertion found


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class Stop:
    """A single pickup or dropoff waypoint in a vehicle's plan."""
    node:         int
    kind:         str            # "PU" or "DO"
    req_id:       str
    earliest:     Optional[float]  # Earliest service time (PU only)
    latest:       Optional[float]  # Latest service time (PU only); None = unconstrained
    service:      float            # Dwell / service time at this stop
    request_time: float            # Epoch at which the originating request arrived


@dataclass
class Request:
    """An on-demand passenger request."""
    id:            str
    pickup_node:   int
    dropoff_node:  int
    earliest:      float           # Earliest pickup time
    request_time:  float           # Wall-clock time the request entered the system

    # Lifecycle tracking — set by the simulation as events occur
    status:          str            = field(default=RequestStatus.PENDING)
    assignment_time: Optional[float] = field(default=None)
    pickup_time:     Optional[float] = field(default=None)
    dropoff_time:    Optional[float] = field(default=None)

    # Computed at request creation for feasibility and metrics
    direct_time:   Optional[float] = field(default=None)

    @property
    def wait_time(self) -> Optional[float]:
        """Actual waiting time: pickup - request arrival."""
        if self.pickup_time is not None:
            return self.pickup_time - self.request_time
        return None

    @property
    def ride_time(self) -> Optional[float]:
        """Actual in-vehicle time."""
        if self.pickup_time is not None and self.dropoff_time is not None:
            return self.dropoff_time - self.pickup_time
        return None

    @property
    def detour_ratio(self) -> Optional[float]:
        """ride_time / direct_time — 1.0 means no detour."""
        if self.ride_time is not None and self.direct_time and self.direct_time > 0:
            return self.ride_time / self.direct_time
        return None


@dataclass
class Vehicle:
    """Fleet vehicle with mutable runtime state."""
    id:       str
    capacity: int
    location: int                  # Current node (updated as vehicle moves)

    plan:             list  = field(default_factory=list)   # Ordered list of Stop objects
    onboard:          set   = field(default_factory=set)    # req_ids currently onboard
    committed_stops:  int   = 0    # Leading stops the dispatcher must not reorder

    def to_state_dict(self, current_time: float) -> dict:
        """
        Snapshot used by the dispatcher and feasibility checker.
        onboard_count reflects passengers already picked up but not yet
        dropped off — the feasibility checker pre-loads this into the
        capacity counter.
        """
        return {
            "capacity":      self.capacity,
            "location":      self.location,
            "time":          current_time,
            "onboard_count": len(self.onboard),
        }