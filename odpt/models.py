# models.py
# Dataclasses: Request, Vehicle, Stop, RequestStatus.
# Not runnable — imported by all other modules.


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

    # Maps req_id -> actual pickup time for passengers currently onboard.
    # Used by the feasibility checker to enforce ride-time constraints
    # on already-boarded passengers when new stops are inserted.
    onboard_pickup_times: dict = field(default_factory=dict)

    # --- In-transit tracking (set by vehicle_process) ---
    # When the vehicle pops a stop and begins traveling, it records the
    # stop here.  The dispatcher uses this to reconstruct the true state.
    in_transit_stop:           Optional[object] = field(default=None, repr=False)
    # Time the vehicle departed toward the in-transit stop
    in_transit_depart_time:    Optional[float]  = field(default=None, repr=False)
    # Expected arrival time at the in-transit stop
    in_transit_eta:            Optional[float]  = field(default=None, repr=False)

    # SimPy Event — succeeded when the dispatcher adds stops to an idle
    # vehicle.  Replaced by vehicle_process after each wake.
    # Initialised in main().
    wake_event: Optional[object] = field(default=None, repr=False)

    def to_state_dict(self, current_time: float) -> dict:
        """
        Snapshot used by the dispatcher and feasibility checker.

        If the vehicle is mid-travel (in_transit_stop is set), we
        provide an accurate starting state:
          - location = departure node (the node the vehicle left from)
          - time = in_transit_depart_time (when the vehicle actually left)
        This ensures travel(departure, in_transit_stop.node, depart_time)
        matches the real travel time the vehicle is experiencing, so the
        feasibility checker's time propagation aligns with actual execution.

        If in_transit_depart_time is not set (shouldn't happen, but
        defensive), falls back to current_time.
        """
        if self.in_transit_stop is not None:
            committed_plan = [self.in_transit_stop] + list(self.plan)
            # Use the actual departure time so the feasibility checker
            # computes the same travel time the vehicle is experiencing.
            start_time = (self.in_transit_depart_time
                          if self.in_transit_depart_time is not None
                          else current_time)
        else:
            committed_plan = list(self.plan)
            start_time = current_time

        return {
            "capacity":              self.capacity,
            "location":              self.location,
            "time":                  start_time,
            "onboard_count":         len(self.onboard),
            "plan_snapshot":         committed_plan,
            "onboard_pickup_times":  dict(self.onboard_pickup_times),
        }