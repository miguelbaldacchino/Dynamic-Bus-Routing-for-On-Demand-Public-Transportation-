# feasibility.py
# Shared feasibility checker and plan-objective evaluator.
#
# Single source of truth for all constraint verification and cost scoring.
# No routing logic lives here.
#
# Commitment / in-progress passenger handling
# -------------------------------------------
# vehicle_process pops PU stops as it serves them.  A vehicle's plan can
# therefore contain a DO stop whose PU stop has already been served (the
# passenger is currently onboard).
#
# check_feasibility: pre-scans for already-onboard passengers and counts
#   them into the initial load.  Their DO stop is a commitment, not a
#   precedence violation.  Ride-time is only checked when the PU time is
#   known within this plan snapshot.
#
# evaluate_plan: already-onboard DO stops contribute travel distance only.
#   Their ride-time cost was recorded at the epoch when they boarded.
#
# In-transit stop handling
# ------------------------
# When the vehicle is mid-travel, to_state_dict() prepends the in-transit
# stop to the plan.  The feasibility checker then sees:
#   [in_transit_stop, ...remaining plan...]
# starting from the vehicle's departure node.  This gives correct travel
# time estimates because travel(departure_node -> in_transit_stop.node)
# accounts for the full leg, and subsequent stops chain from there.
#
# The dispatcher must NOT insert new stops before the in-transit stop
# (it is committed).  This is enforced by the dispatcher, not here.

from __future__ import annotations
from typing import Callable


# ---------------------------------------------------------------------------
# Feasibility checker
# ---------------------------------------------------------------------------

def check_feasibility(
    plan: list,
    vehicle_state: dict,
    system_state: dict,
) -> bool:
    """
    Return True iff *plan* satisfies all hard DARP constraints:

    1. Precedence    — PU before DO, or passenger already onboard.
    2. Capacity      — onboard load never exceeds vehicle capacity.
    3. Pickup TW     — service time >= stop.earliest (wait if early).
    4. Pickup TW UB  — service time <= stop.latest (if set).
    5. Max ride time — ride time <= ride_factor × direct_time
                       (checked only when PU time is known in this plan).

    Parameters
    ----------
    plan          : ordered list of Stop objects
    vehicle_state : dict — keys: capacity, location, time, onboard_count
    system_state  : dict — keys: travel_time, ride_factor, direct_times
    """
    capacity:     int      = vehicle_state["capacity"]
    ride_factor:  float    = system_state["ride_factor"]
    travel_time:  Callable = system_state["travel_time"]

    # Pre-scan: identify already-onboard passengers (DO present, PU already served).
    pu_ids          = {s.req_id for s in plan if s.kind == "PU"}
    do_ids          = {s.req_id for s in plan if s.kind == "DO"}
    already_onboard = do_ids - pu_ids

    # Pre-load capacity with passengers already in the vehicle.
    onboard = vehicle_state.get("onboard_count", len(already_onboard))

    pickup_times: dict[str, float] = {}
    current_node = vehicle_state["location"]
    current_time = vehicle_state["time"]

    for stop in plan:
        current_time += travel_time(current_node, stop.node, current_time)

        if stop.kind == "PU":
            # Lower time window — wait if early
            if stop.earliest and current_time < stop.earliest:
                current_time = stop.earliest

            # Upper time window — reject if too late
            if stop.latest and current_time > stop.latest:
                return False

            onboard += 1
            pickup_times[stop.req_id] = current_time

            if onboard > capacity:
                return False

        elif stop.kind == "DO":
            # True precedence violation only if passenger was never registered
            if stop.req_id not in pu_ids and stop.req_id not in already_onboard:
                return False

            # Ride-time check — only when PU time is available in this snapshot
            if stop.req_id in pickup_times:
                ride_time = current_time - pickup_times[stop.req_id]
                direct    = system_state["direct_times"].get(stop.req_id)
                if direct and ride_time > ride_factor * direct:
                    return False

            onboard -= 1

        current_time += stop.service
        current_node  = stop.node

    return True


# ---------------------------------------------------------------------------
# Objective / cost evaluator
# ---------------------------------------------------------------------------

def evaluate_plan(
    plan: list,
    vehicle_state: dict,
    system_state: dict,
    weights: tuple[float, float, float],
) -> float:
    """
    Weighted cost:  α·distance + β·wait_time + γ·ride_time

    Already-onboard passengers (DO present, PU already served) contribute
    travel distance only — their ride-time cost was counted when they boarded.

    Parameters
    ----------
    weights : (alpha, beta, gamma)
    """
    alpha, beta, gamma = weights
    travel_time: Callable = system_state["travel_time"]

    current_node = vehicle_state["location"]
    current_time = vehicle_state["time"]

    total_distance = total_wait = total_ride = 0.0
    pickup_times: dict[str, float] = {}
    pu_ids = {s.req_id for s in plan if s.kind == "PU"}

    for stop in plan:
        t = travel_time(current_node, stop.node, current_time)
        total_distance += t
        current_time   += t

        if stop.kind == "PU":
            if stop.earliest and current_time < stop.earliest:
                current_time = stop.earliest
            total_wait += current_time - stop.request_time
            pickup_times[stop.req_id] = current_time

        elif stop.kind == "DO":
            if stop.req_id in pickup_times:
                total_ride += current_time - pickup_times[stop.req_id]

        current_time += stop.service
        current_node  = stop.node

    return alpha * total_distance + beta * total_wait + gamma * total_ride