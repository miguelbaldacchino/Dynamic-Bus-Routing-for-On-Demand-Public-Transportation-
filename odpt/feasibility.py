# feasibility.py
from __future__ import annotations
from typing import Callable


def check_feasibility(
    plan: list,
    vehicle_state: dict,
    system_state: dict,
) -> bool:
    """
    Hard DARP constraints:
    1. Precedence  — PU before DO, OR passenger already onboard.
    2. Capacity    — onboard never exceeds vehicle capacity.
    3. Pickup TW   — service >= stop.earliest.
    4. Max ride time — checked only when PU time is known in this plan.
    """
    capacity: int         = vehicle_state["capacity"]
    ride_factor: float    = system_state["ride_factor"]
    travel_time: Callable = system_state["travel_time"]

    # Pre-scan: requests whose PU was already served (only DO remains).
    # These passengers occupy a seat from position 0.
    pu_ids          = {s.req_id for s in plan if s.kind == "PU"}
    do_ids          = {s.req_id for s in plan if s.kind == "DO"}
    already_onboard = do_ids - pu_ids

    onboard      = len(already_onboard)   # pre-load committed passengers
    pickup_times: dict[str, float] = {}

    current_node = vehicle_state["location"]
    current_time = vehicle_state["time"]

    for stop in plan:
        current_time += travel_time(current_node, stop.node, current_time)

        if stop.kind == "PU":
            if stop.earliest and current_time < stop.earliest:
                current_time = stop.earliest
            onboard += 1
            pickup_times[stop.req_id] = current_time
            if onboard > capacity:
                return False

        elif stop.kind == "DO":
            # Real precedence violation: DO with no PU anywhere and not pre-loaded
            if stop.req_id not in pu_ids and stop.req_id not in already_onboard:
                return False
            # Ride-time check — only when PU time is available in this plan
            if stop.req_id in pickup_times:
                ride_time = current_time - pickup_times[stop.req_id]
                direct    = system_state["direct_times"].get(stop.req_id)
                if direct and ride_time > ride_factor * direct:
                    return False
            onboard -= 1

        current_time += stop.service
        current_node  = stop.node

    return True


def evaluate_plan(
    plan: list,
    vehicle_state: dict,
    system_state: dict,
    weights: tuple[float, float, float],
) -> float:
    """
    Weighted cost: α·distance + β·wait_time + γ·ride_time

    DO stops without a matching PU in this plan (already-onboard passengers)
    contribute distance only — their ride-time cost was already counted
    at the epoch when they boarded.
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
            if stop.req_id in pickup_times:          # PU was in this plan
                total_ride += current_time - pickup_times[stop.req_id]
            # else: already onboard — skip, PU cost already recorded

        current_time += stop.service
        current_node  = stop.node

    return alpha * total_distance + beta * total_wait + gamma * total_ride