# dispatcher.py
# Routing / dispatch logic: greedy insertion baseline + SA improvement pass.
#
# Both routines operate on frozen snapshots and never touch SimPy directly.
# The simulation layer calls these functions at each decision epoch.
#
# Fixes applied
# -------------
# - greedy_insert uses vehicle.to_state_dict() which includes the
#   in-transit stop in plan_snapshot.  New stops are only inserted AFTER
#   the committed (in-transit) prefix, preserving execution correctness.
# - Reduced deepcopy overhead: only the base plan is copied once per
#   vehicle; candidate plans are built by list slicing + insertion.
# - SA time budget is per-vehicle (start_time reset per vehicle call).
# - wake_event is triggered when stops are added to an idle vehicle.

from __future__ import annotations

import time
from copy import deepcopy
from typing import Optional

from models import Stop, Request, Vehicle
from feasibility import check_feasibility, evaluate_plan
from metrics import MetricsCollector
from sa import SAPolicy


# ---------------------------------------------------------------------------
# Module-level SA policy instance
# ---------------------------------------------------------------------------

_SA_POLICY: Optional[SAPolicy] = None


def build_sa_policy(cfg) -> None:
    """Initialise the module-level SA policy from a SimulationConfig."""
    global _SA_POLICY
    _SA_POLICY = SAPolicy(
        initial_temp        = cfg.sa_initial_temp,
        cooling_rate        = cfg.sa_cooling_rate,
        iterations          = cfg.sa_iterations,
        decision_time_limit = cfg.sa_time_limit,
    )


def _get_sa_policy() -> SAPolicy:
    """Return SA policy, creating a default one if build_sa_policy() was not called."""
    global _SA_POLICY
    if _SA_POLICY is None:
        _SA_POLICY = SAPolicy()
    return _SA_POLICY


# ---------------------------------------------------------------------------
# Greedy insertion
# ---------------------------------------------------------------------------

def greedy_insert(
    request:      Request,
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights:      tuple[float, float, float],
    metrics=None,   # Optional[MetricsCollector]
) -> bool:
    """
    Insert *request* into the best feasible position across all vehicles.
    Returns True if inserted, False if rejected.

    The in-transit stop (if any) is included as a committed prefix in
    the plan snapshot.  New stops are only inserted into positions AFTER
    this prefix, so the vehicle's current committed movement is never
    disrupted.

    Wake event is triggered if the chosen vehicle was idle.
    """
    import time as _time
    t0 = _time.time()

    best_cost   = float("inf")
    best_choice = None
    max_wait    = system_state.get("max_wait", float("inf"))

    for vid, vehicle in vehicles.items():
        v_state = vehicle.to_state_dict(current_time)
        full_plan = v_state["plan_snapshot"]

        # Number of committed stops (in-transit stop, if present)
        n_committed = 1 if vehicle.in_transit_stop is not None else 0

        # Only try inserting AFTER the committed prefix
        insertable = full_plan[n_committed:]
        n = len(insertable)

        for i in range(n + 1):
            for j in range(i + 1, n + 2):

                # Build candidate: committed prefix + insertable with new PU/DO
                candidate_tail = list(insertable)

                pu = Stop(
                    node         = request.pickup_node,
                    kind         = "PU",
                    req_id       = request.id,
                    earliest     = request.earliest,
                    latest       = request.request_time + max_wait,
                    service      = 1.0,
                    request_time = request.request_time,
                )
                do = Stop(
                    node         = request.dropoff_node,
                    kind         = "DO",
                    req_id       = request.id,
                    earliest     = None,
                    latest       = None,
                    service      = 1.0,
                    request_time = request.request_time,
                )

                candidate_tail.insert(i, pu)
                candidate_tail.insert(j, do)

                candidate = full_plan[:n_committed] + candidate_tail

                if not check_feasibility(candidate, v_state, system_state):
                    continue

                cost = evaluate_plan(candidate, v_state, system_state, weights)

                if cost < best_cost:
                    best_cost   = cost
                    best_choice = (vid, candidate, n_committed)

    elapsed = _time.time() - t0
    if metrics:
        metrics.log_decision_latency(elapsed)

    if best_choice:
        vid, full_candidate, n_committed = best_choice
        vehicle = vehicles[vid]

        # Write back only the non-committed portion to vehicle.plan
        # (the in-transit stop stays on vehicle.in_transit_stop)
        vehicle.plan = full_candidate[n_committed:]

        # Wake idle vehicle if it was waiting
        if vehicle.wake_event is not None and not vehicle.wake_event.triggered:
            vehicle.wake_event.succeed()

        print(f"  -> Inserted {request.id} into {vid}  cost={best_cost:.2f}")
        return True

    print(f"  -> Rejected {request.id} (no feasible insertion)")
    if metrics:
        metrics.mark_rejected(request.id)
    return False


# ---------------------------------------------------------------------------
# SA improvement pass
# ---------------------------------------------------------------------------

def sa_improve(
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights:      tuple[float, float, float],
    metrics:     Optional[MetricsCollector] = None,
) -> None:
    """
    Run one SA improvement pass over all vehicle plans in-place.
    Plans are updated only when SA finds a strictly better solution.

    Each vehicle gets its own independent time budget (per-vehicle limit).
    The in-transit stop is included in the plan snapshot so SA sees the
    full committed route, but SA only modifies non-committed stops.
    """
    sa_system_state = {
        **system_state,
        "vehicles": {},
    }

    for vid, v in vehicles.items():
        vs = v.to_state_dict(current_time)
        n_committed = 1 if v.in_transit_stop is not None else 0
        sa_system_state["vehicles"][vid] = {
            **vs,
            "plan":        deepcopy(vs["plan_snapshot"]),
            "n_committed": n_committed,
        }

    changes = _get_sa_policy().propose(
        sa_system_state, check_feasibility, weights
    )

    # SA has already validated feasibility and confirmed combined cost
    # improvement across all returned vehicles.  We accept all changes
    # as a group — accepting some but not others would decouple
    # inter-vehicle moves, causing requests to vanish from all plans.
    if not changes:
        return

    # Compute combined improvement for logging
    total_before = 0.0
    total_after  = 0.0
    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        v_state = vehicle.to_state_dict(current_time)
        n_committed = 1 if vehicle.in_transit_stop is not None else 0

        total_before += evaluate_plan(
            v_state["plan_snapshot"], v_state, system_state, weights
        )
        total_after += evaluate_plan(
            new_plan, v_state, system_state, weights
        )

    if total_after >= total_before:
        return  # SA said it improved but rounding says no — skip

    # Apply all changes atomically
    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        n_committed = 1 if vehicle.in_transit_stop is not None else 0
        vehicle.plan = new_plan[n_committed:]

        # Wake idle vehicle if SA gave it new stops (inter-vehicle move)
        if (vehicle.wake_event is not None
                and not vehicle.wake_event.triggered
                and vehicle.plan):
            vehicle.wake_event.succeed()

    if metrics:
        metrics.log_improvement()


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def print_plans(vehicles: dict[str, Vehicle]) -> None:
    for vid, vehicle in vehicles.items():
        prefix = ""
        if vehicle.in_transit_stop is not None:
            s = vehicle.in_transit_stop
            prefix = f"[IN-TRANSIT: {s.kind} {s.req_id}] "
        summary = [(s.kind, s.req_id) for s in vehicle.plan]
        print(f"   {vid}: {prefix}{summary}")