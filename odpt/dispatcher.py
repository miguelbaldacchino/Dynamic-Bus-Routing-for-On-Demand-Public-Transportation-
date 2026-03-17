# dispatcher.py
# Routing / dispatch logic: greedy insertion baseline + SA improvement pass.
#
# Both routines operate on frozen snapshots and never touch SimPy directly.
# The simulation layer calls these functions at each decision epoch.
#
# Changes from previous version
# ------------------------------
# - weights forwarded to _SA_POLICY.propose() (was using hardcoded default)
# - deepcopy(v.plan) in snapshot prevents SimPy race condition
# - MetricsCollector hooks added (optional — pass None to skip)
# - Stop.latest set from system_state["max_wait"] on insertion

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
# Instantiated once; hyperparameters come from SimulationConfig via
# build_sa_policy() called from main().

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
        _SA_POLICY = SAPolicy(
            initial_temp        = 10_000,
            cooling_rate        = 0.999,
            iterations          = 20_000,
            decision_time_limit = 0.9,
        )
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

    Stop.latest is set to current_time + system_state["max_wait"] so the
    upper pickup time window is enforced by the feasibility checker.
    """
    import time as _time
    t0 = _time.time()

    best_cost   = float("inf")
    best_choice = None
    max_wait    = system_state.get("max_wait", float("inf"))

    for vid, vehicle in vehicles.items():
        base_plan = vehicle.plan

        for i in range(len(base_plan) + 1):
            for j in range(i + 1, len(base_plan) + 2):

                candidate = deepcopy(base_plan)

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

                candidate.insert(i, pu)
                candidate.insert(j, do)

                v_state = vehicle.to_state_dict(current_time)

                if not check_feasibility(candidate, v_state, system_state):
                    continue

                cost = evaluate_plan(candidate, v_state, system_state, weights)

                if cost < best_cost:
                    best_cost   = cost
                    best_choice = (vid, candidate)

    elapsed = _time.time() - t0
    if metrics:
        metrics.log_decision_latency(elapsed)

    if best_choice:
        vid, plan = best_choice
        vehicles[vid].plan = plan
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

    Key fix: deepcopy(v.plan) produces a frozen snapshot so that
    vehicle_process cannot mutate the plan while SA is reading it.
    """
    sa_system_state = {
        **system_state,
        "vehicles": {
            vid: v.to_state_dict(current_time) | {"plan": deepcopy(v.plan)}
            for vid, v in vehicles.items()
        },
    }

    print("Running SA improvement...")
    # weights forwarded so SA searches the same objective as greedy_insert
    changes = _get_sa_policy().propose(sa_system_state, check_feasibility, weights)

    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        v_state = vehicle.to_state_dict(current_time)

        before = evaluate_plan(vehicle.plan, v_state, system_state, weights)
        after  = evaluate_plan(new_plan,     v_state, system_state, weights)

        if after < before:
            print(f"  SA improved {vid}: {before:.2f} -> {after:.2f}")
            vehicle.plan = new_plan
            if metrics:
                metrics.log_improvement()
        else:
            print(f"  SA no improvement for {vid}: {before:.2f} vs {after:.2f}")


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def print_plans(vehicles: dict[str, Vehicle]) -> None:
    for vid, vehicle in vehicles.items():
        summary = [(s.kind, s.req_id) for s in vehicle.plan]
        print(f"   {vid}: {summary}")