# dispatcher.py
# Routing / dispatch logic: greedy insertion baseline + SA improvement pass.
#
# Both routines operate on the shared system dict and never touch SimPy
# directly — the simulation layer calls into these functions at each
# decision epoch.

from __future__ import annotations

from copy import deepcopy
from typing import Callable

from models import Stop, Request, Vehicle
from feasibility import check_feasibility, evaluate_plan
from sa import SAPolicy


# ---------------------------------------------------------------------------
# Greedy insertion
# ---------------------------------------------------------------------------

def greedy_insert(
    request: Request,
    vehicles: dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights: tuple[float, float, float],
) -> bool:
    """
    Try to insert *request* into the best feasible position across all
    vehicles using an O(n²) position search.

    Returns True if the request was inserted, False if rejected.
    """
    best_cost   = float("inf")
    best_choice: tuple[str, list] | None = None

    for vid, vehicle in vehicles.items():
        base_plan = vehicle.plan

        for i in range(len(base_plan) + 1):
            for j in range(i + 1, len(base_plan) + 2):

                candidate = deepcopy(base_plan)

                pu = Stop(
                    node=request.pickup_node,
                    kind="PU",
                    req_id=request.id,
                    earliest=request.earliest,
                    service=1.0,
                    request_time=request.request_time,
                )
                do = Stop(
                    node=request.dropoff_node,
                    kind="DO",
                    req_id=request.id,
                    earliest=None,
                    service=1.0,
                    request_time=request.request_time,
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

    if best_choice:
        vid, plan = best_choice
        vehicles[vid].plan = plan
        print(f"  → Inserted {request.id} into {vid}  cost={best_cost:.2f}")
        return True

    print(f"  → Rejected {request.id} (no feasible insertion)")
    return False


# ---------------------------------------------------------------------------
# SA improvement pass
# ---------------------------------------------------------------------------

_SA_POLICY = SAPolicy(
    initial_temp=10_000,
    cooling_rate=0.999,
    iterations=20_000,
    decision_time_limit=0.9,
)


def sa_improve(
    vehicles: dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights: tuple[float, float, float],
) -> None:
    """
    Run one SA improvement pass over all vehicle plans in-place.
    Plans are only updated when SA finds a strictly better solution.
    """
    # Build the state dict expected by SAPolicy
    sa_system_state = {
        **system_state,
        "vehicles": {
            vid: v.to_state_dict(current_time) | {"plan": v.plan}
            for vid, v in vehicles.items()
        },
    }

    print("Running SA improvement…")
    changes = _SA_POLICY.propose(sa_system_state, check_feasibility)

    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        v_state = vehicle.to_state_dict(current_time)

        before = evaluate_plan(vehicle.plan, v_state, system_state, weights)
        after  = evaluate_plan(new_plan,     v_state, system_state, weights)

        if after < before:
            print(f"  SA improved {vid}: {before:.2f} → {after:.2f}")
            vehicle.plan = new_plan
        else:
            print(f"  SA no improvement for {vid}: {before:.2f} → {after:.2f}")


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def print_plans(vehicles: dict[str, Vehicle]) -> None:
    for vid, vehicle in vehicles.items():
        summary = [(s.kind, s.req_id) for s in vehicle.plan]
        print(f"   {vid}: {summary}")