# sa.py
# Simulated Annealing improvement policy.
#
# SAPolicy.propose() is the only public entry point.  It receives a
# system_state snapshot and a feasibility_checker callable, then returns
# a dict of {vehicle_id: improved_plan} for any vehicle whose plan
# was improved.  It never touches SimPy or the live vehicle objects.
#
# Key design decisions
# --------------------
# 1. Objective: uses the SAME evaluate_plan() as the dispatcher and greedy
#    inserter.  The old _route_cost() only summed travel distance, which
#    caused SA to search a different landscape than it was judged on.
#
# 2. Neighbour operator: pair-relocate instead of blind swap.
#    A blind swap almost always violates PU-before-DO precedence, so the
#    feasibility checker rejects >90 % of moves and SA never explores.
#    Pair-relocate removes one request's PU+DO stops and re-inserts them
#    at new positions i < j, guaranteeing precedence by construction.
#    A second operator (pair-swap between two requests) is included and
#    selected randomly for diversity.
#
# 3. StopIteration safety: _neighbour uses plain for-loops, not next()
#    inside a generator expression, which raises RuntimeError under
#    PEP 479 (Python >= 3.7).

import math
import random
import time
from copy import deepcopy
from typing import Callable

from feasibility import evaluate_plan


class SAPolicy:

    def __init__(
        self,
        initial_temp: float = 1_000.0,
        cooling_rate: float = 0.995,
        iterations: int = 2_000,
        decision_time_limit: float = 0.05,
    ):
        self.initial_temp        = initial_temp
        self.cooling_rate        = cooling_rate
        self.iterations          = iterations
        self.decision_time_limit = decision_time_limit

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def propose(
        self,
        system_state: dict,
        feasibility_checker: Callable,
        weights: tuple = (1.0, 2.0, 1.0),
    ) -> dict:
        """
        Propose improved plans for vehicles in *system_state* that have
        at least two requests (four stops).

        Returns
        -------
        dict mapping vehicle_id → improved plan.
        Only vehicles whose plan was strictly improved are included.
        """
        start_time = time.time()
        new_plans  = {}

        for vehicle_id, vehicle in system_state["vehicles"].items():
            current_plan = vehicle["plan"]          # <-- always assigned before use

            # Need at least 2 requests (4 stops) to have a non-trivial
            # neighbourhood; a single PU+DO pair cannot be reordered.
            if len(current_plan) < 4:
                continue

            improved = self._run_sa(
                vehicle_id,
                current_plan,
                system_state,
                feasibility_checker,
                weights,
                start_time,
            )

            if improved is not None:
                new_plans[vehicle_id] = improved

        return new_plans

    # ------------------------------------------------------------------
    # Neighbour operators
    # ------------------------------------------------------------------

    def _requests_in_plan(self, plan: list) -> list[str]:
        """
        Return req_ids that have BOTH a PU and a DO stop present in plan.
        This guards against partially-served requests where vehicle_process
        has already popped the PU stop.
        """
        pu_ids = {s.req_id for s in plan if s.kind == "PU"}
        do_ids = {s.req_id for s in plan if s.kind == "DO"}
        return list(pu_ids & do_ids)   # only complete pairs

    def _pair_relocate(self, plan: list) -> list | None:
        """
        Remove one request's PU+DO pair and re-insert at new positions i<j.
        Precedence is preserved by construction.
        Returns None if no complete pair exists.
        """
        candidates = self._requests_in_plan(plan)
        if not candidates:
            return None

        req = random.choice(candidates)

        # Extract the two stops — plain loops, no next() inside generator
        pu_stop = None
        do_stop = None
        for s in plan:
            if s.req_id == req and s.kind == "PU":
                pu_stop = s
            elif s.req_id == req and s.kind == "DO":
                do_stop = s

        if pu_stop is None or do_stop is None:
            return None  # safety: shouldn't happen after _requests_in_plan

        # Build plan without this pair
        stripped = [s for s in plan if s.req_id != req]
        n = len(stripped)

        # Choose new insertion positions i < j
        # i  : position of PU in stripped list (0 … n)
        # j  : position of DO after PU is inserted (i+1 … n+1)
        i = random.randint(0, n)
        j = random.randint(i + 1, n + 1)

        new_plan = deepcopy(stripped)
        new_plan.insert(i, deepcopy(pu_stop))
        new_plan.insert(j, deepcopy(do_stop))
        return new_plan

    def _pair_swap(self, plan: list) -> list | None:
        """
        Swap the positions of two requests' PU+DO pairs.
        Concretely: exchange the PU positions of req_a and req_b,
        and exchange their DO positions.
        Returns None if fewer than two complete pairs exist.
        """
        candidates = self._requests_in_plan(plan)
        if len(candidates) < 2:
            return None

        req_a, req_b = random.sample(candidates, 2)

        new_plan = deepcopy(plan)
        for s in new_plan:
            if s.req_id == req_a:
                s.req_id = req_b
            elif s.req_id == req_b:
                s.req_id = req_a

        # Also swap the node/kind data so the stops point to the right places
        # Simpler alternative: swap (node, req_id) pairs for matching kinds
        # Full implementation: swap the whole Stop objects at those indices.
        idx = {req_a: {}, req_b: {}}
        for i, s in enumerate(plan):
            if s.req_id in idx:
                idx[s.req_id][s.kind] = i

        if ("PU" not in idx[req_a] or "DO" not in idx[req_a] or
                "PU" not in idx[req_b] or "DO" not in idx[req_b]):
            return None

        new_plan = deepcopy(plan)
        # Swap PU stops
        pu_a, pu_b = idx[req_a]["PU"], idx[req_b]["PU"]
        new_plan[pu_a], new_plan[pu_b] = deepcopy(plan[pu_b]), deepcopy(plan[pu_a])
        # Swap DO stops
        do_a, do_b = idx[req_a]["DO"], idx[req_b]["DO"]
        new_plan[do_a], new_plan[do_b] = deepcopy(plan[do_b]), deepcopy(plan[do_a])
        return new_plan

    def _neighbour(self, plan: list) -> list | None:
        """
        Select a neighbour operator at random and apply it.
        Returns None if no valid neighbour could be generated
        (caller should skip this iteration).
        """
        op = random.choice([self._pair_relocate, self._pair_swap])
        return op(plan)

    # ------------------------------------------------------------------
    # Core SA loop
    # ------------------------------------------------------------------

    def _run_sa(
        self,
        vehicle_id: str,
        initial_plan: list,
        system_state: dict,
        feasibility_checker: Callable,
        weights: tuple,
        start_time: float,
    ):
        """
        Run SA for one vehicle.  Returns the best plan found, or None if
        no improvement over the initial plan was achieved.
        """
        print(f"  SA evaluating {vehicle_id} ({len(initial_plan)} stops)")

        vehicle_state = system_state["vehicles"][vehicle_id]

        current      = deepcopy(initial_plan)
        best         = deepcopy(initial_plan)
        current_cost = evaluate_plan(current, vehicle_state, system_state, weights)
        best_cost    = current_cost
        temp         = self.initial_temp

        for _ in range(self.iterations):
            if time.time() - start_time > self.decision_time_limit:
                break

            candidate = self._neighbour(current)
            if candidate is None:
                continue   # operator couldn't produce a move; try again

            if not feasibility_checker(candidate, vehicle_state, system_state):
                continue

            candidate_cost = evaluate_plan(candidate, vehicle_state, system_state, weights)
            delta          = candidate_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current      = candidate
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best      = deepcopy(candidate)
                    best_cost = current_cost

            temp *= self.cooling_rate
            if temp < 1e-4:
                break

        initial_cost = evaluate_plan(initial_plan, vehicle_state, system_state, weights)
        if best_cost < initial_cost:
            print(f"    SA improved {vehicle_id}: {initial_cost:.2f} → {best_cost:.2f}")
            return best

        print(f"    SA no improvement for {vehicle_id} (best={best_cost:.2f}, initial={initial_cost:.2f})")
        return None