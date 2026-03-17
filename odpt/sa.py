# sa.py
# Simulated Annealing improvement policy.
#
# SAPolicy.propose() is the only public entry point.  It receives a
# frozen system_state snapshot and a feasibility_checker callable, then
# returns a dict of {vehicle_id: improved_plan} for improved vehicles.
# It never touches SimPy or live vehicle objects.
#
# Neighbour operators
# -------------------
# _pair_relocate : remove one request's PU+DO and re-insert at new i<j.
#                  Preserves precedence by construction.  Primary operator.
# _pair_swap     : swap the four stop positions of two requests (PU-a↔PU-b,
#                  DO-a↔DO-b).  Works on stop indices directly — no req_id
#                  mutation, no double-deepcopy bug from previous version.

import math
import random
import time
from copy import deepcopy
from typing import Callable

from feasibility import evaluate_plan


class SAPolicy:

    def __init__(
        self,
        initial_temp:        float = 10_000.0,
        cooling_rate:        float = 0.997,
        iterations:          int   = 8_000,
        decision_time_limit: float = 0.5,
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
        weights: tuple = (1.0, 2.0, 3.0),
    ) -> dict:
        """
        Propose improved plans for vehicles that have at least 2 complete
        request pairs (4 stops).

        Returns
        -------
        dict  vehicle_id -> improved plan  (only improved vehicles included)
        """
        start_time = time.time()
        new_plans  = {}

        for vehicle_id, vehicle in system_state["vehicles"].items():
            current_plan = vehicle["plan"]

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

    def _requests_in_plan(self, plan: list) -> list:
        """
        Return req_ids that have BOTH a PU and a DO stop in *plan*.
        Guards against partially-served requests (PU already popped by
        vehicle_process).
        """
        pu_ids = {s.req_id for s in plan if s.kind == "PU"}
        do_ids = {s.req_id for s in plan if s.kind == "DO"}
        return list(pu_ids & do_ids)

    def _pair_relocate(self, plan: list):
        """
        Remove one request's PU+DO pair and re-insert at new positions i<j.
        Precedence guaranteed by construction.
        """
        candidates = self._requests_in_plan(plan)
        if not candidates:
            return None

        req     = random.choice(candidates)
        pu_stop = None
        do_stop = None

        for s in plan:
            if s.req_id == req and s.kind == "PU":
                pu_stop = s
            elif s.req_id == req and s.kind == "DO":
                do_stop = s

        if pu_stop is None or do_stop is None:
            return None

        stripped = [s for s in plan if s.req_id != req]
        n = len(stripped)

        i = random.randint(0, n)
        j = random.randint(i + 1, n + 1)

        new_plan = deepcopy(stripped)
        new_plan.insert(i, deepcopy(pu_stop))
        new_plan.insert(j, deepcopy(do_stop))
        return new_plan

    def _pair_swap(self, plan: list):
        """
        Swap the stop-list positions of two requests' PU+DO pairs.

        Finds the four indices [pu_a, do_a, pu_b, do_b] and swaps the
        Stop objects at those positions directly.  No req_id mutation —
        the Stop objects (with their node/service/earliest data) simply
        move to each other's list positions.
        """
        candidates = self._requests_in_plan(plan)
        if len(candidates) < 2:
            return None

        req_a, req_b = random.sample(candidates, 2)

        idx_pu_a = idx_do_a = idx_pu_b = idx_do_b = None
        for i, s in enumerate(plan):
            if s.req_id == req_a:
                if s.kind == "PU":
                    idx_pu_a = i
                elif s.kind == "DO":
                    idx_do_a = i
            elif s.req_id == req_b:
                if s.kind == "PU":
                    idx_pu_b = i
                elif s.kind == "DO":
                    idx_do_b = i

        if any(x is None for x in [idx_pu_a, idx_do_a, idx_pu_b, idx_do_b]):
            return None

        new_plan = deepcopy(plan)
        new_plan[idx_pu_a], new_plan[idx_pu_b] = (
            deepcopy(plan[idx_pu_b]),
            deepcopy(plan[idx_pu_a]),
        )
        new_plan[idx_do_a], new_plan[idx_do_b] = (
            deepcopy(plan[idx_do_b]),
            deepcopy(plan[idx_do_a]),
        )
        return new_plan

    def _neighbour(self, plan: list):
        complete = self._requests_in_plan(plan)
        if len(complete) >= 2:
            op = random.choice([self._pair_relocate, self._pair_swap])
        else:
            op = self._pair_relocate  # only 1 complete pair, swap is impossible
        return op(plan)

    # ------------------------------------------------------------------
    # Core SA loop
    # ------------------------------------------------------------------

    def _run_sa(
        self,
        vehicle_id:          str,
        initial_plan:        list,
        system_state:        dict,
        feasibility_checker: Callable,
        weights:             tuple,
        start_time:          float,
    ):
        """
        Run SA for one vehicle.
        Returns the best feasible plan found, or None if no improvement.
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
                continue

            if not feasibility_checker(candidate, vehicle_state, system_state):
                continue

            candidate_cost = evaluate_plan(candidate, vehicle_state,
                                           system_state, weights)
            delta = candidate_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current      = candidate
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best      = deepcopy(candidate)
                    best_cost = current_cost

            temp *= self.cooling_rate
            if temp < 1e-4:
                break

        initial_cost = evaluate_plan(initial_plan, vehicle_state,
                                     system_state, weights)
        if best_cost < initial_cost:
            print(f"    SA improved {vehicle_id}: {initial_cost:.2f} -> {best_cost:.2f}")
            return best

        print(f"    SA no improvement for {vehicle_id} "
              f"(best={best_cost:.2f}, initial={initial_cost:.2f})")
        return None