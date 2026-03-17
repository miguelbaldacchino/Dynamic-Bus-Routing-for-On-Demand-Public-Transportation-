# sa.py
# Simulated Annealing improvement policy.
#
# SAPolicy.propose() is the only public entry point.  It receives a
# frozen system_state snapshot and a feasibility_checker callable, then
# returns a dict of {vehicle_id: improved_plan} for improved vehicles.
# It never touches SimPy or live vehicle objects.
#
# Fixes applied
# -------------
# - Per-vehicle time budget (start_time reset per vehicle, not shared).
# - Inter-vehicle relocate operator: moves a request PU+DO pair from
#   one vehicle to another, enabling cross-vehicle optimisation.
# - Reduced deepcopy: neighbour operators use list() + shallow copies
#   where possible; only the best solution is deep-copied.
# - n_committed: operators do not touch committed (in-transit) stops.
#
# Neighbour operators
# -------------------
# _pair_relocate     : remove one request's PU+DO and re-insert at new
#                      positions i<j within the same vehicle.
# _pair_swap         : swap positions of two requests' PU+DO pairs.
# _inter_vehicle_move: move a request from one vehicle to another.

import math
import random
import time
from copy import deepcopy
from typing import Callable

from feasibility import evaluate_plan


class SAPolicy:

    def __init__(
        self,
        initial_temp:        float = 5_000.0,
        cooling_rate:        float = 0.995,
        iterations:          int   = 5_000,
        decision_time_limit: float = 0.3,
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
        weights: tuple = (1.0, 2.0, 2.5),
    ) -> dict:
        """
        Propose improved plans for vehicles.

        Two-phase approach:
          Phase 1: intra-vehicle SA for each vehicle with >=4 stops.
          Phase 2: inter-vehicle SA across pairs of vehicles.

        Each vehicle / pair gets its own independent time budget.

        Returns
        -------
        dict  vehicle_id -> improved plan  (only improved vehicles)
        """
        new_plans = {}

        # Phase 1 — intra-vehicle improvement
        for vehicle_id, vehicle in system_state["vehicles"].items():
            current_plan = vehicle["plan"]
            n_committed  = vehicle.get("n_committed", 0)

            # Need at least 2 movable request pairs (4 stops after committed)
            if len(current_plan) - n_committed < 4:
                continue

            improved = self._run_sa_intra(
                vehicle_id, current_plan, n_committed,
                system_state, feasibility_checker, weights,
            )
            if improved is not None:
                new_plans[vehicle_id] = improved

        # Phase 2 — inter-vehicle moves
        vehicle_ids = list(system_state["vehicles"].keys())
        if len(vehicle_ids) >= 2:
            inter_results = self._run_sa_inter(
                vehicle_ids, system_state, feasibility_checker, weights,
                existing_improvements=new_plans,
            )
            new_plans.update(inter_results)

        return new_plans

    # ------------------------------------------------------------------
    # Helper: movable requests in a plan
    # ------------------------------------------------------------------

    def _requests_in_plan(self, plan: list, n_committed: int = 0) -> list:
        """
        Return req_ids that have BOTH a PU and DO in the movable portion
        of the plan (after n_committed leading stops).
        """
        movable = plan[n_committed:]
        pu_ids = {s.req_id for s in movable if s.kind == "PU"}
        do_ids = {s.req_id for s in movable if s.kind == "DO"}
        return list(pu_ids & do_ids)

    # ------------------------------------------------------------------
    # Intra-vehicle neighbour operators
    # ------------------------------------------------------------------

    def _pair_relocate(self, plan: list, n_committed: int):
        """
        Remove one request's PU+DO pair from the movable portion and
        re-insert at new random positions i<j (after committed prefix).
        """
        candidates = self._requests_in_plan(plan, n_committed)
        if not candidates:
            return None

        req = random.choice(candidates)
        committed = plan[:n_committed]
        movable   = plan[n_committed:]

        pu_stop = do_stop = None
        for s in movable:
            if s.req_id == req and s.kind == "PU":
                pu_stop = s
            elif s.req_id == req and s.kind == "DO":
                do_stop = s

        if pu_stop is None or do_stop is None:
            return None

        stripped = [s for s in movable if s.req_id != req]
        n = len(stripped)

        i = random.randint(0, n)
        j = random.randint(i + 1, n + 1)

        new_movable = list(stripped)
        new_movable.insert(i, pu_stop)
        new_movable.insert(j, do_stop)
        return committed + new_movable

    def _pair_swap(self, plan: list, n_committed: int):
        """
        Swap the positions of two requests' PU+DO pairs within the
        movable portion of the plan.
        """
        candidates = self._requests_in_plan(plan, n_committed)
        if len(candidates) < 2:
            return None

        req_a, req_b = random.sample(candidates, 2)

        # Work on the full plan but only swap within movable portion
        idx_pu_a = idx_do_a = idx_pu_b = idx_do_b = None
        for i, s in enumerate(plan):
            if i < n_committed:
                continue
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

        new_plan = list(plan)
        new_plan[idx_pu_a], new_plan[idx_pu_b] = plan[idx_pu_b], plan[idx_pu_a]
        new_plan[idx_do_a], new_plan[idx_do_b] = plan[idx_do_b], plan[idx_do_a]
        return new_plan

    def _neighbour_intra(self, plan: list, n_committed: int):
        complete = self._requests_in_plan(plan, n_committed)
        if len(complete) >= 2:
            op = random.choice([self._pair_relocate, self._pair_swap])
        else:
            op = self._pair_relocate
        return op(plan, n_committed)

    # ------------------------------------------------------------------
    # Intra-vehicle SA loop
    # ------------------------------------------------------------------

    def _run_sa_intra(
        self,
        vehicle_id:          str,
        initial_plan:        list,
        n_committed:         int,
        system_state:        dict,
        feasibility_checker: Callable,
        weights:             tuple,
    ):
        """
        Run SA for one vehicle (intra-vehicle moves only).
        Returns the best feasible plan found, or None if no improvement.
        Each call gets its own fresh time budget.
        """
        vehicle_state = system_state["vehicles"][vehicle_id]
        start_time    = time.time()  # per-vehicle budget

        current      = list(initial_plan)
        best         = list(initial_plan)
        current_cost = evaluate_plan(current, vehicle_state, system_state, weights)
        best_cost    = current_cost
        temp         = self.initial_temp

        for _ in range(self.iterations):
            if time.time() - start_time > self.decision_time_limit:
                break

            candidate = self._neighbour_intra(current, n_committed)
            if candidate is None:
                continue

            if not feasibility_checker(candidate, vehicle_state, system_state):
                continue

            candidate_cost = evaluate_plan(
                candidate, vehicle_state, system_state, weights
            )
            delta = candidate_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current      = candidate
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best      = list(candidate)
                    best_cost = current_cost

            temp *= self.cooling_rate
            if temp < 1e-4:
                break

        initial_cost = evaluate_plan(
            initial_plan, vehicle_state, system_state, weights
        )
        if best_cost < initial_cost:
            return best
        return None

    # ------------------------------------------------------------------
    # Inter-vehicle SA
    # ------------------------------------------------------------------

    def _run_sa_inter(
        self,
        vehicle_ids:            list,
        system_state:           dict,
        feasibility_checker:    Callable,
        weights:                tuple,
        existing_improvements:  dict,
    ) -> dict:
        """
        Try moving requests between vehicle pairs.
        Uses a quick random-sampling approach with SA acceptance.
        Returns dict of vehicle_id -> improved plan.
        """
        start_time = time.time()
        improvements = {}

        # Build working copies of plans (incorporate any intra improvements)
        working = {}
        for vid in vehicle_ids:
            if vid in existing_improvements:
                working[vid] = list(existing_improvements[vid])
            else:
                working[vid] = list(system_state["vehicles"][vid]["plan"])

        # Quick inter-vehicle pass
        n_attempts = min(self.iterations // 2, 2000)
        temp = self.initial_temp

        for _ in range(n_attempts):
            if time.time() - start_time > self.decision_time_limit:
                break

            # Pick a source vehicle with movable requests
            src_vid = random.choice(vehicle_ids)
            src_info = system_state["vehicles"][src_vid]
            n_committed_src = src_info.get("n_committed", 0)
            movable = self._requests_in_plan(working[src_vid], n_committed_src)

            if not movable:
                continue

            # Pick a destination vehicle (different from source)
            dst_vid = random.choice(vehicle_ids)
            if dst_vid == src_vid:
                continue

            dst_info = system_state["vehicles"][dst_vid]
            n_committed_dst = dst_info.get("n_committed", 0)

            # Pick a request to move
            req = random.choice(movable)

            # Remove PU+DO from source
            src_committed = working[src_vid][:n_committed_src]
            src_movable   = working[src_vid][n_committed_src:]
            pu_stop = do_stop = None
            for s in src_movable:
                if s.req_id == req and s.kind == "PU":
                    pu_stop = s
                elif s.req_id == req and s.kind == "DO":
                    do_stop = s

            if pu_stop is None or do_stop is None:
                continue

            new_src_movable = [s for s in src_movable if s.req_id != req]
            new_src = src_committed + new_src_movable

            # Insert into destination at random positions
            dst_committed = working[dst_vid][:n_committed_dst]
            dst_movable   = working[dst_vid][n_committed_dst:]
            n_dst = len(dst_movable)

            i = random.randint(0, n_dst)
            j = random.randint(i + 1, n_dst + 1)

            new_dst_movable = list(dst_movable)
            new_dst_movable.insert(i, pu_stop)
            new_dst_movable.insert(j, do_stop)
            new_dst = dst_committed + new_dst_movable

            # Check feasibility of both new plans
            if not feasibility_checker(new_src, src_info, system_state):
                continue
            if not feasibility_checker(new_dst, dst_info, system_state):
                continue

            # Compute cost delta (sum of both vehicles)
            old_cost = (
                evaluate_plan(working[src_vid], src_info, system_state, weights)
                + evaluate_plan(working[dst_vid], dst_info, system_state, weights)
            )
            new_cost = (
                evaluate_plan(new_src, src_info, system_state, weights)
                + evaluate_plan(new_dst, dst_info, system_state, weights)
            )
            delta = new_cost - old_cost

            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                working[src_vid] = new_src
                working[dst_vid] = new_dst
                improvements[src_vid] = new_src
                improvements[dst_vid] = new_dst

            temp *= self.cooling_rate
            if temp < 1e-4:
                break

        # Compare total cost of ALL touched vehicles: return all or none.
        # Inter-vehicle moves are coupled — one vehicle's cost may rise
        # while the other's falls.  Only the combined delta matters.
        touched = set(improvements.keys())
        if not touched:
            return {}

        old_total = sum(
            evaluate_plan(
                system_state["vehicles"][vid]["plan"],
                system_state["vehicles"][vid],
                system_state, weights,
            )
            for vid in touched
        )
        new_total = sum(
            evaluate_plan(
                working[vid],
                system_state["vehicles"][vid],
                system_state, weights,
            )
            for vid in touched
        )

        if new_total < old_total:
            return {vid: working[vid] for vid in touched}
        return {}