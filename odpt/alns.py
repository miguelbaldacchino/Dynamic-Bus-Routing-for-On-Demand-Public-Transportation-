# alns.py
# Adaptive Large Neighbourhood Search. Destroy/repair operators with
# roulette-wheel weight adaptation. DARP-aware feasibility repair.
# Not runnable standalone — imported by dispatcher.py.

import math
import random
import time
from copy import deepcopy
from typing import Callable, Optional

from feasibility import check_feasibility, evaluate_plan


# ---------------------------------------------------------------------------
# Operator name constants
# ---------------------------------------------------------------------------

DESTROY_RANDOM = "random_removal"
DESTROY_WORST  = "worst_removal"
DESTROY_SHAW   = "shaw_removal"

REPAIR_GREEDY  = "greedy_insertion"
REPAIR_REGRET  = "regret2_insertion"

DESTROY_OPS = [DESTROY_RANDOM, DESTROY_WORST, DESTROY_SHAW]
REPAIR_OPS  = [REPAIR_GREEDY,  REPAIR_REGRET]

# Adaptive score bonuses (σ1..σ3, σ4=0 implicit)
SCORE_NEW_BEST   = 10
SCORE_IMPROVING  = 6
SCORE_ACCEPTED   = 2


class ALNSPolicy:

    def __init__(
        self,
        iterations:          int   = 150,
        q_min:               int   = 1,
        q_max:               int   = 6,
        reaction_factor:     float = 0.1,
        initial_temp_factor: float = 0.5,
        cooling_rate:        float = 0.992,
        decision_time_limit: float = 0.3,
        rng: Optional[random.Random] = None,
    ):
        """
        Parameters
        ----------
        iterations : int
            Max ALNS iterations per vehicle (time budget is the hard cap).
        q_min, q_max : int
            Range for number of requests removed per destroy operation.
        reaction_factor : float
            How fast operator weights adapt to performance (0=no adapt,
            1=fully replace with latest score).  Ropke & Pisinger use 0.1.
        initial_temp_factor : float
            Initial temperature = initial_temp_factor * initial_cost.
            Calibrates SA acceptance to the problem scale automatically.
        cooling_rate : float
            Temperature cooling per iteration.
        decision_time_limit : float
            Hard wall-clock limit per vehicle (seconds).
        """
        self.iterations          = iterations
        self.q_min               = q_min
        self.q_max               = q_max
        self.reaction_factor     = reaction_factor
        self.initial_temp_factor = initial_temp_factor
        self.cooling_rate        = cooling_rate
        self.decision_time_limit = decision_time_limit
        # Private RNG — isolated from the global state driving demand/noise.
        self.rng = rng if rng is not None else random.Random()

    # ==================================================================
    # Public interface — matches SAPolicy / GAPolicy / TSPolicy
    # ==================================================================

    def propose(
        self,
        system_state:        dict,
        feasibility_checker: Callable,
        weights:             tuple = (1.0, 2.0, 2.5),
    ) -> dict:
        """
        Propose improved plans for all vehicles using ALNS.

        Two-phase:
          Phase 1 — intra-vehicle ALNS (per vehicle, own time budget).
          Phase 2 — inter-vehicle ALNS (global destroy-repair across
                    all vehicles simultaneously).

        Returns dict { vehicle_id: improved_plan } — only improved.
        """
        new_plans = {}

        # Phase 1 — intra-vehicle
        for vehicle_id, vehicle in system_state["vehicles"].items():
            current_plan = vehicle["plan"]
            n_committed  = vehicle.get("n_committed", 0)

            if len(current_plan) - n_committed < 4:
                continue

            improved = self._run_alns_intra(
                vehicle_id, current_plan, n_committed,
                system_state, feasibility_checker, weights,
            )
            if improved is not None:
                new_plans[vehicle_id] = improved

        # Phase 2 — inter-vehicle
        vehicle_ids = list(system_state["vehicles"].keys())
        if len(vehicle_ids) >= 2:
            inter_results = self._run_alns_inter(
                vehicle_ids, system_state, feasibility_checker, weights,
                existing_improvements=new_plans,
            )
            new_plans.update(inter_results)

        return new_plans

    # ==================================================================
    # Adaptive operator selection
    # ==================================================================

    def _roulette_select(self, ops: list[str], weights: dict[str, float]) -> str:
        """Select one operator via roulette-wheel proportional to weights."""
        total = sum(weights[op] for op in ops)
        r = self.rng.uniform(0, total)
        cumulative = 0.0
        for op in ops:
            cumulative += weights[op]
            if r <= cumulative:
                return op
        return ops[-1]

    @staticmethod
    def _update_weights(
        weights:  dict[str, float],
        op:       str,
        score:    float,
        r:        float,
    ) -> None:
        """
        Exponential smoothing weight update.
        w_i ← (1 - r) * w_i + r * score
        Weights never fall below 0.01 to ensure all operators stay
        reachable (floor avoids dead operators).
        """
        weights[op] = max(0.01, (1 - r) * weights[op] + r * score)

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _requests_in_plan(plan: list, n_committed: int = 0) -> list:
        """Return req_ids with BOTH PU and DO in the movable portion."""
        movable = plan[n_committed:]
        pu_ids  = {s.req_id for s in movable if s.kind == "PU"}
        do_ids  = {s.req_id for s in movable if s.kind == "DO"}
        return list(pu_ids & do_ids)

    @staticmethod
    def _extract_request_stops(
        plan: list, n_committed: int, req_id: str
    ) -> tuple:
        """
        Return (pu_stop, do_stop, stripped_plan) — the plan with
        this request's PU and DO removed from the movable portion.
        """
        committed = plan[:n_committed]
        movable   = plan[n_committed:]
        pu_stop = do_stop = None
        for s in movable:
            if s.req_id == req_id:
                if s.kind == "PU":
                    pu_stop = s
                elif s.kind == "DO":
                    do_stop = s
        stripped = [s for s in movable if s.req_id != req_id]
        return pu_stop, do_stop, committed + stripped

    @staticmethod
    def _best_insertion_cost(
        pu_stop,
        do_stop,
        plan:         list,
        n_committed:  int,
        vehicle_state: dict,
        system_state: dict,
        weights:      tuple,
        feasibility_checker: Callable,
    ) -> tuple[float, Optional[list]]:
        """
        Find the best feasible (i,j) insertion for (pu_stop, do_stop)
        in plan.  Returns (best_cost, best_plan) or (inf, None).
        """
        committed  = plan[:n_committed]
        insertable = plan[n_committed:]
        n = len(insertable)

        best_cost = float("inf")
        best_plan = None

        for i in range(n + 1):
            for j in range(i + 1, n + 2):
                candidate = list(insertable)
                candidate.insert(i, pu_stop)
                candidate.insert(j, do_stop)
                full = committed + candidate

                if not feasibility_checker(full, vehicle_state, system_state):
                    continue

                cost = evaluate_plan(full, vehicle_state, system_state, weights)
                if cost < best_cost:
                    best_cost = cost
                    best_plan = full

        return best_cost, best_plan

    # ==================================================================
    # Destroy operators
    # ==================================================================

    def _destroy_random(
        self,
        plans:       dict[str, list],
        n_committed: dict[str, int],
        q:           int,
    ) -> tuple[list, dict]:
        """
        Remove q randomly chosen requests from plans.
        Returns (removed_list, new_plans) where removed_list is a list
        of (req_id, pu_stop, do_stop) tuples.
        """
        # Collect all removable (req_id, vehicle_id) pairs
        pool = []
        for vid, plan in plans.items():
            for req_id in self._requests_in_plan(plan, n_committed.get(vid, 0)):
                pool.append((req_id, vid))

        if not pool:
            return [], dict(plans)

        q_actual = min(q, len(pool))
        chosen   = self.rng.sample(pool, q_actual)

        removed  = []
        new_plans = {vid: list(p) for vid, p in plans.items()}

        for req_id, vid in chosen:
            nc = n_committed.get(vid, 0)
            pu, do, stripped = self._extract_request_stops(
                new_plans[vid], nc, req_id
            )
            if pu is not None and do is not None:
                new_plans[vid] = stripped
                removed.append((req_id, pu, do))

        return removed, new_plans

    def _destroy_worst(
        self,
        plans:        dict[str, list],
        n_committed:  dict[str, int],
        q:            int,
        system_state: dict,
        weights:      tuple,
    ) -> tuple[list, dict]:
        """
        Remove the q requests that contribute the most cost.
        Marginal cost of request r = cost(plan) - cost(plan without r).
        """
        # Score every removable request by its marginal cost saving
        scored = []
        for vid, plan in plans.items():
            nc = n_committed.get(vid, 0)
            v_state = system_state["vehicles"][vid]
            base_cost = evaluate_plan(plan, v_state, system_state, weights)

            for req_id in self._requests_in_plan(plan, nc):
                pu, do, stripped = self._extract_request_stops(plan, nc, req_id)
                if pu is None or do is None:
                    continue
                stripped_cost = evaluate_plan(
                    stripped, v_state, system_state, weights
                )
                saving = base_cost - stripped_cost  # positive = expensive request
                scored.append((saving, req_id, vid))

        if not scored:
            return [], dict(plans)

        # Sort descending by saving; take top q with small noise for diversity
        scored.sort(key=lambda x: -x[0])
        q_actual = min(q, len(scored))

        removed   = []
        new_plans = {vid: list(p) for vid, p in plans.items()}
        removed_ids = set()

        for saving, req_id, vid in scored:
            if len(removed) >= q_actual:
                break
            if req_id in removed_ids:
                continue
            nc = n_committed.get(vid, 0)
            pu, do, stripped = self._extract_request_stops(
                new_plans[vid], nc, req_id
            )
            if pu is not None and do is not None:
                new_plans[vid] = stripped
                removed.append((req_id, pu, do))
                removed_ids.add(req_id)

        return removed, new_plans

    def _destroy_shaw(
        self,
        plans:        dict[str, list],
        n_committed:  dict[str, int],
        q:            int,
        system_state: dict,
    ) -> tuple[list, dict]:
        """
        Shaw removal: remove q requests that are relationally similar
        to a randomly chosen seed request.

        Relatedness(r1, r2) combines:
          - pickup distance normalised by max pairwise distance
          - |earliest_r1 - earliest_r2| normalised by max_wait
          - 1 if assigned to the same vehicle (they compete for capacity)

        Lower relatedness score = more similar = selected first.
        Shaw (1998) CP-98; Ropke & Pisinger (2006) Sec 3.2.
        """
        travel_fn = system_state["travel_time"]
        max_wait  = system_state.get("max_wait", 30.0)

        # Build flat list of (req_id, vid) for all movable requests
        pool = []
        req_to_vid = {}
        for vid, plan in plans.items():
            for req_id in self._requests_in_plan(plan, n_committed.get(vid, 0)):
                pool.append((req_id, vid))
                req_to_vid[req_id] = vid

        if not pool:
            return [], dict(plans)

        # Build lookup: req_id -> (pu_stop, do_stop)
        stop_map = {}
        for vid, plan in plans.items():
            nc = n_committed.get(vid, 0)
            movable = plan[nc:]
            pu_lookup = {s.req_id: s for s in movable if s.kind == "PU"}
            do_lookup = {s.req_id: s for s in movable if s.kind == "DO"}
            for req_id in pu_lookup:
                if req_id in do_lookup:
                    stop_map[req_id] = (pu_lookup[req_id], do_lookup[req_id])

        if not stop_map:
            return [], dict(plans)

        # Estimate max pairwise pickup distance for normalisation
        sample_reqs = list(stop_map.keys())
        max_dist = 1.0
        if len(sample_reqs) >= 2:
            for _ in range(min(20, len(sample_reqs))):
                r1, r2 = self.rng.sample(sample_reqs, 2)
                d = travel_fn(
                    stop_map[r1][0].node,
                    stop_map[r2][0].node, 0.0,
                )
                if d > max_dist:
                    max_dist = d

        # Choose seed request
        seed_req, seed_vid = self.rng.choice(pool)
        if seed_req not in stop_map:
            return [], dict(plans)

        seed_pu = stop_map[seed_req][0]

        def relatedness(req_id: str) -> float:
            if req_id not in stop_map:
                return float("inf")
            pu = stop_map[req_id][0]
            dist_norm = travel_fn(seed_pu.node, pu.node, 0.0) / max_dist
            tw_diff   = abs(seed_pu.earliest - pu.earliest) / max(max_wait, 1.0) \
                        if seed_pu.earliest and pu.earliest else 0.0
            same_veh  = 1.0 if req_to_vid.get(req_id) == seed_vid else 0.0
            # Weights φ=0.4, χ=0.4, ψ=0.2 (typical Shaw values)
            return 0.4 * dist_norm + 0.4 * tw_diff + 0.2 * same_veh

        # Sort all other requests by relatedness to seed
        others = [(relatedness(req_id), req_id, vid)
                  for req_id, vid in pool if req_id != seed_req]
        others.sort(key=lambda x: x[0])

        # Build removal set: seed + most similar (up to q)
        to_remove = [seed_req] + [req_id for _, req_id, _ in others]
        to_remove  = to_remove[:q]

        removed   = []
        new_plans = {vid: list(p) for vid, p in plans.items()}

        for req_id in to_remove:
            vid = req_to_vid.get(req_id)
            if vid is None:
                continue
            nc = n_committed.get(vid, 0)
            pu, do, stripped = self._extract_request_stops(
                new_plans[vid], nc, req_id
            )
            if pu is not None and do is not None:
                new_plans[vid] = stripped
                removed.append((req_id, pu, do))

        return removed, new_plans

    # ==================================================================
    # Repair operators
    # ==================================================================

    def _repair_greedy(
        self,
        removed:      list,
        plans:        dict[str, list],
        n_committed:  dict[str, int],
        vehicle_ids:  list[str],
        system_state: dict,
        weights:      tuple,
        feasibility_checker: Callable,
    ) -> Optional[dict]:
        """
        Greedily reinsert removed requests one at a time.
        For each request, find the cheapest feasible (vehicle, i, j)
        across all vehicles and apply it.  If any request cannot be
        inserted anywhere, the repair fails (returns None).
        """
        new_plans = {vid: list(p) for vid, p in plans.items()}

        # Shuffle to remove ordering bias
        order = list(removed)
        self.rng.shuffle(order)

        for req_id, pu_stop, do_stop in order:
            best_cost = float("inf")
            best_vid  = None
            best_plan = None

            for vid in vehicle_ids:
                nc = n_committed.get(vid, 0)
                v_state = system_state["vehicles"][vid]
                cost, candidate = self._best_insertion_cost(
                    pu_stop, do_stop,
                    new_plans[vid], nc,
                    v_state, system_state, weights,
                    feasibility_checker,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_vid  = vid
                    best_plan = candidate

            if best_plan is None:
                return None  # infeasible repair

            new_plans[best_vid] = best_plan

        return new_plans

    def _repair_regret2(
        self,
        removed:      list,
        plans:        dict[str, list],
        n_committed:  dict[str, int],
        vehicle_ids:  list[str],
        system_state: dict,
        weights:      tuple,
        feasibility_checker: Callable,
    ) -> Optional[dict]:
        """
        Regret-2 insertion: at each step insert the request with the
        highest regret = (2nd-best insertion cost) - (best insertion cost).

        High regret means the request will suffer most if not inserted
        now.  This prevents greedy short-sightedness.

        Ropke & Pisinger (2006) show regret-k dominates greedy on most
        PDPTW instances, especially when q is large.
        """
        new_plans = {vid: list(p) for vid, p in plans.items()}
        remaining = list(removed)

        while remaining:
            # Compute best and 2nd-best insertion cost for each request
            regrets = []

            for req_id, pu_stop, do_stop in remaining:
                costs_per_vehicle = []

                for vid in vehicle_ids:
                    nc = n_committed.get(vid, 0)
                    v_state = system_state["vehicles"][vid]
                    cost, _ = self._best_insertion_cost(
                        pu_stop, do_stop,
                        new_plans[vid], nc,
                        v_state, system_state, weights,
                        feasibility_checker,
                    )
                    if cost < float("inf"):
                        costs_per_vehicle.append((cost, vid))

                if not costs_per_vehicle:
                    # Cannot insert anywhere — repair fails
                    return None

                costs_per_vehicle.sort(key=lambda x: x[0])
                best_cost  = costs_per_vehicle[0][0]
                second_cost = costs_per_vehicle[1][0] \
                              if len(costs_per_vehicle) > 1 else best_cost
                regret = second_cost - best_cost
                best_vid = costs_per_vehicle[0][1]

                regrets.append((regret, req_id, pu_stop, do_stop, best_vid))

            # Insert the request with the highest regret
            regrets.sort(key=lambda x: -x[0])
            _, req_id, pu_stop, do_stop, best_vid = regrets[0]

            nc = n_committed.get(best_vid, 0)
            v_state = system_state["vehicles"][best_vid]
            _, best_plan = self._best_insertion_cost(
                pu_stop, do_stop,
                new_plans[best_vid], nc,
                v_state, system_state, weights,
                feasibility_checker,
            )

            if best_plan is None:
                return None

            new_plans[best_vid] = best_plan
            remaining = [(r, pu, do) for r, pu, do in remaining if r != req_id]

        return new_plans

    # ==================================================================
    # Intra-vehicle ALNS loop
    # ==================================================================

    def _run_alns_intra(
        self,
        vehicle_id:          str,
        initial_plan:        list,
        n_committed:         int,
        system_state:        dict,
        feasibility_checker: Callable,
        weights:             tuple,
    ) -> Optional[list]:
        """
        Run ALNS for one vehicle (intra-vehicle destroy/repair only).
        Returns the best feasible plan found, or None if no improvement.

        For intra-vehicle ALNS, "all vehicles" = just this one.
        The destroy operators remove requests from this vehicle only,
        and the repair operators reinsert them into this vehicle only.
        If a request cannot be reinserted, the iteration is discarded.
        """
        vehicle_state = system_state["vehicles"][vehicle_id]
        start_time    = time.time()

        movable_reqs = self._requests_in_plan(initial_plan, n_committed)
        if len(movable_reqs) < 2:
            return None

        initial_cost = evaluate_plan(
            initial_plan, vehicle_state, system_state, weights
        )
        if initial_cost == 0:
            return None

        # Working state: single-vehicle "fleet"
        plans       = {vehicle_id: list(initial_plan)}
        n_comm      = {vehicle_id: n_committed}
        vehicle_ids = [vehicle_id]

        current_plans = {vehicle_id: list(initial_plan)}
        best_plans    = {vehicle_id: list(initial_plan)}
        current_cost  = initial_cost
        best_cost     = initial_cost

        # Operator weights (initialised uniformly)
        d_weights = {op: 1.0 for op in DESTROY_OPS}
        r_weights = {op: 1.0 for op in REPAIR_OPS}

        temp = self.initial_temp_factor * initial_cost

        for _ in range(self.iterations):
            if time.time() - start_time > self.decision_time_limit:
                break

            # Number of requests to remove
            n_movable = len(self._requests_in_plan(
                current_plans[vehicle_id], n_committed
            ))
            if n_movable < 2:
                break
            q = self.rng.randint(
                self.q_min,
                min(self.q_max, max(1, n_movable - 1)),
            )

            # Select and apply destroy operator
            d_op = self._roulette_select(DESTROY_OPS, d_weights)

            if d_op == DESTROY_RANDOM:
                removed, destroyed = self._destroy_random(
                    current_plans, n_comm, q
                )
            elif d_op == DESTROY_WORST:
                removed, destroyed = self._destroy_worst(
                    current_plans, n_comm, q, system_state, weights
                )
            else:  # SHAW
                removed, destroyed = self._destroy_shaw(
                    current_plans, n_comm, q, system_state
                )

            if not removed:
                continue

            # Select and apply repair operator
            r_op = self._roulette_select(REPAIR_OPS, r_weights)

            if r_op == REPAIR_GREEDY:
                repaired = self._repair_greedy(
                    removed, destroyed, n_comm, vehicle_ids,
                    system_state, weights, feasibility_checker,
                )
            else:  # REGRET-2
                repaired = self._repair_regret2(
                    removed, destroyed, n_comm, vehicle_ids,
                    system_state, weights, feasibility_checker,
                )

            if repaired is None:
                # Infeasible repair — penalise operator, skip
                self._update_weights(d_weights, d_op, 0.0, self.reaction_factor)
                self._update_weights(r_weights, r_op, 0.0, self.reaction_factor)
                continue

            new_cost = evaluate_plan(
                repaired[vehicle_id], vehicle_state, system_state, weights
            )
            delta = new_cost - current_cost

            # Acceptance (SA-style)
            accepted = False
            if delta < 0:
                accepted = True
            elif temp > 1e-6 and self.rng.random() < math.exp(-delta / temp):
                accepted = True

            # Score the operators
            if new_cost < best_cost:
                score = SCORE_NEW_BEST
                best_plans = {vehicle_id: list(repaired[vehicle_id])}
                best_cost  = new_cost
            elif accepted and delta < 0:
                score = SCORE_IMPROVING
            elif accepted:
                score = SCORE_ACCEPTED
            else:
                score = 0

            self._update_weights(d_weights, d_op, score, self.reaction_factor)
            self._update_weights(r_weights, r_op, score, self.reaction_factor)

            if accepted:
                current_plans = repaired
                current_cost  = new_cost

            temp *= self.cooling_rate

        if best_cost < initial_cost:
            return best_plans[vehicle_id]
        return None

    # ==================================================================
    # Inter-vehicle ALNS (Phase 2)
    # ==================================================================

    def _run_alns_inter(
        self,
        vehicle_ids:           list,
        system_state:          dict,
        feasibility_checker:   Callable,
        weights:               tuple,
        existing_improvements: dict,
    ) -> dict:
        """
        Run ALNS across all vehicles simultaneously.

        This is where ALNS most clearly outperforms SA/TS: destroy
        operators can remove requests from multiple vehicles in one
        step, and repair operators can reinsert them anywhere — enabling
        improvements that require moving several requests simultaneously.

        Returns dict { vehicle_id: improved_plan } for improved vehicles.
        """
        start_time = time.time()

        # Build working copies
        working = {}
        for vid in vehicle_ids:
            if vid in existing_improvements:
                working[vid] = list(existing_improvements[vid])
            else:
                working[vid] = list(system_state["vehicles"][vid]["plan"])

        n_comm = {
            vid: system_state["vehicles"][vid].get("n_committed", 0)
            for vid in vehicle_ids
        }

        # Compute initial combined cost
        initial_total = sum(
            evaluate_plan(
                working[vid],
                system_state["vehicles"][vid],
                system_state, weights,
            )
            for vid in vehicle_ids
        )
        if initial_total == 0:
            return {}

        current_plans = {vid: list(working[vid]) for vid in vehicle_ids}
        best_plans    = {vid: list(working[vid]) for vid in vehicle_ids}
        current_cost  = initial_total
        best_cost     = initial_total

        d_weights = {op: 1.0 for op in DESTROY_OPS}
        r_weights = {op: 1.0 for op in REPAIR_OPS}

        temp = self.initial_temp_factor * initial_total

        n_attempts = min(self.iterations, 1000)

        for _ in range(n_attempts):
            if time.time() - start_time > self.decision_time_limit:
                break

            # Total removable across all vehicles
            total_movable = sum(
                len(self._requests_in_plan(current_plans[vid], n_comm[vid]))
                for vid in vehicle_ids
            )
            if total_movable < 2:
                break

            q = self.rng.randint(
                self.q_min,
                min(self.q_max, max(1, total_movable - 1)),
            )

            # Destroy
            d_op = self._roulette_select(DESTROY_OPS, d_weights)

            if d_op == DESTROY_RANDOM:
                removed, destroyed = self._destroy_random(
                    current_plans, n_comm, q
                )
            elif d_op == DESTROY_WORST:
                removed, destroyed = self._destroy_worst(
                    current_plans, n_comm, q, system_state, weights
                )
            else:
                removed, destroyed = self._destroy_shaw(
                    current_plans, n_comm, q, system_state
                )

            if not removed:
                continue

            # Repair
            r_op = self._roulette_select(REPAIR_OPS, r_weights)

            if r_op == REPAIR_GREEDY:
                repaired = self._repair_greedy(
                    removed, destroyed, n_comm, vehicle_ids,
                    system_state, weights, feasibility_checker,
                )
            else:
                repaired = self._repair_regret2(
                    removed, destroyed, n_comm, vehicle_ids,
                    system_state, weights, feasibility_checker,
                )

            if repaired is None:
                self._update_weights(d_weights, d_op, 0.0, self.reaction_factor)
                self._update_weights(r_weights, r_op, 0.0, self.reaction_factor)
                continue

            new_cost = sum(
                evaluate_plan(
                    repaired[vid],
                    system_state["vehicles"][vid],
                    system_state, weights,
                )
                for vid in vehicle_ids
            )
            delta = new_cost - current_cost

            accepted = False
            if delta < 0:
                accepted = True
            elif temp > 1e-6 and self.rng.random() < math.exp(-delta / temp):
                accepted = True

            if new_cost < best_cost:
                score      = SCORE_NEW_BEST
                best_plans = {vid: list(repaired[vid]) for vid in vehicle_ids}
                best_cost  = new_cost
            elif accepted and delta < 0:
                score = SCORE_IMPROVING
            elif accepted:
                score = SCORE_ACCEPTED
            else:
                score = 0

            self._update_weights(d_weights, d_op, score, self.reaction_factor)
            self._update_weights(r_weights, r_op, score, self.reaction_factor)

            if accepted:
                current_plans = repaired
                current_cost  = new_cost

            temp *= self.cooling_rate

        # Return only vehicles whose plan actually improved
        if best_cost >= initial_total:
            return {}

        improvements = {}
        for vid in vehicle_ids:
            orig = working[vid]
            improved = best_plans[vid]
            v_state  = system_state["vehicles"][vid]
            orig_cost = evaluate_plan(orig, v_state, system_state, weights)
            new_cost  = evaluate_plan(improved, v_state, system_state, weights)
            if improved != orig or new_cost < orig_cost:
                improvements[vid] = improved

        return improvements