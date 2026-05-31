# ts.py
# Tabu Search route optimiser. Neighbourhood: 2-opt + or-opt moves
# with recency-based tabu tenure.
# Not runnable standalone — imported by dispatcher.py.

import random
import time
from copy import deepcopy
from typing import Callable, Optional

from feasibility import evaluate_plan


# ---------------------------------------------------------------------------
# Move attribute type (named tuple would also work but plain tuple is faster)
# ---------------------------------------------------------------------------
# move = (req_id: str, pu_pos: int, do_pos: int)
# pu_pos and do_pos are positions in the movable (post-committed) slice.


class TSPolicy:

    def __init__(
        self,
        tabu_tenure:         int   = 7,
        max_neighbours:      int   = 50,
        iterations:          int   = 200,
        patience:            int   = 30,
        decision_time_limit: float = 0.3,
        rng: Optional[random.Random] = None,
    ):
        """
        Parameters
        ----------
        tabu_tenure : int
            Number of iterations a move stays tabu.  Cordeau & Laporte
            (2003) use dynamic tenure in [5, 10]; 7 is a good static
            default for your problem scale (6-20 stops/vehicle).
        max_neighbours : int
            Maximum number of neighbour plans evaluated per iteration.
            Caps the O(n²) pair_relocate neighbourhood.  For 20 stops
            (10 requests), full pair_relocate = ~90 moves; cap at 50
            for time-budget compliance.
        iterations : int
            Maximum TS iterations per vehicle.  With 0.3s budget and
            ~1ms per iteration (50 neighbours × feasibility check),
            300 iterations is the natural cap; 200 is conservative.
        patience : int
            Iterations without improvement before diversification
            (random restart from best-known solution).
        decision_time_limit : float
            Hard wall-clock limit per vehicle (seconds).  Same as SA/GA
            for fair comparison.
        """
        self.tabu_tenure         = tabu_tenure
        self.max_neighbours      = max_neighbours
        self.iterations          = iterations
        self.patience            = patience
        self.decision_time_limit = decision_time_limit
        # Private RNG — isolated from the global state driving demand/noise.
        self.rng = rng if rng is not None else random.Random()

    # ==================================================================
    # Public interface — matches SAPolicy.propose / GAPolicy.propose
    # ==================================================================

    def propose(
        self,
        system_state:        dict,
        feasibility_checker: Callable,
        weights:             tuple = (1.0, 2.0, 2.5),
    ) -> dict:
        """
        Propose improved plans for all vehicles.

        Two-phase approach:
          Phase 1: intra-vehicle TS for each vehicle with >=4 movable stops.
          Phase 2: inter-vehicle request transfer (greedy acceptance).

        Parameters
        ----------
        system_state : dict
            Frozen snapshot — "vehicles" sub-dict with plan, n_committed,
            capacity, location, time, onboard_count, onboard_pickup_times.
        feasibility_checker : callable
            check_feasibility(plan, vehicle_state, system_state) -> bool
        weights : (alpha, beta, gamma) for evaluate_plan

        Returns
        -------
        dict  { vehicle_id: improved_plan }  — only improved vehicles
        """
        new_plans = {}

        # Phase 1 — intra-vehicle TS
        for vehicle_id, vehicle in system_state["vehicles"].items():
            current_plan = vehicle["plan"]
            n_committed  = vehicle.get("n_committed", 0)

            # Need at least 2 movable request pairs (4 stops)
            if len(current_plan) - n_committed < 4:
                continue

            improved = self._run_ts_intra(
                vehicle_id, current_plan, n_committed,
                system_state, feasibility_checker, weights,
            )
            if improved is not None:
                new_plans[vehicle_id] = improved

        # Phase 2 — inter-vehicle request transfers
        vehicle_ids = list(system_state["vehicles"].keys())
        if len(vehicle_ids) >= 2:
            inter_results = self._run_ts_inter(
                vehicle_ids, system_state, feasibility_checker, weights,
                existing_improvements=new_plans,
            )
            new_plans.update(inter_results)

        return new_plans

    # ==================================================================
    # Helper: movable requests
    # ==================================================================

    @staticmethod
    def _requests_in_plan(plan: list, n_committed: int = 0) -> list:
        """Return req_ids with BOTH PU and DO in the movable portion."""
        movable = plan[n_committed:]
        pu_ids  = {s.req_id for s in movable if s.kind == "PU"}
        do_ids  = {s.req_id for s in movable if s.kind == "DO"}
        return list(pu_ids & do_ids)

    # ==================================================================
    # Neighbourhood generation
    # ==================================================================

    def _generate_neighbours(
        self,
        plan:        list,
        n_committed: int,
    ) -> list[tuple]:
        """
        Generate the relocate neighbourhood for the current plan.

        For each request with both PU and DO in the movable portion,
        enumerate all feasible (i, j) insertion positions with i < j
        and (i, j) different from the current positions.

        Returns a list of (move, candidate_plan) pairs where
        move = (req_id, new_pu_pos, new_do_pos).

        The neighbourhood is capped at self.max_neighbours to bound
        the per-iteration cost.  When the full neighbourhood exceeds
        the cap, we sample uniformly without replacement.  This keeps
        the per-iteration cost O(max_neighbours) regardless of n.
        """
        candidates_list = self._requests_in_plan(plan, n_committed)
        if not candidates_list:
            return []

        committed = plan[:n_committed]
        movable   = plan[n_committed:]
        n         = len(movable)

        # Collect all valid (req, i, j) moves
        all_moves = []

        for req in candidates_list:
            # Find current positions in movable slice
            cur_pu = cur_do = None
            for idx, s in enumerate(movable):
                if s.req_id == req:
                    if s.kind == "PU":
                        cur_pu = idx
                    elif s.kind == "DO":
                        cur_do = idx

            if cur_pu is None or cur_do is None:
                continue

            pu_stop = movable[cur_pu]
            do_stop = movable[cur_do]

            # Stripped movable without this request's stops
            stripped = [s for s in movable if s.req_id != req]
            n_stripped = len(stripped)  # always n - 2

            for i in range(n_stripped + 1):
                for j in range(i + 1, n_stripped + 2):
                    # Skip if this reproduces the current ordering
                    # (inserting back at the same effective positions)
                    if i == cur_pu and j - 1 == cur_do - 1:
                        continue

                    new_movable = list(stripped)
                    new_movable.insert(i, pu_stop)
                    new_movable.insert(j, do_stop)
                    all_moves.append((req, i, j, committed + new_movable))

        if not all_moves:
            return []

        # Cap the neighbourhood
        if len(all_moves) > self.max_neighbours:
            all_moves = self.rng.sample(all_moves, self.max_neighbours)

        return [(req, i, j, plan_) for req, i, j, plan_ in all_moves]

    def _generate_swap_neighbours(
        self,
        plan:        list,
        n_committed: int,
    ) -> list[tuple]:
        """
        Generate the pair-swap neighbourhood.

        For each pair of requests (A, B) in the movable portion,
        swap PU_A ↔ PU_B and DO_A ↔ DO_B simultaneously.

        Move attribute: ("swap", req_a, req_b) — tabu-listed as a
        3-tuple to distinguish from relocate moves.

        Returns list of (move, candidate_plan).
        """
        candidates_list = self._requests_in_plan(plan, n_committed)
        if len(candidates_list) < 2:
            return []

        neighbours = []
        pairs = [(candidates_list[i], candidates_list[j])
                 for i in range(len(candidates_list))
                 for j in range(i + 1, len(candidates_list))]

        # Cap swap neighbourhood too
        if len(pairs) > self.max_neighbours // 2:
            pairs = self.rng.sample(pairs, self.max_neighbours // 2)

        for req_a, req_b in pairs:
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
                continue

            new_plan = list(plan)
            new_plan[idx_pu_a], new_plan[idx_pu_b] = plan[idx_pu_b], plan[idx_pu_a]
            new_plan[idx_do_a], new_plan[idx_do_b] = plan[idx_do_b], plan[idx_do_a]

            move = ("swap", req_a, req_b)
            neighbours.append((move, new_plan))

        return neighbours

    # ==================================================================
    # Intra-vehicle TS loop
    # ==================================================================

    def _run_ts_intra(
        self,
        vehicle_id:          str,
        initial_plan:        list,
        n_committed:         int,
        system_state:        dict,
        feasibility_checker: Callable,
        weights:             tuple,
    ) -> Optional[list]:
        """
        Run TS for one vehicle (intra-vehicle moves only).

        Returns the best feasible plan found, or None if no improvement.
        Each call gets its own fresh time budget and empty tabu list.

        Algorithm
        ---------
        At each iteration:
          1. Generate the combined relocate + swap neighbourhood.
          2. Evaluate all feasible, non-tabu neighbours (aspiration
             overrides tabu if the move beats global best).
          3. Accept the best non-tabu (or aspirated) neighbour even
             if it worsens the current solution — this is the key
             distinction from SA (no temperature) and greedy (no
             worsening acceptance).
          4. Add the accepted move to the tabu list.
          5. Evict expired tabu entries.
          6. If no non-tabu feasible neighbour exists, accept the
             least-tabu move (tabu list restart).
          7. Diversification: if `patience` iterations pass without
             a global best improvement, restart from global best.
        """
        vehicle_state = system_state["vehicles"][vehicle_id]
        start_time    = time.time()

        current      = list(initial_plan)
        best         = list(initial_plan)
        initial_cost = evaluate_plan(initial_plan, vehicle_state, system_state, weights)
        current_cost = initial_cost
        best_cost    = initial_cost

        # tabu_list: move -> iteration number at which it was made
        tabu_list:   dict = {}
        iteration    = 0
        no_improve   = 0

        for iteration in range(self.iterations):
            if time.time() - start_time > self.decision_time_limit:
                break

            # Evict expired tabu entries
            expired = [m for m, made_at in tabu_list.items()
                       if iteration - made_at >= self.tabu_tenure]
            for m in expired:
                del tabu_list[m]

            # --- Build combined neighbourhood ---
            relocate_neighbours = self._generate_neighbours(current, n_committed)
            swap_neighbours     = self._generate_swap_neighbours(current, n_committed)

            # Combine: relocate moves get move key (req_id, pu_pos, do_pos)
            all_neighbours = []
            for (req, i, j, plan_) in relocate_neighbours:
                move = (req, i, j)
                all_neighbours.append((move, plan_))
            for (move, plan_) in swap_neighbours:
                all_neighbours.append((move, plan_))

            if not all_neighbours:
                break

            # --- Evaluate each neighbour ---
            best_move      = None
            best_move_plan = None
            best_move_cost = float("inf")
            fallback_move  = None   # least-tabu feasible move (for tabu restart)
            fallback_plan  = None
            fallback_cost  = float("inf")
            fallback_age   = -1

            for move, candidate in all_neighbours:
                if not feasibility_checker(candidate, vehicle_state, system_state):
                    continue

                cost = evaluate_plan(candidate, vehicle_state, system_state, weights)

                is_tabu    = move in tabu_list
                aspirated  = cost < best_cost  # aspiration criterion

                if not is_tabu or aspirated:
                    if cost < best_move_cost:
                        best_move      = move
                        best_move_plan = candidate
                        best_move_cost = cost
                else:
                    # Tabu but not aspirated — track as fallback
                    age = iteration - tabu_list[move]
                    if age > fallback_age or (age == fallback_age and cost < fallback_cost):
                        fallback_move  = move
                        fallback_plan  = candidate
                        fallback_cost  = cost
                        fallback_age   = age

            # --- Select the move to make ---
            if best_move is not None:
                chosen_move = best_move
                chosen_plan = best_move_plan
                chosen_cost = best_move_cost
            elif fallback_move is not None:
                # All non-tabu neighbours were infeasible — use least-tabu
                chosen_move = fallback_move
                chosen_plan = fallback_plan
                chosen_cost = fallback_cost
            else:
                # No feasible neighbour at all — terminate
                break

            # --- Update current solution ---
            current      = chosen_plan
            current_cost = chosen_cost
            tabu_list[chosen_move] = iteration

            # --- Update global best ---
            if current_cost < best_cost:
                best      = list(current)
                best_cost = current_cost
                no_improve = 0
            else:
                no_improve += 1

            # --- Diversification: restart from best if stuck ---
            if no_improve >= self.patience:
                current      = list(best)
                current_cost = best_cost
                no_improve   = 0
                # Partially clear tabu list on restart to allow
                # re-exploring recently forbidden moves from a new angle
                tabu_list = {m: v for m, v in tabu_list.items()
                             if iteration - v < self.tabu_tenure // 2}

        if best_cost < initial_cost:
            return best
        return None

    # ==================================================================
    # Inter-vehicle request transfer (Phase 2)
    # ==================================================================

    def _run_ts_inter(
        self,
        vehicle_ids:           list,
        system_state:          dict,
        feasibility_checker:   Callable,
        weights:               tuple,
        existing_improvements: dict,
    ) -> dict:
        """
        Try moving requests between vehicles using greedy acceptance.

        Uses random sampling (same as SA/GA phase 2) for fair comparison.
        A move is accepted only if it strictly improves the combined cost
        of both affected vehicles.

        Returns dict of vehicle_id -> improved plan.
        """
        start_time   = time.time()
        improvements = {}
        n_attempts   = min(self.iterations * 5, 2000)

        # Build working copies (incorporate any intra-vehicle improvements)
        working = {}
        for vid in vehicle_ids:
            if vid in existing_improvements:
                working[vid] = list(existing_improvements[vid])
            else:
                working[vid] = list(system_state["vehicles"][vid]["plan"])

        for _ in range(n_attempts):
            if time.time() - start_time > self.decision_time_limit:
                break

            # Pick source vehicle with movable requests
            src_vid  = self.rng.choice(vehicle_ids)
            src_info = system_state["vehicles"][src_vid]
            n_committed_src = src_info.get("n_committed", 0)
            movable  = self._requests_in_plan(working[src_vid], n_committed_src)

            if not movable:
                continue

            # Pick a different destination vehicle
            dst_vid = self.rng.choice(vehicle_ids)
            if dst_vid == src_vid:
                continue

            dst_info = system_state["vehicles"][dst_vid]
            n_committed_dst = dst_info.get("n_committed", 0)

            req = self.rng.choice(movable)

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

            # Insert into destination at best feasible position (greedy)
            dst_committed = working[dst_vid][:n_committed_dst]
            dst_movable   = working[dst_vid][n_committed_dst:]
            n_dst         = len(dst_movable)

            best_dst_cost = float("inf")
            best_dst_plan = None

            for i in range(n_dst + 1):
                for j in range(i + 1, n_dst + 2):
                    candidate = list(dst_movable)
                    candidate.insert(i, pu_stop)
                    candidate.insert(j, do_stop)
                    full_dst = dst_committed + candidate

                    if not feasibility_checker(full_dst, dst_info, system_state):
                        continue

                    cost = evaluate_plan(full_dst, dst_info, system_state, weights)
                    if cost < best_dst_cost:
                        best_dst_cost = cost
                        best_dst_plan = full_dst

            if best_dst_plan is None:
                continue

            if not feasibility_checker(new_src, src_info, system_state):
                continue

            # Accept only if combined cost strictly improves
            old_cost = (
                evaluate_plan(working[src_vid], src_info, system_state, weights)
                + evaluate_plan(working[dst_vid], dst_info, system_state, weights)
            )
            new_cost = (
                evaluate_plan(new_src,        src_info, system_state, weights)
                + evaluate_plan(best_dst_plan, dst_info, system_state, weights)
            )

            if new_cost < old_cost:
                working[src_vid] = new_src
                working[dst_vid] = best_dst_plan
                improvements[src_vid] = new_src
                improvements[dst_vid] = best_dst_plan

        # Verify total improvement across all touched vehicles
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