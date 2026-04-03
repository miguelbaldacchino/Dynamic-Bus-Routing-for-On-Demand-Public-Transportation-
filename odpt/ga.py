# ga.py
# Genetic Algorithm improvement policy for dynamic DARP.
#
# Drop-in replacement / companion for SAPolicy.  Same interface:
#   GAPolicy.propose(system_state, feasibility_checker, weights)
#   -> dict { vehicle_id: improved_plan }
#
# Design decisions (thesis-relevant)
# -----------------------------------
# 1. Population = permutations of MOVABLE stops within a vehicle's plan.
#    Committed/in-transit stops are frozen as a prefix — never touched.
#
# 2. Representation: each individual is a list of stops (the movable
#    portion of a vehicle's plan).  This is a direct/order-based
#    encoding, not a binary one, because DARP solutions are ordered
#    sequences with hard precedence constraints (PU before DO).
#
# 3. Crossover: Order Crossover (OX) adapted for paired PU/DO stops.
#    Standard OX would break precedence, so we repair after crossover
#    by scanning for any DO that appears before its PU and swapping
#    them.  This is the standard DARP-GA repair approach (Jørgensen
#    et al. 2007).
#
# 4. Mutation operators (same as SA for comparability):
#    - pair_relocate: remove one request's PU+DO and re-insert at
#      random positions i<j.
#    - pair_swap: swap the positions of two requests' PU+DO pairs.
#
# 5. Selection: tournament selection (size 3).  Elitism preserves
#    the best individual across generations.
#
# 6. Inter-vehicle moves: after intra-vehicle GA for each vehicle,
#    a quick inter-vehicle phase tries moving requests between
#    vehicles (same as SA phase 2), using GA-style selection.
#
# 7. Time budget: same per-vehicle wall-clock limit as SA, ensuring
#    fair comparison under identical computational budgets.
#
# All constraint checking uses the shared feasibility checker.
# No routing logic lives here — only plan permutation and selection.

import random
import time
from copy import deepcopy
from typing import Callable, Optional

from feasibility import evaluate_plan


class GAPolicy:
    """
    Genetic Algorithm improvement policy.

    Parameters
    ----------
    population_size : int
        Number of individuals per generation.
    generations : int
        Maximum generations per vehicle (may terminate early on time).
    crossover_rate : float
        Probability of applying crossover to produce offspring.
    mutation_rate : float
        Probability of mutating each offspring.
    tournament_size : int
        Number of candidates in tournament selection.
    elite_count : int
        Number of best individuals carried forward unchanged.
    decision_time_limit : float
        Wall-clock seconds allowed per vehicle (for fair comparison
        with SA, which also gets a per-vehicle time budget).
    """

    def __init__(
        self,
        population_size:     int   = 30,
        generations:         int   = 200,
        crossover_rate:      float = 0.85,
        mutation_rate:       float = 0.40,
        tournament_size:     int   = 3,
        elite_count:         int   = 2,
        decision_time_limit: float = 0.3,
    ):
        self.population_size     = population_size
        self.generations         = generations
        self.crossover_rate      = crossover_rate
        self.mutation_rate       = mutation_rate
        self.tournament_size     = tournament_size
        self.elite_count         = elite_count
        self.decision_time_limit = decision_time_limit

    # ==================================================================
    # Public interface (matches SAPolicy.propose)
    # ==================================================================

    def propose(
        self,
        system_state: dict,
        feasibility_checker: Callable,
        weights: tuple = (1.0, 2.0, 2.5),
    ) -> dict:
        """
        Propose improved plans for vehicles using a Genetic Algorithm.

        Two-phase approach (mirrors SA):
          Phase 1: intra-vehicle GA for each vehicle with >=4 movable stops.
          Phase 2: inter-vehicle request transfer attempts.

        Parameters
        ----------
        system_state : dict
            Frozen snapshot with "vehicles" sub-dict.
            Each vehicle has: plan, n_committed, capacity, location, time,
            onboard_count, onboard_pickup_times.
        feasibility_checker : callable
            check_feasibility(plan, vehicle_state, system_state) -> bool
        weights : tuple
            (alpha, beta, gamma) for evaluate_plan.

        Returns
        -------
        dict  { vehicle_id: improved_plan }  (only vehicles that improved)
        """
        new_plans = {}

        # Phase 1 — intra-vehicle GA
        for vehicle_id, vehicle in system_state["vehicles"].items():
            current_plan = vehicle["plan"]
            n_committed  = vehicle.get("n_committed", 0)

            # Need at least 2 movable request pairs (4 stops)
            if len(current_plan) - n_committed < 4:
                continue

            improved = self._run_ga_intra(
                vehicle_id, current_plan, n_committed,
                system_state, feasibility_checker, weights,
            )
            if improved is not None:
                new_plans[vehicle_id] = improved

        # Phase 2 — inter-vehicle request transfers
        vehicle_ids = list(system_state["vehicles"].keys())
        if len(vehicle_ids) >= 2:
            inter_results = self._run_inter_vehicle(
                vehicle_ids, system_state, feasibility_checker, weights,
                existing_improvements=new_plans,
            )
            new_plans.update(inter_results)

        return new_plans

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _requests_in_plan(plan: list, n_committed: int = 0) -> list:
        """
        Return req_ids that have BOTH PU and DO in the movable portion.
        """
        movable = plan[n_committed:]
        pu_ids = {s.req_id for s in movable if s.kind == "PU"}
        do_ids = {s.req_id for s in movable if s.kind == "DO"}
        return list(pu_ids & do_ids)

    @staticmethod
    def _repair_precedence(stops: list) -> list:
        """
        Repair precedence violations in a stop sequence.

        Scans left-to-right.  If a DO appears before its PU has been
        seen, swap the two stops in-place.  This is O(n) and preserves
        as much of the ordering as possible.

        This is the standard repair operator for DARP GAs (Jørgensen
        et al. 2007): apply crossover freely, then repair.
        """
        seen_pu = set()
        result = list(stops)

        # First pass: find PU positions for quick lookup
        pu_positions = {}
        for i, s in enumerate(result):
            if s.kind == "PU":
                pu_positions[s.req_id] = i

        # Second pass: fix any DO that appears before its PU
        for i in range(len(result)):
            s = result[i]
            if s.kind == "DO" and s.req_id not in seen_pu:
                # PU hasn't been seen yet — find it and move it before this DO
                if s.req_id in pu_positions:
                    pu_idx = pu_positions[s.req_id]
                    if pu_idx > i:
                        # Remove PU from its current position
                        pu_stop = result.pop(pu_idx)
                        # Insert PU just before this DO
                        result.insert(i, pu_stop)
                        # Update tracking
                        seen_pu.add(s.req_id)
                        # Rebuild pu_positions after modification
                        pu_positions = {
                            st.req_id: idx
                            for idx, st in enumerate(result)
                            if st.kind == "PU"
                        }
                        continue
            if s.kind == "PU":
                seen_pu.add(s.req_id)

        return result

    # ==================================================================
    # Crossover: Order Crossover (OX) with DARP repair
    # ==================================================================

    def _order_crossover(self, parent_a: list, parent_b: list) -> list:
        """
        Order Crossover (OX) adapted for DARP stop sequences.

        1. Select a random substring from parent_a.
        2. Fill remaining positions with stops from parent_b in order,
           skipping stops already placed.
        3. Repair any precedence violations (DO before PU).

        Returns one offspring.
        """
        n = len(parent_a)
        if n < 2:
            return list(parent_a)

        # Select crossover segment [cx1, cx2)
        cx1 = random.randint(0, n - 2)
        cx2 = random.randint(cx1 + 1, n)

        # Child starts with the segment from parent_a
        child = [None] * n
        segment = parent_a[cx1:cx2]
        child[cx1:cx2] = segment

        # IDs already placed (by req_id + kind to handle PU/DO separately)
        placed = {(s.req_id, s.kind) for s in segment}

        # Fill remaining positions from parent_b in order
        b_filtered = [
            s for s in parent_b
            if (s.req_id, s.kind) not in placed
        ]

        fill_idx = 0
        for i in range(n):
            if child[i] is None:
                if fill_idx < len(b_filtered):
                    child[i] = b_filtered[fill_idx]
                    fill_idx += 1

        # Safety: if any None remains (shouldn't happen with valid inputs),
        # fill from parent_a
        for i in range(n):
            if child[i] is None:
                child[i] = parent_a[i]

        # Repair precedence violations
        child = self._repair_precedence(child)
        return child

    # ==================================================================
    # Mutation operators (same as SA for comparability)
    # ==================================================================

    @staticmethod
    def _mutate_pair_relocate(stops: list) -> Optional[list]:
        """
        Remove one request's PU+DO pair and re-insert at random
        positions i<j.  Same operator as SA._pair_relocate.
        """
        # Find requests with both PU and DO
        pu_ids = {s.req_id for s in stops if s.kind == "PU"}
        do_ids = {s.req_id for s in stops if s.kind == "DO"}
        candidates = list(pu_ids & do_ids)

        if not candidates:
            return None

        req = random.choice(candidates)
        pu_stop = do_stop = None
        for s in stops:
            if s.req_id == req and s.kind == "PU":
                pu_stop = s
            elif s.req_id == req and s.kind == "DO":
                do_stop = s

        if pu_stop is None or do_stop is None:
            return None

        stripped = [s for s in stops if s.req_id != req]
        n = len(stripped)

        i = random.randint(0, n)
        j = random.randint(i + 1, n + 1)

        result = list(stripped)
        result.insert(i, pu_stop)
        result.insert(j, do_stop)
        return result

    @staticmethod
    def _mutate_pair_swap(stops: list) -> Optional[list]:
        """
        Swap the positions of two requests' PU+DO pairs.
        Same operator as SA._pair_swap.
        """
        pu_ids = {s.req_id for s in stops if s.kind == "PU"}
        do_ids = {s.req_id for s in stops if s.kind == "DO"}
        candidates = list(pu_ids & do_ids)

        if len(candidates) < 2:
            return None

        req_a, req_b = random.sample(candidates, 2)

        idx_pu_a = idx_do_a = idx_pu_b = idx_do_b = None
        for i, s in enumerate(stops):
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

        result = list(stops)
        result[idx_pu_a], result[idx_pu_b] = stops[idx_pu_b], stops[idx_pu_a]
        result[idx_do_a], result[idx_do_b] = stops[idx_do_b], stops[idx_do_a]
        return result

    def _mutate(self, stops: list) -> Optional[list]:
        """Apply a random mutation operator."""
        candidates = self._requests_in_plan(stops, 0)
        if len(candidates) >= 2:
            op = random.choice([self._mutate_pair_relocate,
                                self._mutate_pair_swap])
        else:
            op = self._mutate_pair_relocate
        return op(stops)

    # ==================================================================
    # Selection
    # ==================================================================

    def _tournament_select(
        self, population: list[list], fitnesses: list[float],
    ) -> list:
        """
        Tournament selection: pick tournament_size individuals at random,
        return the one with the lowest cost (best fitness).
        """
        indices = random.sample(range(len(population)), self.tournament_size)
        best_idx = min(indices, key=lambda i: fitnesses[i])
        return list(population[best_idx])

    # ==================================================================
    # Intra-vehicle GA
    # ==================================================================

    def _run_ga_intra(
        self,
        vehicle_id:          str,
        initial_plan:        list,
        n_committed:         int,
        system_state:        dict,
        feasibility_checker: Callable,
        weights:             tuple,
    ) -> Optional[list]:
        """
        Run GA for one vehicle (intra-vehicle moves only).

        The population consists of permutations of the movable portion
        of the plan.  The committed prefix is frozen.

        Returns the best feasible plan found, or None if no improvement.
        """
        vehicle_state = system_state["vehicles"][vehicle_id]
        start_time    = time.time()

        committed = initial_plan[:n_committed]
        movable   = initial_plan[n_committed:]

        # --- Evaluate initial plan ---
        initial_cost = evaluate_plan(
            initial_plan, vehicle_state, system_state, weights,
        )

        # --- Seed population ---
        # Individual 0 = current plan (movable portion).
        # Others = random mutations of the current plan.
        population = [list(movable)]
        fitnesses  = [initial_cost]

        for _ in range(self.population_size - 1):
            candidate = self._generate_random_individual(
                movable, committed, vehicle_state,
                system_state, feasibility_checker, weights,
            )
            if candidate is not None:
                full = committed + candidate
                cost = evaluate_plan(full, vehicle_state, system_state, weights)
                population.append(candidate)
                fitnesses.append(cost)
            else:
                # Fall back to a copy of the original
                population.append(list(movable))
                fitnesses.append(initial_cost)

        best_individual = list(movable)
        best_cost       = initial_cost

        # --- Generational loop ---
        for gen in range(self.generations):
            if time.time() - start_time > self.decision_time_limit:
                break

            new_population = []
            new_fitnesses  = []

            # Elitism: carry forward the best individuals unchanged
            elite_indices = sorted(
                range(len(fitnesses)), key=lambda i: fitnesses[i],
            )[:self.elite_count]

            for ei in elite_indices:
                new_population.append(list(population[ei]))
                new_fitnesses.append(fitnesses[ei])

            # Fill the rest with offspring
            while len(new_population) < self.population_size:
                if time.time() - start_time > self.decision_time_limit:
                    break

                # Selection
                parent_a = self._tournament_select(population, fitnesses)
                parent_b = self._tournament_select(population, fitnesses)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._order_crossover(parent_a, parent_b)
                else:
                    child = list(parent_a)

                # Mutation
                if random.random() < self.mutation_rate:
                    mutated = self._mutate(child)
                    if mutated is not None:
                        child = mutated

                # Feasibility check
                full_plan = committed + child
                if not feasibility_checker(full_plan, vehicle_state, system_state):
                    # Infeasible offspring — try a repair by re-running
                    # precedence repair and re-checking
                    child = self._repair_precedence(child)
                    full_plan = committed + child
                    if not feasibility_checker(full_plan, vehicle_state, system_state):
                        # Still infeasible — discard, copy a parent instead
                        new_population.append(list(parent_a))
                        cost = evaluate_plan(
                            committed + parent_a, vehicle_state,
                            system_state, weights,
                        )
                        new_fitnesses.append(cost)
                        continue

                cost = evaluate_plan(full_plan, vehicle_state, system_state, weights)
                new_population.append(child)
                new_fitnesses.append(cost)

                # Track global best
                if cost < best_cost:
                    best_cost       = cost
                    best_individual = list(child)

            population = new_population
            fitnesses  = new_fitnesses

        # Return improvement only if strictly better
        if best_cost < initial_cost:
            return committed + best_individual
        return None

    def _generate_random_individual(
        self,
        movable:             list,
        committed:           list,
        vehicle_state:       dict,
        system_state:        dict,
        feasibility_checker: Callable,
        weights:             tuple,
        max_attempts:        int = 5,
    ) -> Optional[list]:
        """
        Generate a random feasible individual by applying random
        mutations to the current movable plan.  Tries up to
        max_attempts times.
        """
        for _ in range(max_attempts):
            # Apply 1-3 random mutations
            candidate = list(movable)
            for _ in range(random.randint(1, 3)):
                mutated = self._mutate(candidate)
                if mutated is not None:
                    candidate = mutated

            candidate = self._repair_precedence(candidate)
            full = committed + candidate
            if feasibility_checker(full, vehicle_state, system_state):
                return candidate

        return None

    # ==================================================================
    # Inter-vehicle request transfer (Phase 2)
    # ==================================================================

    def _run_inter_vehicle(
        self,
        vehicle_ids:           list,
        system_state:          dict,
        feasibility_checker:   Callable,
        weights:               tuple,
        existing_improvements: dict,
    ) -> dict:
        """
        Try moving requests between vehicles.
        Uses random sampling with greedy acceptance (keep if improves
        the combined cost of both vehicles).

        Same structure as SA's inter-vehicle phase for fair comparison.
        """
        start_time    = time.time()
        improvements  = {}
        max_attempts  = min(self.generations * self.population_size // 2, 2000)

        # Build working copies (incorporate any intra improvements)
        working = {}
        for vid in vehicle_ids:
            if vid in existing_improvements:
                working[vid] = list(existing_improvements[vid])
            else:
                working[vid] = list(system_state["vehicles"][vid]["plan"])

        for _ in range(max_attempts):
            if time.time() - start_time > self.decision_time_limit:
                break

            src_vid = random.choice(vehicle_ids)
            src_info = system_state["vehicles"][src_vid]
            n_committed_src = src_info.get("n_committed", 0)
            movable = self._requests_in_plan(working[src_vid], n_committed_src)

            if not movable:
                continue

            dst_vid = random.choice(vehicle_ids)
            if dst_vid == src_vid:
                continue

            dst_info = system_state["vehicles"][dst_vid]
            n_committed_dst = dst_info.get("n_committed", 0)

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

            if not feasibility_checker(new_src, src_info, system_state):
                continue
            if not feasibility_checker(new_dst, dst_info, system_state):
                continue

            # Greedy acceptance: keep if combined cost improves
            old_cost = (
                evaluate_plan(working[src_vid], src_info, system_state, weights)
                + evaluate_plan(working[dst_vid], dst_info, system_state, weights)
            )
            new_cost = (
                evaluate_plan(new_src, src_info, system_state, weights)
                + evaluate_plan(new_dst, dst_info, system_state, weights)
            )

            if new_cost < old_cost:
                working[src_vid] = new_src
                working[dst_vid] = new_dst
                improvements[src_vid] = new_src
                improvements[dst_vid] = new_dst

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
