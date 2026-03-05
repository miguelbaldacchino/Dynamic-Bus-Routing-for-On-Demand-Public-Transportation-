# sa.py

import math
import random
import time
from copy import deepcopy


class SAPolicy:


    def __init__(self,
                 initial_temp=1000.0,
                 cooling_rate=0.995,
                 iterations=2000,
                 decision_time_limit=0.05):

        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.decision_time_limit = decision_time_limit

    # ==========================================================
    # PUBLIC INTERFACE 
    # ==========================================================

    def propose(self, system_state, feasibility_checker):

        start_time = time.time()
        new_plans = {}

        for vehicle_id, vehicle in system_state["vehicles"].items():

            current_plan = vehicle["plan"]

            if len(current_plan) <= 2:
                continue  # nothing to optimise

            improved_plan = self._run_sa(
                vehicle_id,
                current_plan,
                system_state,
                feasibility_checker,
                start_time
            )

            if improved_plan is not None:
                new_plans[vehicle_id] = improved_plan

        return new_plans

    # ==========================================================
    # INTERNAL SA LOGIC
    # ==========================================================

    def _route_cost(self, plan, vehicle_state, coords, travel_time):


        total = 0.0
        current_node = vehicle_state["location"]
        current_time = vehicle_state["time"]

        for stop in plan:
            travel = travel_time(current_node, stop.node, current_time)
            total += travel
            current_time += travel + stop.service
            current_node = stop.node

        return total

    def _neighbour(self, plan):
        new_plan = deepcopy(plan)
        i, j = random.sample(range(len(new_plan)), 2)
        new_plan[i], new_plan[j] = new_plan[j], new_plan[i]
        return new_plan

    def _run_sa(self,
                vehicle_id,
                initial_plan,
                system_state,
                feasibility_checker,
                start_time):
        print(f"SA evaluating vehicle {vehicle_id} with {len(initial_plan)} stops")
        vehicle_state = system_state["vehicles"][vehicle_id]
        coords = system_state["coords"]
        travel_time = system_state["travel_time"]

        current = deepcopy(initial_plan)
        best = deepcopy(initial_plan)

        current_cost = self._route_cost(current, vehicle_state, coords, travel_time)
        best_cost = current_cost

        temp = self.initial_temp

        for _ in range(self.iterations):

            # Enforce strict latency cap
            if time.time() - start_time > self.decision_time_limit:
                break

            candidate = self._neighbour(current)

            if not feasibility_checker(candidate, vehicle_state, system_state):
                continue

            candidate_cost = self._route_cost(candidate, vehicle_state, coords, travel_time)
            delta = candidate_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = candidate
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best = deepcopy(candidate)
                    best_cost = current_cost

            temp *= self.cooling_rate
            if temp < 1e-4:
                break

        if best_cost < self._route_cost(initial_plan, vehicle_state, coords, travel_time):
            return best
        return None