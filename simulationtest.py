import simpy
import math
import random
from dataclasses import dataclass
from copy import deepcopy
from sa import SAPolicy


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Stop:
    node: int
    kind: str # PU or DO
    req_id: str
    earliest: float
    service: float # service time at this stop
    request_time: float # original arrival time of the request 


# ============================================================
# TRAVEL TIME
# ============================================================

# euclidean distance as travel time (for simplicity)
def travel_time(a, b, coords):
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.hypot(ax - bx, ay - by)


# ============================================================
# FEASIBILITY CHECKER 
# ============================================================

def feasibility_checker(plan, vehicle_state, system_state):

    capacity = vehicle_state["capacity"]
    ride_factor = system_state["ride_factor"]

    onboard = 0
    pickup_times = {}

    current_node = vehicle_state["location"]
    current_time = vehicle_state["time"]

    for stop in plan:

        t = system_state["travel_time"](current_node, stop.node, current_time)
        current_time += t

        if stop.kind == "PU":
            # pickup time window check
            if stop.earliest and current_time < stop.earliest:
                current_time = stop.earliest

            onboard += 1
            pickup_times[stop.req_id] = current_time
            # capacity check
            if onboard > capacity:
                return False

        elif stop.kind == "DO":
            # precedence check (pickup must be before dropoff)
            if stop.req_id not in pickup_times:
                return False

            ride_time = current_time - pickup_times[stop.req_id]
            direct = system_state["direct_times"][stop.req_id]
            # max ride time check (e.g. no more than 2x direct time)
            if ride_time > ride_factor * direct:
                return False

            onboard -= 1

        current_time += stop.service
        current_node = stop.node

    return True


# ============================================================
# OBJECTIVE
# ============================================================

def evaluate_plan(plan, vehicle_state, system_state, weights):
    # α * distance + β * wait_time + γ * ride_time
    alpha, beta, gamma = weights

    current_node = vehicle_state["location"]
    current_time = vehicle_state["time"]

    total_distance = 0
    total_wait = 0
    total_ride = 0

    pickup_times = {}

    for stop in plan:

        t = system_state["travel_time"](current_node, stop.node, current_time)
        total_distance += t
        current_time += t

        if stop.kind == "PU":

            if stop.earliest and current_time < stop.earliest:
                current_time = stop.earliest

            wait = current_time - stop.request_time
            total_wait += wait
            pickup_times[stop.req_id] = current_time

        elif stop.kind == "DO":
            ride_time = current_time - pickup_times[stop.req_id]
            total_ride += ride_time

        current_time += stop.service
        current_node = stop.node

    return alpha * total_distance + beta * total_wait + gamma * total_ride


# ============================================================
# PRINT
# ============================================================

def print_plan(system):
    for vid, vehicle in system["vehicles"].items():
        plan_str = [(s.kind, s.req_id) for s in vehicle["plan"]]
        print(f"   {vid} plan: {plan_str}")


# ============================================================
# GREEDY INSERTION 
# ============================================================

def greedy_insert_request(request, system, env, weights):

    print(f"Attempting insertion for {request['id']}")

    best_choice = None
    best_cost = float("inf")

    for vid, vehicle in system["vehicles"].items():

        base_plan = vehicle["plan"]

        # currently O(n²)
        for i in range(len(base_plan) + 1):
            for j in range(i + 1, len(base_plan) + 2):

                candidate = deepcopy(base_plan)

                pu = Stop(request["pickup_node"], "PU",
                          request["id"], request["earliest"],
                          1.0, request["request_time"])

                do = Stop(request["dropoff_node"], "DO",
                          request["id"], None,
                          1.0, request["request_time"])

                candidate.insert(i, pu)
                candidate.insert(j, do)

                vehicle_state = {
                    "capacity": vehicle["capacity"],
                    "location": vehicle["location"],
                    "time": env.now
                }

                system_state = system["system_state"]

                if not feasibility_checker(candidate, vehicle_state, system_state):
                    continue

                cost = evaluate_plan(candidate, vehicle_state, system_state, weights)

                print(f"  Feasible in {vid} at positions ({i},{j}) cost={cost:.2f}")

                if cost < best_cost:                                                                                                                                                                                                                                                                                                                                
                    best_cost = cost
                    best_choice = (vid, candidate)

    if best_choice:
        vid, plan = best_choice
        system["vehicles"][vid]["plan"] = plan
        print(f"Inserted {request['id']} into {vid} with cost={best_cost:.2f}")
    else:
        print(f"Rejected {request['id']}")

    print_plan(system)


# ============================================================
# SA IMPROVEMENT 
# ============================================================

def improve_with_sa(system, env, weights):

    policy = SAPolicy(
        initial_temp=10000,
        cooling_rate=0.999,
        iterations=20000,
        decision_time_limit=0.9
    )

    system_state = {
        "vehicles": {},
        "coords": system["coords"],
        "travel_time": lambda a, b, t: travel_time(a, b, system["coords"]),
        "ride_factor": system["ride_factor"],
        "direct_times": system["direct_times"]
    }

    for vid, vehicle in system["vehicles"].items():
        system_state["vehicles"][vid] = {
            "plan": vehicle["plan"],
            "capacity": vehicle["capacity"],
            "location": vehicle["location"],
            "time": env.now
        }

    print("Running SA improvement...")

    changes = policy.propose(system_state, feasibility_checker)

    for vid, new_plan in changes.items():

        vehicle = system["vehicles"][vid]

        vehicle_state = {
            "capacity": vehicle["capacity"],
            "location": vehicle["location"],
            "time": env.now
        }

        system_state_eval = system["system_state"]

        before = evaluate_plan(vehicle["plan"],
                            vehicle_state,
                            system_state_eval,
                            weights)

        after = evaluate_plan(new_plan,
                            vehicle_state,
                            system_state_eval,
                            weights)

        if after < before:
            print(f"SA improved {vid}: {before:.2f} → {after:.2f}")
        else:
            print(f"SA no improvement for {vid}: {before:.2f} → {after:.2f}")

        system["vehicles"][vid]["plan"] = new_plan
        
        print_plan(system)


# ============================================================
# VEHICLE PROCESS
# ============================================================

def vehicle_process(env, vid, system):

    vehicle = system["vehicles"][vid]
    print(f"[t={env.now:.1f}] {vid} started at depot")

    while True:

        if not vehicle["plan"]:
            yield env.timeout(1)
            continue

        stop = vehicle["plan"].pop(0)

        print(f"[t={env.now:.1f}] {vid} travelling to {stop.kind}-{stop.req_id}")

        t = travel_time(vehicle["location"], stop.node, system["coords"])
        yield env.timeout(t)

        vehicle["location"] = stop.node

        if stop.kind == "PU" and stop.earliest and env.now < stop.earliest:
            yield env.timeout(stop.earliest - env.now)

        yield env.timeout(stop.service)

        print(f"[t={env.now:.1f}] {vid} served {stop.kind} {stop.req_id}")


# ============================================================
# REQUEST GENERATOR 
# ============================================================

def request_generator(env, system, weights, timeout):

    for i in range(1, 41):

        yield env.timeout(timeout)

        pu = random.randint(1, 6)
        do = random.randint(1, 6)
        while do == pu:
            do = random.randint(1, 6)

        req = {
            "id": f"R{i}",
            "pickup_node": pu,
            "dropoff_node": do,
            "earliest": env.now,
            "request_time": env.now
        }

        print(f"\n[t={env.now:.1f}] Request {req['id']} ({pu}->{do})")

        direct = travel_time(pu, do, system["coords"])
        system["direct_times"][req["id"]] = direct

        greedy_insert_request(req, system, env, weights)
        improve_with_sa(system, env, weights)


# ============================================================
# MAIN
# ============================================================

def main():

    env = simpy.Environment()

    coords = {
        0: (0, 0),
        1: (1, 2),
        2: (4, 1),
        3: (2, 6),
        4: (7, 5),
        5: (8, 1),
        6: (5, 8),
    }

    system = {
        "coords": coords,
        "ride_factor": 2.3,
        "direct_times": {},
        "vehicles": {
            "Bus-1": {"capacity": 14, "location": 0, "plan": []},
            "Bus-2": {"capacity": 14, "location": 0, "plan": []}
        }
    }

    system["system_state"] = {
        "travel_time": lambda a, b, t: travel_time(a, b, coords),
        "ride_factor": system["ride_factor"],
        "direct_times": system["direct_times"]
    }

    weights = (1.0, 2.0, 1.0)

    env.process(vehicle_process(env, "Bus-1", system))
    env.process(vehicle_process(env, "Bus-2", system))
    env.process(request_generator(env, system, weights, 3))

    env.run(until=400)


if __name__ == "__main__":
    main()