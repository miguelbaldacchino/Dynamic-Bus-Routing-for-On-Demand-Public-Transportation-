# simulation.py
# SimPy simulation: vehicle processes, request generator, and main().
#
# This is the only file that imports simpy.  All routing logic is
# delegated to dispatcher.py; all data structures come from models.py.

import simpy
import random

from models import Stop, Request, Vehicle
from travel import DEFAULT_COORDS, make_travel_fn
from feasibility import check_feasibility
from dispatcher import greedy_insert, sa_improve, print_plans


# ---------------------------------------------------------------------------
# Vehicle process
# ---------------------------------------------------------------------------

def vehicle_process(env: simpy.Environment, vehicle: Vehicle, travel_fn) -> None:
    """
    SimPy process: repeatedly pops the next stop from the vehicle plan,
    travels to it, waits if early for a pickup, then serves the stop.
    """
    print(f"[t={env.now:.1f}] {vehicle.id} started at depot (node {vehicle.location})")

    while True:
        if not vehicle.plan:
            yield env.timeout(1)
            continue

        stop = vehicle.plan.pop(0)
        print(f"[t={env.now:.1f}] {vehicle.id} → {stop.kind} {stop.req_id} at node {stop.node}")

        travel = travel_fn(vehicle.location, stop.node, env.now)
        yield env.timeout(travel)
        vehicle.location = stop.node

        # Wait at pickup if vehicle arrives before time window opens
        if stop.kind == "PU" and stop.earliest and env.now < stop.earliest:
            yield env.timeout(stop.earliest - env.now)

        yield env.timeout(stop.service)
        print(f"[t={env.now:.1f}] {vehicle.id} served {stop.kind} {stop.req_id}")


# ---------------------------------------------------------------------------
# Request generator process
# ---------------------------------------------------------------------------

def request_generator(
    env: simpy.Environment,
    vehicles: dict[str, Vehicle],
    system_state: dict,
    weights: tuple[float, float, float],
    inter_arrival: float,
    n_requests: int,
    direct_times: dict,
    travel_fn,
) -> None:
    """
    SimPy process: generates *n_requests* requests spaced *inter_arrival*
    apart, inserts each greedily, then runs SA improvement.
    """
    for i in range(1, n_requests + 1):
        yield env.timeout(inter_arrival)

        pu = random.randint(1, 6)
        do = random.randint(1, 6)
        while do == pu:
            do = random.randint(1, 6)

        req = Request(
            id=f"R{i}",
            pickup_node=pu,
            dropoff_node=do,
            earliest=env.now,
            request_time=env.now,
        )

        # Store direct travel time for feasibility checks
        direct_times[req.id] = travel_fn(pu, do, env.now)

        print(f"\n[t={env.now:.1f}] New request {req.id}  ({pu} → {do})")

        inserted = greedy_insert(req, vehicles, system_state, env.now, weights)

        if inserted:
            sa_improve(vehicles, system_state, env.now, weights)

        print_plans(vehicles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    env    = simpy.Environment()
    coords = DEFAULT_COORDS

    travel_fn    = make_travel_fn(coords)
    direct_times: dict = {}

    # Build fleet
    vehicles = {
        "Bus-1": Vehicle(id="Bus-1", capacity=12, location=0),
        "Bus-2": Vehicle(id="Bus-2", capacity=12, location=0),
        "Bus-3": Vehicle(id="Bus-3", capacity=12, location=0),
        "Bus-4": Vehicle(id="Bus-4", capacity=12, location=0),
        "Bus-5": Vehicle(id="Bus-5", capacity=12, location=0),
    }

    # System state dict — passed to feasibility checker and dispatcher
    system_state = {
        "travel_time":  travel_fn,
        "ride_factor":  2.3,
        "direct_times": direct_times,
        "coords":       coords,
    }

    weights = (1.0, 2.0, 1.0)  # (alpha, beta, gamma)

    # Spawn SimPy processes
    for vehicle in vehicles.values():
        env.process(vehicle_process(env, vehicle, travel_fn))

    env.process(
        request_generator(
            env,
            vehicles,
            system_state,
            weights,
            inter_arrival=1.5,
            n_requests=100,
            direct_times=direct_times,
            travel_fn=travel_fn,
        )
    )

    env.run(until=400)


if __name__ == "__main__":
    main()