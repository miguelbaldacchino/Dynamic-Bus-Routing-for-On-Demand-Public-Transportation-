# main.py
# SimPy simulation — full operating day (07:00 to 18:00).
# All times in MINUTES.  Simulation t=0 = 07:00.


import simpy
import random

from models import Request, Vehicle, RequestStatus
from travel import DEFAULT_COORDS, make_travel_fn
from dispatcher import greedy_insert, sa_improve, print_plans
from config import SimulationConfig, arrival_rate
from metrics import MetricsCollector


def sim_time_to_clock(t: float) -> str:
    total = int(7 * 60 + t)
    return f"{total // 60:02d}:{total % 60:02d}"


# ---------------------------------------------------------------------------
# Vehicle process
# ---------------------------------------------------------------------------

def vehicle_process(
    env:          simpy.Environment,
    vehicle:      Vehicle,
    travel_fn,
    system_state: dict,
    requests:     dict,
    metrics:      MetricsCollector,
    cfg:          SimulationConfig,
    verbose:      bool = False,
) -> None:

    if verbose:
        print(f"[{sim_time_to_clock(env.now)}] {vehicle.id} ready at depot")

    while True:
        if not vehicle.plan:
            yield env.timeout(1)
            continue

        stop = vehicle.plan.pop(0)

        if verbose:
            print(f"[{sim_time_to_clock(env.now)}] {vehicle.id} "
                  f"-> {stop.kind} {stop.req_id} node {stop.node}")

        travel = travel_fn(vehicle.location, stop.node, env.now)
        metrics.log_distance(travel)
        yield env.timeout(travel)
        vehicle.location = stop.node

        # Wait at pickup if vehicle arrives before earliest time
        if stop.kind == "PU" and stop.earliest and env.now < stop.earliest:
            yield env.timeout(stop.earliest - env.now)

        yield env.timeout(stop.service)

        # ----------------------------------------------------------------
        # Pickup: record actual time, check wait-time violation
        # ----------------------------------------------------------------
        if stop.kind == "PU":
            actual_wait = env.now - stop.request_time

            # Execution-time wait violation
            if stop.latest is not None and env.now > stop.latest:
                metrics.log_violation(
                    kind    = "wait",
                    req_id  = stop.req_id,
                    value   = actual_wait,
                    limit   = cfg.max_wait,
                    t       = env.now,
                )

            vehicle.onboard.add(stop.req_id)
            metrics.mark_pickup(stop.req_id, env.now)

            if stop.req_id in requests:
                req            = requests[stop.req_id]
                req.pickup_time = env.now
                req.status      = RequestStatus.ONBOARD

            if verbose:
                print(f"[{sim_time_to_clock(env.now)}] {vehicle.id} "
                      f"picked up {stop.req_id}  "
                      f"(waited {actual_wait:.1f} min)")

        # ----------------------------------------------------------------
        # Dropoff: record actual time, check ride-time violation
        # ----------------------------------------------------------------
        elif stop.kind == "DO":
            vehicle.onboard.discard(stop.req_id)
            metrics.mark_dropoff(stop.req_id, env.now)

            if stop.req_id in requests:
                req             = requests[stop.req_id]
                req.dropoff_time = env.now
                req.status       = RequestStatus.COMPLETED

                # Execution-time ride-time violation
                if req.pickup_time is not None and req.direct_time:
                    actual_ride = env.now - req.pickup_time
                    max_ride    = cfg.ride_factor * req.direct_time
                    if actual_ride > max_ride:
                        metrics.log_violation(
                            kind   = "ride",
                            req_id = stop.req_id,
                            value  = actual_ride,
                            limit  = max_ride,
                            t      = env.now,
                        )

            if verbose:
                print(f"[{sim_time_to_clock(env.now)}] {vehicle.id} "
                      f"dropped off {stop.req_id}")


# ---------------------------------------------------------------------------
# Request generator
# ---------------------------------------------------------------------------

def request_generator(
    env:          simpy.Environment,
    vehicles:     dict,
    system_state: dict,
    weights:      tuple,
    cfg:          SimulationConfig,
    direct_times: dict,
    travel_fn,
    requests:     dict,
    metrics:      MetricsCollector,
    verbose:      bool = False,
) -> None:

    for i in range(1, cfg.n_requests + 1):
        yield env.timeout(arrival_rate(env.now, cfg))

        pu = random.randint(1, cfg.n_nodes)
        do = random.randint(1, cfg.n_nodes)
        while do == pu:
            do = random.randint(1, cfg.n_nodes)

        req = Request(
            id           = f"R{i}",
            pickup_node  = pu,
            dropoff_node = do,
            earliest     = env.now,
            request_time = env.now,
        )
        req.direct_time      = travel_fn(pu, do, env.now)
        direct_times[req.id] = req.direct_time
        requests[req.id]     = req
        metrics.register(req.id, env.now, req.direct_time)

        clock = sim_time_to_clock(env.now)
        print(f"[{clock}] Request {req.id:>5}  {pu:>2} -> {do:<2}", end="  ")

        inserted = greedy_insert(
            req, vehicles, system_state, env.now, weights, metrics
        )

        if inserted:
            req.status          = RequestStatus.ASSIGNED
            req.assignment_time = env.now
            sa_improve(vehicles, system_state, env.now, weights, metrics)

        if verbose:
            print_plans(vehicles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: SimulationConfig = None, verbose: bool = False) -> MetricsCollector:
    if cfg is None:
        cfg = SimulationConfig()

    random.seed(cfg.seed)

    env          = simpy.Environment()
    coords       = DEFAULT_COORDS
    travel_fn    = make_travel_fn(coords)
    direct_times: dict = {}
    requests:     dict = {}
    metrics       = MetricsCollector()

    vehicles = {
        f"Bus-{k+1}": Vehicle(
            id       = f"Bus-{k+1}",
            capacity = cfg.vehicle_capacity,
            location = cfg.depot_node,
        )
        for k in range(cfg.fleet_size)
    }

    system_state = {
        "travel_time":  travel_fn,
        "ride_factor":  cfg.ride_factor,
        "direct_times": direct_times,
        "coords":       coords,
        "max_wait":     cfg.max_wait,
    }

    print(f"\nSimulation: {sim_time_to_clock(0)} - {sim_time_to_clock(cfg.horizon)}")
    print(f"Fleet     : {cfg.fleet_size} buses x capacity {cfg.vehicle_capacity}")
    print(f"Demand    : {cfg.n_requests} requests, profile={cfg.demand_profile}")
    print(f"Constraints: max_wait={cfg.max_wait} min, "
          f"ride_factor={cfg.ride_factor}")
    print("-" * 60)

    for vehicle in vehicles.values():
        env.process(vehicle_process(
            env, vehicle, travel_fn, system_state,
            requests, metrics, cfg, verbose,
        ))

    env.process(request_generator(
        env, vehicles, system_state, cfg.weights,
        cfg, direct_times, travel_fn, requests, metrics, verbose,
    ))

    env.run(until=cfg.horizon)

    metrics.print_summary()
    return metrics


if __name__ == "__main__":
    main()