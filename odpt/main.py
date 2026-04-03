# main.py
# Unified SimPy simulation — Malta On Demand (05:30 to 22:30).
# All times in MINUTES.  Simulation t=0 = 05:30.
#
# Single entry point for ALL policies. No more separate main files.
#
# Usage:
#   python main.py                                                  # greedy+sa (default)
#   python main.py --policy greedy                                  # greedy only (no SA)
#   python main.py --policy rl --model rl_outputs/run_008/model.zip # tuned RL
#   python main.py --policy rl --model rl_outputs/run_006/model.zip # base RL
#   python main.py --policy rl+sa --model rl_outputs/run_008/model.zip  # hybrid
#   python main.py --no-viz                                         # skip map
#   python main.py --verbose                                        # per-stop prints

import simpy
import random
import os
import argparse

from models import Request, Vehicle, RequestStatus
from malta_travel import DEFAULT_COORDS, make_travel_fn
from dispatcher import build_policy, dispatch_request, print_plans
from config import SimulationConfig, arrival_rate
from metrics import MetricsCollector


# ===================================================================
# Known model registry — quick labels for thesis runs
# ===================================================================
MODEL_REGISTRY = {
    "rl_tuned":   "rl_outputs/run_008/model.zip",
    "rl_base":    "rl_outputs/run_006/model.zip",
}


def sim_time_to_clock(t: float) -> str:
    """Convert simulation time (minutes from 05:30) to HH:MM clock string."""
    total = int(5 * 60 + 30 + t)
    return f"{total // 60:02d}:{total % 60:02d}"


def _policy_label(policy: str, model_path: str = None) -> str:
    """Human-readable label for the summary JSON and banner."""
    if model_path:
        # Check if it matches a known model
        for name, path in MODEL_REGISTRY.items():
            if model_path.rstrip("/\\") == path.rstrip("/\\"):
                return f"{policy} ({name})"
        # Fall back to directory name
        parts = model_path.replace("\\", "/").split("/")
        run_part = next((p for p in parts if p.startswith("run_")), model_path)
        return f"{policy} ({run_part})"
    return policy


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
    logger,
    sim_rng:      random.Random,
    noise_rng:    random.Random,
    verbose:      bool = False,
) -> None:

    if verbose:
        print(f"[{sim_time_to_clock(env.now)}] {vehicle.id} ready at depot")

    while True:
        if not vehicle.plan:
            vehicle.wake_event = env.event()
            yield vehicle.wake_event
            continue

        stop = vehicle.plan.pop(0)

        planned_travel = travel_fn(vehicle.location, stop.node, env.now)
        vehicle.in_transit_stop        = stop
        vehicle.in_transit_depart_time = env.now
        vehicle.in_transit_eta         = env.now + planned_travel

        if logger:
            logger.log_depart(
                env.now, sim_time_to_clock(env.now), vehicle.id,
                vehicle.location, stop.node, stop.req_id, stop.kind,
            )

        if verbose:
            print(f"[{sim_time_to_clock(env.now)}] {vehicle.id} "
                  f"-> {stop.kind} {stop.req_id} node {stop.node}")

        if cfg.travel_noise > 0 and planned_travel > 0:
            noise_factor = noise_rng.lognormvariate(0.0, cfg.travel_noise)
            actual_travel = planned_travel * noise_factor
        else:
            actual_travel = planned_travel

        metrics.log_distance(actual_travel)
        yield env.timeout(actual_travel)

        vehicle.location               = stop.node
        vehicle.in_transit_stop        = None
        vehicle.in_transit_depart_time = None
        vehicle.in_transit_eta         = None

        if logger:
            logger.log_arrive(
                env.now, sim_time_to_clock(env.now), vehicle.id,
                stop.node, stop.req_id, stop.kind,
            )

        if stop.kind == "PU" and stop.earliest and env.now < stop.earliest:
            yield env.timeout(stop.earliest - env.now)

        # --- Pickup ---
        if stop.kind == "PU":
            actual_wait = env.now - stop.request_time

            if stop.latest is not None and env.now > stop.latest:
                metrics.log_violation(
                    kind="wait", req_id=stop.req_id,
                    value=actual_wait, limit=cfg.max_wait, t=env.now,
                )

            vehicle.onboard.add(stop.req_id)
            vehicle.onboard_pickup_times[stop.req_id] = env.now
            metrics.mark_pickup(stop.req_id, env.now)

            if stop.req_id in requests:
                req             = requests[stop.req_id]
                req.pickup_time = env.now
                req.status      = RequestStatus.ONBOARD

            if logger:
                logger.log_pickup(
                    env.now, sim_time_to_clock(env.now), vehicle.id,
                    stop.node, stop.req_id, actual_wait,
                )

            if verbose:
                print(f"[{sim_time_to_clock(env.now)}] {vehicle.id} "
                      f"picked up {stop.req_id}  "
                      f"(waited {actual_wait:.1f} min)")

        # --- Dropoff ---
        elif stop.kind == "DO":
            vehicle.onboard.discard(stop.req_id)
            vehicle.onboard_pickup_times.pop(stop.req_id, None)
            metrics.mark_dropoff(stop.req_id, env.now)

            ride_time = None
            if stop.req_id in requests:
                req              = requests[stop.req_id]
                req.dropoff_time = env.now
                req.status       = RequestStatus.COMPLETED

                if req.pickup_time is not None and req.direct_time:
                    actual_ride = env.now - req.pickup_time
                    ride_time = actual_ride
                    max_ride    = cfg.ride_factor * req.direct_time
                    if actual_ride > max_ride:
                        metrics.log_violation(
                            kind="ride", req_id=stop.req_id,
                            value=actual_ride, limit=max_ride, t=env.now,
                        )

            if logger:
                logger.log_dropoff(
                    env.now, sim_time_to_clock(env.now), vehicle.id,
                    stop.node, stop.req_id, ride_time,
                )

            if verbose:
                print(f"[{sim_time_to_clock(env.now)}] {vehicle.id} "
                      f"dropped off {stop.req_id}")

        yield env.timeout(stop.service)


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
    logger,
    sim_rng:      random.Random,
    verbose:      bool = False,
) -> None:

    for i in range(1, cfg.n_requests + 1):
        mean_gap = arrival_rate(env.now, cfg)
        if cfg.stochastic_arrivals:
            gap = sim_rng.expovariate(1.0 / mean_gap)
        else:
            gap = mean_gap
        yield env.timeout(gap)

        if env.now >= cfg.service_end:
            break

        pu = sim_rng.randint(1, cfg.n_nodes)
        do = sim_rng.randint(1, cfg.n_nodes)
        while do == pu:
            do = sim_rng.randint(1, cfg.n_nodes)

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

        if logger:
            logger.log_request(env.now, clock, req.id, pu, do)

        # === SINGLE DISPATCH CALL — policy handled internally ===
        inserted = dispatch_request(
            req, vehicles, system_state, env.now, weights, metrics,
        )

        if inserted:
            req.status          = RequestStatus.ASSIGNED
            req.assignment_time = env.now
        else:
            if logger:
                logger.log_reject(env.now, clock, req.id, pu, do)

        if verbose:
            print_plans(vehicles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    cfg:        SimulationConfig = None,
    model_path: str  = None,
    verbose:    bool = False,
    visualize:  bool = True,
) -> MetricsCollector:

    if cfg is None:
        cfg = SimulationConfig()

    # ---------------------------------------------------------------
    # Three independent RNG instances — the correct reproducibility fix.
    #
    # _sim_rng   drives ONLY request arrivals: inter-arrival gaps and
    #            pickup/dropoff node sampling.  Identical sequence on
    #            every run regardless of algorithm or vehicle movements.
    #
    # _noise_rng drives ONLY travel noise: lognormal draw per vehicle
    #            leg.  Separated from _sim_rng so that vehicle movement
    #            patterns (which differ per algorithm) never shift the
    #            arrival sequence.
    #
    # _algo_rng  drives ONLY algorithm internals: SA temperature draws,
    #            GA crossover/mutation, TS neighbourhood sampling, ALNS
    #            operator selection.
    #
    # Result: every policy on seed=42 processes the exact same request
    # stream with the exact same inter-arrival gaps and node assignments.
    # ---------------------------------------------------------------
    _sim_rng   = random.Random(cfg.seed)
    _noise_rng = random.Random(cfg.seed + 1)
    _algo_rng  = random.Random(cfg.seed + 999)

    env          = simpy.Environment()
    coords       = DEFAULT_COORDS
    travel_fn    = make_travel_fn(coords)
    direct_times: dict = {}
    requests:     dict = {}
    metrics       = MetricsCollector()

    # Optional visualisation logger (don't crash if visualize.py is missing)
    logger = None
    try:
        from visualize import EventLogger
        logger = EventLogger()
    except ImportError:
        pass

    # === Build the dispatch policy ===
    build_policy(cfg, model_path=model_path, algo_rng=_algo_rng)

    vehicles = {
        f"Bus-{k+1}": Vehicle(
            id       = f"Bus-{k+1}",
            capacity = cfg.vehicle_capacity,
            location = cfg.depot_node,
        )
        for k in range(cfg.fleet_size)
    }

    system_state = {
        "travel_time":      travel_fn,
        "ride_factor":      cfg.ride_factor,
        "direct_times":     direct_times,
        "coords":           coords,
        "max_wait":         cfg.max_wait,
        "ride_time_margin": cfg.ride_time_margin,
    }

    # === Banner — adapts to active policy ===
    label = _policy_label(cfg.policy, model_path)
    print(f"\n{'=' * 60}")
    print(f"SIMULATION — {label}")
    print(f"{'=' * 60}")
    print(f"  Service   : {sim_time_to_clock(0)} - {sim_time_to_clock(cfg.service_end)}"
          f"  (sim until {sim_time_to_clock(cfg.horizon)})")
    print(f"  Fleet     : {cfg.fleet_size} buses x capacity {cfg.vehicle_capacity}")
    print(f"  Demand    : {cfg.n_requests} requests, "
          f"profile={cfg.demand_profile}, "
          f"stochastic={cfg.stochastic_arrivals}")
    print(f"  Constraints: max_wait={cfg.max_wait} min, "
          f"ride_factor={cfg.ride_factor}")
    print(f"  Policy    : {label}")

    if "sa" in cfg.policy.lower():
        print(f"  SA params : T0={cfg.sa_initial_temp}, "
              f"cool={cfg.sa_cooling_rate}, "
              f"iters={cfg.sa_iterations}/vehicle, "
              f"time={cfg.sa_time_limit}s/vehicle")

    if "ga" in cfg.policy.lower():
        print(f"  GA params : pop={cfg.ga_population}, "
              f"gen={cfg.ga_generations}, "
              f"cx={cfg.ga_crossover}, mut={cfg.ga_mutation}, "
              f"time={cfg.ga_time_limit}s/vehicle")

    if "ts" in cfg.policy.lower():
        print(f"  TS params : tenure={cfg.ts_tabu_tenure}, "
              f"neighbours={cfg.ts_max_neighbours}, "
              f"iters={cfg.ts_iterations}/vehicle, "
              f"patience={cfg.ts_patience}, "
              f"time={cfg.ts_time_limit}s/vehicle")

    if "alns" in cfg.policy.lower():
        print(f"  ALNS params: iters={cfg.alns_iterations}, "
              f"q=[{cfg.alns_q_min},{cfg.alns_q_max}], "
              f"reaction={cfg.alns_reaction}, "
              f"time={cfg.alns_time_limit}s/vehicle")

    if model_path:
        print(f"  Model     : {model_path}")

    print(f"{'=' * 60}\n")

    for vehicle in vehicles.values():
        env.process(vehicle_process(
            env, vehicle, travel_fn, system_state,
            requests, metrics, cfg, logger, _sim_rng, _noise_rng, verbose,
        ))

    env.process(request_generator(
        env, vehicles, system_state, cfg.weights,
        cfg, direct_times, travel_fn, requests, metrics, logger, _sim_rng, verbose,
    ))

    env.run(until=cfg.horizon)

    metrics.print_summary()

    # --- Generate outputs ---
    run_dir = _make_run_dir(cfg)
    _save_summary(metrics, cfg, label, run_dir)

    if visualize and logger:
        try:
            from visualize import generate_map
            print("\nGenerating visualization...")
            events_path = os.path.join(run_dir, "events.json")
            map_path    = os.path.join(run_dir, "map.html")
            logger.to_json(events_path)
            generate_map(logger, map_path)
        except ImportError:
            print("  (visualize module not found — skipping map)")

    print(f"\nOutputs saved to: {run_dir}")
    return metrics


def _make_run_dir(cfg: SimulationConfig) -> str:
    os.makedirs("outputs", exist_ok=True)
    existing = [
        d for d in os.listdir("outputs")
        if os.path.isdir(os.path.join("outputs", d)) and d.startswith("run_")
    ]
    if existing:
        nums = []
        for d in existing:
            try:
                nums.append(int(d.split("_")[1]))
            except (IndexError, ValueError):
                pass
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1
    run_dir = os.path.join("outputs", f"run_{next_num:03d}")
    os.makedirs(run_dir)
    return run_dir


def _save_summary(
    metrics: MetricsCollector,
    cfg: SimulationConfig,
    policy_label: str,
    run_dir: str,
):
    import json as _json

    summary = {
        "config": {
            "seed":                cfg.seed,
            "service_end":         cfg.service_end,
            "horizon":             cfg.horizon,
            "n_requests":          cfg.n_requests,
            "inter_arrival":       cfg.inter_arrival,
            "demand_profile":      cfg.demand_profile,
            "stochastic_arrivals": cfg.stochastic_arrivals,
            "n_nodes":             cfg.n_nodes,
            "fleet_size":          cfg.fleet_size,
            "vehicle_capacity":    cfg.vehicle_capacity,
            "depot_node":          cfg.depot_node,
            "ride_factor":         cfg.ride_factor,
            "max_wait":            cfg.max_wait,
            "ride_time_margin":    cfg.ride_time_margin,
            "travel_noise":        cfg.travel_noise,
            "weights":             list(cfg.weights),
            "policy":              policy_label,
        },
        "metrics": metrics.summary(),
    }

    # Only include SA params if SA is active
    if "sa" in cfg.policy.lower():
        summary["config"]["sa_initial_temp"] = cfg.sa_initial_temp
        summary["config"]["sa_cooling_rate"] = cfg.sa_cooling_rate
        summary["config"]["sa_iterations"]   = cfg.sa_iterations
        summary["config"]["sa_time_limit"]   = cfg.sa_time_limit

    # Only include GA params if GA is active
    if "ga" in cfg.policy.lower():
        summary["config"]["ga_population"]   = cfg.ga_population
        summary["config"]["ga_generations"]  = cfg.ga_generations
        summary["config"]["ga_crossover"]    = cfg.ga_crossover
        summary["config"]["ga_mutation"]     = cfg.ga_mutation
        summary["config"]["ga_tournament"]   = cfg.ga_tournament
        summary["config"]["ga_elite"]        = cfg.ga_elite
        summary["config"]["ga_time_limit"]   = cfg.ga_time_limit

    # Only include TS params if TS is active
    if "ts" in cfg.policy.lower():
        summary["config"]["ts_tabu_tenure"]    = cfg.ts_tabu_tenure
        summary["config"]["ts_max_neighbours"] = cfg.ts_max_neighbours
        summary["config"]["ts_iterations"]     = cfg.ts_iterations
        summary["config"]["ts_patience"]       = cfg.ts_patience
        summary["config"]["ts_time_limit"]     = cfg.ts_time_limit

    # Only include ALNS params if ALNS is active
    if "alns" in cfg.policy.lower():
        summary["config"]["alns_iterations"]  = cfg.alns_iterations
        summary["config"]["alns_q_min"]       = cfg.alns_q_min
        summary["config"]["alns_q_max"]       = cfg.alns_q_max
        summary["config"]["alns_reaction"]    = cfg.alns_reaction
        summary["config"]["alns_temp_factor"] = cfg.alns_temp_factor
        summary["config"]["alns_cooling"]     = cfg.alns_cooling
        summary["config"]["alns_time_limit"]  = cfg.alns_time_limit

    path = os.path.join(run_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Malta On Demand — Dynamic DARP Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py                                                   # greedy+sa (default)
  python main.py --policy greedy                                   # greedy only
  python main.py --policy greedy+sa                                # greedy+sa
  python main.py --policy greedy+ga                                # greedy+ga
  python main.py --policy greedy+ts                                # greedy+ts
  python main.py --policy greedy+alns                              # greedy+alns
  python main.py --policy rl --model rl_outputs/run_008/model.zip  # tuned RL
  python main.py --policy rl --model rl_tuned                      # shortcut
  python main.py --policy rl+sa  --model rl_tuned                  # RL + SA
  python main.py --policy rl+ga  --model rl_tuned                  # RL + GA
  python main.py --policy rl+ts  --model rl_tuned                  # RL + TS
  python main.py --policy rl+alns --model rl_tuned                 # RL + ALNS
""",
    )
    parser.add_argument(
        "--policy", default="greedy+sa",
        choices=["greedy", "greedy+sa", "greedy+ga", "greedy+ts", "greedy+alns",
                 "rl", "rl+sa", "rl+ga", "rl+ts", "rl+alns"],
        help="Dispatch policy (default: greedy+sa)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to RL model.zip, or shortcut name: 'rl_tuned', 'rl_base'",
    )
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--fleet",    type=int, default=6)
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--no-viz",   action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve model shortcut names
    model_path = args.model
    if model_path and model_path in MODEL_REGISTRY:
        model_path = MODEL_REGISTRY[model_path]

    # Validate: RL policies need a model
    if "rl" in args.policy and not model_path:
        print(f"ERROR: --policy {args.policy} requires --model <path>")
        print(f"  Available shortcuts: {list(MODEL_REGISTRY.keys())}")
        exit(1)

    cfg = SimulationConfig(
        seed       = args.seed,
        n_requests = args.requests,
        fleet_size = args.fleet,
        policy     = args.policy,
    )

    main(
        cfg        = cfg,
        model_path = model_path,
        verbose    = args.verbose,
        visualize  = not args.no_viz,
    )