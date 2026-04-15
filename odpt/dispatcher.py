# dispatcher.py
# Unified dispatch interface — plug any policy into the simulation.
#
# Architecture:
#   dispatch_request() is the SINGLE entry point called by main.py.
#   It delegates to the active policy based on build_policy() config.
#   All policies share the same feasibility checker and cost evaluator.
#
# Supported policies (cfg.policy string):
#   "greedy"      — greedy best-position insertion only
#   "greedy+sa"   — greedy insertion + SA improvement pass  (default)
#   "rl"          — RL vehicle assignment + greedy within-vehicle
#   "rl+sa"       — RL assignment + greedy insertion + SA improvement
#
# Usage:
#   from dispatcher import build_policy, dispatch_request, print_plans
#   build_policy(cfg, model_path="rl_outputs/run_008/model.zip")
#   inserted = dispatch_request(req, vehicles, system_state, now, weights, metrics)

from __future__ import annotations

import random
import time as _time
from copy import deepcopy
from typing import Optional

from models import Stop, Request, Vehicle
from feasibility import check_feasibility, evaluate_plan
from metrics import MetricsCollector
from sa import SAPolicy
from ga import GAPolicy
from ts import TSPolicy
from alns import ALNSPolicy


# ===================================================================
# Module-level policy state
# ===================================================================

_POLICY_NAME: str = "greedy+sa"
_SA_POLICY:   Optional[SAPolicy]   = None
_GA_POLICY:   Optional[GAPolicy]   = None
_TS_POLICY:   Optional[TSPolicy]   = None
_ALNS_POLICY: Optional[ALNSPolicy] = None
_RL_MODEL = None
_RL_CFG   = None
# Dedicated RNG for algorithm internals — seeded independently of the
# simulation RNG so that SA/GA/TS/ALNS random calls never advance the
# state that drives demand arrivals and travel noise.
_ALGO_RNG: Optional[random.Random] = None


def build_policy(
    cfg,
    model_path: str = None,
    algo_rng: Optional[random.Random] = None,
) -> None:
    """
    Initialise the dispatch policy.

    Parameters
    ----------
    cfg : SimulationConfig
        cfg.policy selects the algorithm.
    model_path : str, optional
        Path to trained MaskablePPO model.zip (required for RL policies).
    algo_rng : random.Random, optional
        Dedicated RNG instance for algorithm internals.  Must be seeded
        independently of the simulation RNG that drives demand and noise.
        If None, each policy creates its own Random() from os.urandom —
        results are non-deterministic across algorithm comparisons.
    """
    global _POLICY_NAME, _SA_POLICY, _GA_POLICY, _TS_POLICY, _ALNS_POLICY, \
           _RL_MODEL, _RL_CFG, _ALGO_RNG

    _POLICY_NAME = cfg.policy.lower().strip()
    _RL_CFG      = cfg
    _ALGO_RNG    = algo_rng  # stored so _improve functions can inspect if needed

    # --- SA setup ---
    if "sa" in _POLICY_NAME:
        _SA_POLICY = SAPolicy(
            initial_temp        = cfg.sa_initial_temp,
            cooling_rate        = cfg.sa_cooling_rate,
            iterations          = cfg.sa_iterations,
            decision_time_limit = cfg.sa_time_limit,
            rng                 = algo_rng,
        )
    else:
        _SA_POLICY = None

    # --- GA setup ---
    if "ga" in _POLICY_NAME:
        _GA_POLICY = GAPolicy(
            population_size     = cfg.ga_population,
            generations         = cfg.ga_generations,
            crossover_rate      = cfg.ga_crossover,
            mutation_rate       = cfg.ga_mutation,
            tournament_size     = cfg.ga_tournament,
            elite_count         = cfg.ga_elite,
            decision_time_limit = cfg.ga_time_limit,
            rng                 = algo_rng,
        )
    else:
        _GA_POLICY = None

    # --- TS setup ---
    if "ts" in _POLICY_NAME:
        _TS_POLICY = TSPolicy(
            tabu_tenure         = cfg.ts_tabu_tenure,
            max_neighbours      = cfg.ts_max_neighbours,
            iterations          = cfg.ts_iterations,
            patience            = cfg.ts_patience,
            decision_time_limit = cfg.ts_time_limit,
            rng                 = algo_rng,
        )
    else:
        _TS_POLICY = None

    # --- ALNS setup ---
    if "alns" in _POLICY_NAME:
        _ALNS_POLICY = ALNSPolicy(
            iterations          = cfg.alns_iterations,
            q_min               = cfg.alns_q_min,
            q_max               = cfg.alns_q_max,
            reaction_factor     = cfg.alns_reaction,
            initial_temp_factor = cfg.alns_temp_factor,
            cooling_rate        = cfg.alns_cooling,
            decision_time_limit = cfg.alns_time_limit,
            rng                 = algo_rng,
        )
    else:
        _ALNS_POLICY = None

    # --- RL model setup ---
    if "rl" in _POLICY_NAME:
        if model_path is None:
            raise ValueError(
                f"Policy '{_POLICY_NAME}' requires model_path "
                f"pointing to a trained .zip file."
            )
        from sb3_contrib import MaskablePPO
        _RL_MODEL = MaskablePPO.load(model_path, device="auto")
        print(f"  RL model loaded: {model_path}")
    else:
        _RL_MODEL = None


# Legacy aliases — keep rl_dispatcher.py and old mains importable
# without breaking, but they delegate to the unified interface.
def build_sa_policy(cfg):
    """Legacy: initialise SA-only. Prefer build_policy()."""
    global _SA_POLICY
    _SA_POLICY = SAPolicy(
        initial_temp        = cfg.sa_initial_temp,
        cooling_rate        = cfg.sa_cooling_rate,
        iterations          = cfg.sa_iterations,
        decision_time_limit = cfg.sa_time_limit,
    )


# ===================================================================
# Public dispatch entry point
# ===================================================================

def dispatch_request(
    request:      Request,
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights:      tuple[float, float, float],
    metrics:      Optional[MetricsCollector] = None,
) -> bool:
    """
    Insert *request* into a vehicle plan using the active policy.

    Returns True if inserted, False if rejected.
    This is the ONLY function main.py needs to call.

    Latency logged here covers the FULL decision epoch:
    insertion (greedy or RL) + improvement pass (SA, GA, or TS).
    This is the correct definition for thesis latency comparisons.
    """
    t0 = _time.time()

    if "rl" in _POLICY_NAME:
        inserted = _rl_insert(
            request, vehicles, system_state, current_time, weights, metrics,
        )
    else:
        inserted = _greedy_insert(
            request, vehicles, system_state, current_time, weights, metrics,
        )

    # SA improvement pass (for greedy+sa and rl+sa)
    if inserted and _SA_POLICY is not None:
        _sa_improve(vehicles, system_state, current_time, weights, metrics)

    # GA improvement pass (for greedy+ga and rl+ga)
    if inserted and _GA_POLICY is not None:
        _ga_improve(vehicles, system_state, current_time, weights, metrics)

    # TS improvement pass (for greedy+ts and rl+ts)
    if inserted and _TS_POLICY is not None:
        _ts_improve(vehicles, system_state, current_time, weights, metrics)

    # ALNS improvement pass (for greedy+alns and rl+alns)
    if inserted and _ALNS_POLICY is not None:
        _alns_improve(vehicles, system_state, current_time, weights, metrics)

    # Log total decision latency (insertion + any improvement pass)
    if metrics:
        metrics.log_decision_latency(_time.time() - t0)

    return inserted


# ===================================================================
# Legacy public aliases (so old code still works if imported directly)
# ===================================================================

def greedy_insert(request, vehicles, system_state, current_time, weights, metrics=None):
    """Legacy wrapper — greedy insertion only, no SA."""
    return _greedy_insert(request, vehicles, system_state, current_time, weights, metrics)


def sa_improve(vehicles, system_state, current_time, weights, metrics=None):
    """Legacy wrapper — SA improvement pass."""
    _sa_improve(vehicles, system_state, current_time, weights, metrics)


# ===================================================================
# Greedy insertion (internal)
# ===================================================================

def _greedy_insert(
    request:      Request,
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights:      tuple[float, float, float],
    metrics:      Optional[MetricsCollector] = None,
) -> bool:
    """
    Insert *request* into the best feasible position across all vehicles.
    Returns True if inserted, False if rejected.
    """
    best_cost   = float("inf")
    best_choice = None
    max_wait    = system_state.get("max_wait", float("inf"))

    for vid, vehicle in vehicles.items():
        result = _find_best_insertion(
            request, vehicle, current_time, system_state, weights, max_wait,
        )
        if result is not None:
            candidate, n_committed, cost = result
            if cost < best_cost:
                best_cost   = cost
                best_choice = (vid, candidate, n_committed)

    if best_choice:
        vid, full_candidate, n_committed = best_choice
        vehicle = vehicles[vid]
        vehicle.plan = full_candidate[n_committed:]

        # Wake idle vehicle
        if vehicle.wake_event is not None and not vehicle.wake_event.triggered:
            vehicle.wake_event.succeed()

        print(f"  -> Inserted {request.id} into {vid}  cost={best_cost:.2f}")
        return True

    print(f"  -> Rejected {request.id} (no feasible insertion)")
    if metrics:
        metrics.mark_rejected(request.id)
    return False


# ===================================================================
# RL insertion (internal)
# ===================================================================

def _rl_insert(
    request:      Request,
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights:      tuple[float, float, float],
    metrics:      Optional[MetricsCollector] = None,
) -> bool:
    """
    RL vehicle assignment + greedy best-position within-vehicle insertion.
    Returns True if inserted, False if rejected.
    """
    import numpy as np

    model = _RL_MODEL
    if model is None:
        raise RuntimeError("RL model not loaded. Call build_policy() first.")

    vehicle_ids = sorted(vehicles.keys())
    n_vehicles = len(vehicle_ids)
    max_wait = system_state.get("max_wait", float("inf"))

    # Find best insertion per vehicle (for routing, not assignment)
    n_actions = n_vehicles + 1
    mask = np.zeros(n_actions, dtype=np.int8)
    mask[0] = 1  # reject always available
    insertions = {}

    for idx, vid in enumerate(vehicle_ids):
        result = _find_best_insertion(
            request, vehicles[vid], current_time, system_state, weights, max_wait,
        )
        if result is not None:
            mask[idx + 1] = 1
            insertions[vid] = result

    # If no vehicle can take this request, reject
    if mask.sum() <= 1:
        if metrics:
            metrics.mark_rejected(request.id)
        print(f"  -> RL rejected {request.id} (no feasible vehicle)")
        return False

    # Pad mask to model's trained action space size if fleet size differs.
    # The model was trained with fleet_size=6 (7 actions). When running a
    # sensitivity scenario with fewer/more vehicles, pad with zeros (masked out).
    model_n_actions = model.action_space.n
    if len(mask) < model_n_actions:
        mask = np.concatenate([mask, np.zeros(model_n_actions - len(mask), dtype=np.int8)])
    elif len(mask) > model_n_actions:
        # More vehicles than trained on — truncate to model capacity
        mask = mask[:model_n_actions]
        vehicle_ids = vehicle_ids[:model_n_actions - 1]
        n_vehicles = len(vehicle_ids)

    # Encode state and get RL action
    obs = _encode_live_state(
        request, vehicles, vehicle_ids, current_time, system_state,
        _RL_CFG, metrics,
    )
    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
    action = int(action)

    if action == 0 or action > n_vehicles:
        if metrics:
            metrics.mark_rejected(request.id)
        print(f"  -> RL rejected {request.id} (agent chose reject)")
        return False

    vid = vehicle_ids[action - 1]
    if vid not in insertions:
        if metrics:
            metrics.mark_rejected(request.id)
        print(f"  -> RL rejected {request.id} (chosen vehicle infeasible)")
        return False

    candidate_plan, n_committed, cost = insertions[vid]
    vehicle = vehicles[vid]
    vehicle.plan = candidate_plan[n_committed:]

    # Wake idle vehicle
    if (vehicle.wake_event is not None
            and not vehicle.wake_event.triggered
            and vehicle.plan):
        vehicle.wake_event.succeed()

    print(f"  -> RL inserted {request.id} into {vid}  cost={cost:.2f}")
    return True


# ===================================================================
# Shared: best-position insertion for one vehicle
# ===================================================================

def _find_best_insertion(
    request:      Request,
    vehicle:      Vehicle,
    current_time: float,
    system_state: dict,
    weights:      tuple,
    max_wait:     float = float("inf"),
) -> Optional[tuple]:
    """
    Find the best feasible (PU, DO) insertion positions for *request*
    in *vehicle*'s plan.

    Returns (candidate_plan, n_committed, cost) or None.
    """
    v_state = vehicle.to_state_dict(current_time)
    full_plan = v_state["plan_snapshot"]
    n_committed = 1 if vehicle.in_transit_stop is not None else 0
    insertable = full_plan[n_committed:]
    n = len(insertable)

    best_cost = float("inf")
    best_candidate = None

    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            candidate_tail = list(insertable)

            pu = Stop(
                node         = request.pickup_node,
                kind         = "PU",
                req_id       = request.id,
                earliest     = request.earliest,
                latest       = request.request_time + max_wait,
                service      = 1.0,
                request_time = request.request_time,
            )
            do = Stop(
                node         = request.dropoff_node,
                kind         = "DO",
                req_id       = request.id,
                earliest     = None,
                latest       = None,
                service      = 1.0,
                request_time = request.request_time,
            )

            candidate_tail.insert(i, pu)
            candidate_tail.insert(j, do)
            candidate = full_plan[:n_committed] + candidate_tail

            if not check_feasibility(candidate, v_state, system_state):
                continue

            cost = evaluate_plan(candidate, v_state, system_state, weights)
            if cost < best_cost:
                best_cost = cost
                best_candidate = candidate

    if best_candidate is not None:
        return best_candidate, n_committed, best_cost
    return None


# ===================================================================
# SA improvement pass (internal)
# ===================================================================

def _sa_improve(
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights:      tuple[float, float, float],
    metrics:      Optional[MetricsCollector] = None,
) -> None:
    """
    Run one SA improvement pass over all vehicle plans.
    Plans updated only when SA finds a strictly better solution.
    """
    if _SA_POLICY is None:
        return

    sa_system_state = {
        **system_state,
        "vehicles": {},
    }

    for vid, v in vehicles.items():
        vs = v.to_state_dict(current_time)
        n_committed = 1 if v.in_transit_stop is not None else 0
        sa_system_state["vehicles"][vid] = {
            **vs,
            "plan":        deepcopy(vs["plan_snapshot"]),
            "n_committed": n_committed,
        }

    changes = _SA_POLICY.propose(
        sa_system_state, check_feasibility, weights,
    )

    if not changes:
        return

    # Verify combined improvement
    total_before = 0.0
    total_after  = 0.0
    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        v_state = vehicle.to_state_dict(current_time)
        total_before += evaluate_plan(
            v_state["plan_snapshot"], v_state, system_state, weights,
        )
        total_after += evaluate_plan(
            new_plan, v_state, system_state, weights,
        )

    if total_after >= total_before:
        return

    # Apply all changes atomically
    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        n_committed = 1 if vehicle.in_transit_stop is not None else 0
        vehicle.plan = new_plan[n_committed:]

        if (vehicle.wake_event is not None
                and not vehicle.wake_event.triggered
                and vehicle.plan):
            vehicle.wake_event.succeed()

    if metrics:
        metrics.log_improvement()


# ===================================================================
# GA improvement pass (internal)
# ===================================================================

def _ga_improve(
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights:      tuple[float, float, float],
    metrics:      Optional[MetricsCollector] = None,
) -> None:
    """
    Run one GA improvement pass over all vehicle plans.
    Structurally identical to _sa_improve — only the policy object differs.
    Plans are updated only when GA finds a strictly better combined solution.
    """
    if _GA_POLICY is None:
        return

    ga_system_state = {
        **system_state,
        "vehicles": {},
    }

    for vid, v in vehicles.items():
        vs = v.to_state_dict(current_time)
        n_committed = 1 if v.in_transit_stop is not None else 0
        ga_system_state["vehicles"][vid] = {
            **vs,
            "plan":        deepcopy(vs["plan_snapshot"]),
            "n_committed": n_committed,
        }

    changes = _GA_POLICY.propose(
        ga_system_state, check_feasibility, weights,
    )

    if not changes:
        return

    # Verify combined improvement across all touched vehicles
    total_before = 0.0
    total_after  = 0.0
    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        v_state = vehicle.to_state_dict(current_time)
        total_before += evaluate_plan(
            v_state["plan_snapshot"], v_state, system_state, weights,
        )
        total_after += evaluate_plan(
            new_plan, v_state, system_state, weights,
        )

    if total_after >= total_before:
        return

    # Apply all changes atomically
    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        n_committed = 1 if vehicle.in_transit_stop is not None else 0
        vehicle.plan = new_plan[n_committed:]

        if (vehicle.wake_event is not None
                and not vehicle.wake_event.triggered
                and vehicle.plan):
            vehicle.wake_event.succeed()

    if metrics:
        metrics.log_improvement()


# ===================================================================
# TS improvement pass (internal)
# ===================================================================

def _ts_improve(
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights:      tuple[float, float, float],
    metrics:      Optional[MetricsCollector] = None,
) -> None:
    """
    Run one TS improvement pass over all vehicle plans.
    Structurally identical to _sa_improve and _ga_improve.
    Plans are updated only when TS finds a strictly better combined solution.
    """
    if _TS_POLICY is None:
        return

    ts_system_state = {
        **system_state,
        "vehicles": {},
    }

    for vid, v in vehicles.items():
        vs = v.to_state_dict(current_time)
        n_committed = 1 if v.in_transit_stop is not None else 0
        ts_system_state["vehicles"][vid] = {
            **vs,
            "plan":        deepcopy(vs["plan_snapshot"]),
            "n_committed": n_committed,
        }

    changes = _TS_POLICY.propose(
        ts_system_state, check_feasibility, weights,
    )

    if not changes:
        return

    # Verify combined improvement across all touched vehicles
    total_before = 0.0
    total_after  = 0.0
    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        v_state = vehicle.to_state_dict(current_time)
        total_before += evaluate_plan(
            v_state["plan_snapshot"], v_state, system_state, weights,
        )
        total_after += evaluate_plan(
            new_plan, v_state, system_state, weights,
        )

    if total_after >= total_before:
        return

    # Apply all changes atomically
    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        n_committed = 1 if vehicle.in_transit_stop is not None else 0
        vehicle.plan = new_plan[n_committed:]

        if (vehicle.wake_event is not None
                and not vehicle.wake_event.triggered
                and vehicle.plan):
            vehicle.wake_event.succeed()

    if metrics:
        metrics.log_improvement()


# ===================================================================
# ALNS improvement pass (internal)
# ===================================================================

def _alns_improve(
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights:      tuple[float, float, float],
    metrics:      Optional[MetricsCollector] = None,
) -> None:
    """
    Run one ALNS improvement pass over all vehicle plans.
    Structurally identical to _sa_improve / _ga_improve / _ts_improve.
    Plans are updated only when ALNS finds a strictly better combined solution.
    """
    if _ALNS_POLICY is None:
        return

    alns_system_state = {
        **system_state,
        "vehicles": {},
    }

    for vid, v in vehicles.items():
        vs = v.to_state_dict(current_time)
        n_committed = 1 if v.in_transit_stop is not None else 0
        alns_system_state["vehicles"][vid] = {
            **vs,
            "plan":        deepcopy(vs["plan_snapshot"]),
            "n_committed": n_committed,
        }

    changes = _ALNS_POLICY.propose(
        alns_system_state, check_feasibility, weights,
    )

    if not changes:
        return

    # Verify combined improvement across all touched vehicles
    total_before = 0.0
    total_after  = 0.0
    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        v_state = vehicle.to_state_dict(current_time)
        total_before += evaluate_plan(
            v_state["plan_snapshot"], v_state, system_state, weights,
        )
        total_after += evaluate_plan(
            new_plan, v_state, system_state, weights,
        )

    if total_after >= total_before:
        return

    # Apply all changes atomically
    for vid, new_plan in changes.items():
        vehicle = vehicles[vid]
        n_committed = 1 if vehicle.in_transit_stop is not None else 0
        vehicle.plan = new_plan[n_committed:]

        if (vehicle.wake_event is not None
                and not vehicle.wake_event.triggered
                and vehicle.plan):
            vehicle.wake_event.succeed()

    if metrics:
        metrics.log_improvement()


# ===================================================================
# RL state encoding (mirrors rl_env._encode_state exactly)
# ===================================================================

def _encode_live_state(
    request, vehicles, vehicle_ids, current_time, system_state, cfg,
    metrics=None,
):
    """Encode live SimPy state into the RL observation vector.

    Supports three observation layouts:
      - Standard (74 dims): USE_V6_FEATURES=False, USE_ANTICIPATORY_FEATURES=False
      - Anticipatory (78 dims): USE_ANTICIPATORY_FEATURES=True
      - V6 (106 dims): USE_V6_FEATURES=True — 12 per-vehicle features + 4 global
    """
    import numpy as np
    import rl_env as _rl_env
    from rl_env import (get_obs_size, OBS_PER_VEHICLE, OBS_REQUEST,
                        MAX_VEHICLES, USE_ANTICIPATORY_FEATURES,
                        USE_V6_FEATURES, OBS_PER_VEHICLE_V6, get_obs_size_v6)
    from malta_travel import DEFAULT_COORDS, congestion_factor
    from config import arrival_rate

    obs = np.zeros(get_obs_size_v6(), dtype=np.float32)

    lon_min, lon_max = 14.35, 14.55
    lat_min, lat_max = 35.85, 35.95
    travel_fn = system_state["travel_time"]

    def norm_coord(node_id):
        if node_id in DEFAULT_COORDS:
            lon, lat = DEFAULT_COORDS[node_id]
            x = (lon - lon_min) / (lon_max - lon_min) * 2 - 1
            y = (lat - lat_min) / (lat_max - lat_min) * 2 - 1
            return np.clip(x, -1, 1), np.clip(y, -1, 1)
        return 0.0, 0.0

    # Per-vehicle features — 12 for v6, 8 otherwise
    veh_feats = OBS_PER_VEHICLE_V6 if USE_V6_FEATURES else OBS_PER_VEHICLE

    for i, vid in enumerate(vehicle_ids[:MAX_VEHICLES]):
        v = vehicles[vid]
        base = i * veh_feats
        x, y = norm_coord(v.location)
        obs[base + 0] = x
        obs[base + 1] = y
        obs[base + 2] = len(v.onboard) / max(v.capacity, 1)
        obs[base + 3] = min(len(v.plan) / 20.0, 1.0)
        obs[base + 4] = 1.0 if not v.plan and v.in_transit_stop is None else 0.0
        if v.plan:
            t_next = travel_fn(v.location, v.plan[0].node, current_time)
            obs[base + 5] = min(t_next / 30.0, 1.0)
        d_pu = travel_fn(v.location, request.pickup_node, current_time)
        d_do = travel_fn(v.location, request.dropoff_node, current_time)
        obs[base + 6] = min(d_pu / 30.0, 1.0)
        obs[base + 7] = min(d_do / 30.0, 1.0)

        # V6 extra features [8-11]: schedule tightness
        if USE_V6_FEATURES:
            if not v.plan:
                obs[base + 8]  = 0.0
                obs[base + 9]  = 1.0
                obs[base + 10] = 1.0
                obs[base + 11] = 0.0
            else:
                cur_node = v.location
                cur_time = current_time
                pu_slacks, wait_qualities, urgencies = [], [], []
                for stop in v.plan:
                    cur_time += travel_fn(cur_node, stop.node, cur_time)
                    if stop.kind == "PU":
                        if stop.earliest and cur_time < stop.earliest:
                            cur_time = stop.earliest
                        latest = getattr(stop, "latest", None)
                        if latest is not None:
                            pu_slacks.append(
                                max(0.0, latest - cur_time) / max(cfg.max_wait, 1.0)
                            )
                        if stop.request_time is not None:
                            est_wait = max(0.0, cur_time - stop.request_time)
                            wait_qualities.append(
                                1.0 - min(est_wait / max(cfg.max_wait, 1.0), 1.0)
                            )
                            elapsed = current_time - stop.request_time
                            urgencies.append(
                                min(max(elapsed, 0.0) / max(cfg.max_wait, 1.0), 1.0)
                            )
                    cur_time += stop.service
                    cur_node  = stop.node
                makespan = cur_time - current_time
                obs[base + 8]  = min(makespan / max(cfg.service_end, 1), 1.0)
                obs[base + 9]  = min(pu_slacks,        default=1.0) if pu_slacks else 1.0
                obs[base + 10] = float(np.mean(wait_qualities)) if wait_qualities else 1.0
                obs[base + 11] = max(urgencies,         default=0.0) if urgencies else 0.0

    req_base = MAX_VEHICLES * veh_feats
    px, py = norm_coord(request.pickup_node)
    dx, dy = norm_coord(request.dropoff_node)
    obs[req_base + 0] = px
    obs[req_base + 1] = py
    obs[req_base + 2] = dx
    obs[req_base + 3] = dy
    obs[req_base + 4] = min((request.direct_time or 0) / 30.0, 1.0)
    obs[req_base + 5] = 0.0

    g_base = req_base + OBS_REQUEST
    obs[g_base + 0] = current_time / max(cfg.service_end, 1)
    n_busy = sum(1 for v in vehicles.values()
                 if v.plan or v.in_transit_stop is not None)
    obs[g_base + 1] = n_busy / max(len(vehicles), 1)
    obs[g_base + 2] = 0.0  # served fraction — not tracked in live dispatch
    obs[g_base + 3] = (congestion_factor(current_time) - 0.5) * 2

    if USE_ANTICIPATORY_FEATURES:
        current_rate = arrival_rate(current_time, cfg)
        base_rate    = getattr(cfg, 'inter_arrival', 3.0)
        obs[g_base + 4] = min((base_rate / max(current_rate, 0.1)) / 2.5, 1.0)

        future_t    = min(current_time + 30, cfg.service_end)
        future_rate = arrival_rate(future_t, cfg)
        obs[g_base + 5] = min((base_rate / max(future_rate, 0.1)) / 2.5, 1.0)

        total_onboard  = sum(len(v.onboard) for v in vehicles.values())
        total_capacity = sum(v.capacity for v in vehicles.values())
        obs[g_base + 6] = 1.0 - (total_onboard / max(total_capacity, 1))

        if metrics is not None:
            s = metrics.summary()
            n_rejected = s.get("rejected", 0)
            n_total    = s.get("total_requests", 1)
            obs[g_base + 7] = min(n_rejected / max(n_total, 1), 1.0)
        else:
            obs[g_base + 7] = 0.0

    return obs


# ===================================================================
# Debug helper
# ===================================================================

def print_plans(vehicles: dict[str, Vehicle]) -> None:
    for vid, vehicle in vehicles.items():
        prefix = ""
        if vehicle.in_transit_stop is not None:
            s = vehicle.in_transit_stop
            prefix = f"[IN-TRANSIT: {s.kind} {s.req_id}] "
        summary = [(s.kind, s.req_id) for s in vehicle.plan]
        print(f"   {vid}: {prefix}{summary}")