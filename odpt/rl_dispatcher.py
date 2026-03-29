#!/usr/bin/env python3
# rl_dispatcher.py
# Drop-in RL dispatch policy for the live SimPy simulation.
#
# Loads a trained MaskablePPO model and uses it for vehicle assignment.
# Greedy best-position insertion handles within-vehicle routing.
#
# Usage in main.py:
#   from rl_dispatcher import build_rl_policy, rl_insert
#   build_rl_policy("rl_outputs/run_001/model.zip")
#
#   # In request_generator, replace greedy_insert() with:
#   inserted = rl_insert(req, vehicles, system_state, current_time, weights, metrics)

from __future__ import annotations

import time as _time
import numpy as np
from typing import Optional

from models import Stop, Request, Vehicle
from feasibility import check_feasibility, evaluate_plan
from malta_travel import DEFAULT_COORDS, congestion_factor
from config import arrival_rate
from rl_env import (OBS_SIZE, OBS_PER_VEHICLE, OBS_REQUEST, OBS_GLOBAL,
                    MAX_VEHICLES, USE_ANTICIPATORY_FEATURES)


# ---------------------------------------------------------------------------
# Module-level model
# ---------------------------------------------------------------------------

_MODEL = None


def build_rl_policy(model_path: str, device: str = "auto") -> None:
    """Load a trained MaskablePPO model for dispatch."""
    global _MODEL
    from sb3_contrib import MaskablePPO
    _MODEL = MaskablePPO.load(model_path, device=device)
    print(f"RL policy loaded: {model_path}")


def _get_model():
    if _MODEL is None:
        raise RuntimeError("RL model not loaded. Call build_rl_policy(path) first.")
    return _MODEL


# ---------------------------------------------------------------------------
# State encoding (mirrors rl_env._encode_state)
# ---------------------------------------------------------------------------

def _encode_live_state(
    request: Request,
    vehicles: dict[str, Vehicle],
    vehicle_ids: list[str],
    current_time: float,
    system_state: dict,
    cfg,
    metrics=None,
) -> np.ndarray:
    """
    Encode live SimPy state into the same observation format as DARPEnv.
    Must mirror rl_env._encode_state exactly (78 dims with v2 features).
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

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

    for i, vid in enumerate(vehicle_ids[:MAX_VEHICLES]):
        v = vehicles[vid]
        base = i * OBS_PER_VEHICLE
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

    req_base = MAX_VEHICLES * OBS_PER_VEHICLE
    px, py = norm_coord(request.pickup_node)
    dx, dy = norm_coord(request.dropoff_node)
    obs[req_base + 0] = px
    obs[req_base + 1] = py
    obs[req_base + 2] = dx
    obs[req_base + 3] = dy
    obs[req_base + 4] = min((request.direct_time or 0) / 30.0, 1.0)
    obs[req_base + 5] = 0.0

    # --- Global features (8 total — must match rl_env) ---
    g_base = req_base + OBS_REQUEST
    obs[g_base + 0] = current_time / max(cfg.service_end, 1)
    n_busy = sum(1 for v in vehicles.values()
                 if v.plan or v.in_transit_stop is not None)
    obs[g_base + 1] = n_busy / max(len(vehicles), 1)
    obs[g_base + 2] = 0.0  # served fraction — not tracked in live dispatch
    obs[g_base + 3] = (congestion_factor(current_time) - 0.5) * 2

    # --- Anticipatory features (gated by USE_ANTICIPATORY_FEATURES) ---
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


# ---------------------------------------------------------------------------
# Best-position insertion per vehicle (same as greedy)
# ---------------------------------------------------------------------------

def _find_best_insertion(
    request: Request,
    vehicle: Vehicle,
    current_time: float,
    system_state: dict,
    weights: tuple,
) -> Optional[tuple]:
    """
    Find the best feasible insertion for request into vehicle.
    Returns (candidate_plan, n_committed, cost) or None.
    """
    v_state = vehicle.to_state_dict(current_time)
    full_plan = v_state["plan_snapshot"]
    n_committed = 1 if vehicle.in_transit_stop is not None else 0
    insertable = full_plan[n_committed:]
    n = len(insertable)
    max_wait = system_state.get("max_wait", float("inf"))

    best_cost = float("inf")
    best_candidate = None

    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            candidate_tail = list(insertable)
            pu = Stop(
                node=request.pickup_node, kind="PU", req_id=request.id,
                earliest=request.earliest,
                latest=request.request_time + max_wait,
                service=1.0, request_time=request.request_time,
            )
            do = Stop(
                node=request.dropoff_node, kind="DO", req_id=request.id,
                earliest=None, latest=None,
                service=1.0, request_time=request.request_time,
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


# ---------------------------------------------------------------------------
# Public interface: rl_insert (drop-in for greedy_insert)
# ---------------------------------------------------------------------------

def rl_insert(
    request: Request,
    vehicles: dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights: tuple[float, float, float],
    metrics=None,
) -> bool:
    """
    Insert request using the trained RL policy for vehicle assignment,
    with greedy best-position for within-vehicle insertion.

    Same signature as greedy_insert() for drop-in use.
    Returns True if inserted, False if rejected.
    """
    t0 = _time.time()
    model = _get_model()

    vehicle_ids = sorted(vehicles.keys())
    n_vehicles = len(vehicle_ids)

    # Find best insertion per vehicle
    n_actions = n_vehicles + 1
    mask = np.zeros(n_actions, dtype=np.int8)
    mask[0] = 1  # reject always available
    insertions = {}

    for idx, vid in enumerate(vehicle_ids):
        result = _find_best_insertion(
            request, vehicles[vid], current_time, system_state, weights,
        )
        if result is not None:
            mask[idx + 1] = 1
            insertions[vid] = result

    # If no vehicle can take this request, reject
    if mask.sum() <= 1:
        elapsed = _time.time() - t0
        if metrics:
            metrics.log_decision_latency(elapsed)
            metrics.mark_rejected(request.id)
        print(f"  -> RL rejected {request.id} (no feasible vehicle)")
        return False

    # Encode state and get RL action
    from config import SimulationConfig
    cfg = SimulationConfig()
    obs = _encode_live_state(
        request, vehicles, vehicle_ids, current_time, system_state, cfg,
        metrics,
    )
    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
    action = int(action)

    elapsed = _time.time() - t0
    if metrics:
        metrics.log_decision_latency(elapsed)

    if action == 0 or action > n_vehicles:
        if metrics:
            metrics.mark_rejected(request.id)
        print(f"  -> RL rejected {request.id} (agent chose reject)")
        return False

    # Apply the insertion for the chosen vehicle
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