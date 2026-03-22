# rl_dispatcher.py
# Drop-in RL dispatch policy for the live SimPy simulation.
#
# This module provides rl_insert() which has the same signature as
# greedy_insert() in dispatcher.py, so it can be swapped in with
# minimal changes to main.py.
#
# The RL policy uses a pre-trained PPO agent to select the insertion
# action.  It still relies on the shared feasibility checker —
# all actions are masked to feasible insertions only.
#
# Usage in main.py:
#   from rl_dispatcher import build_rl_policy, rl_insert
#   build_rl_policy("rl_outputs/ppo_final.pt")
#   # Then in request_generator, replace greedy_insert with rl_insert
#
# For the thesis evaluation, you can also use this in hybrid mode:
#   inserted = rl_insert(req, vehicles, system_state, ...)
#   if inserted:
#       sa_improve(vehicles, system_state, ...)  # SA refinement on top

from __future__ import annotations

import numpy as np
from typing import Optional

from models import Stop, Request, Vehicle
from feasibility import check_feasibility, evaluate_plan
from malta_travel import DEFAULT_COORDS, congestion_factor
from rl_env import OBS_SIZE, MAX_ACTIONS, MAX_VEHICLES, OBS_PER_VEHICLE


# ---------------------------------------------------------------------------
# Module-level agent
# ---------------------------------------------------------------------------

_RL_AGENT = None
_DEVICE = "cpu"


def build_rl_policy(
    model_path: str,
    device: str = "cpu",
    hidden: int = 256,
) -> None:
    """Load a trained PPO agent for use in the live simulation."""
    global _RL_AGENT, _DEVICE
    from rl_agent import PPOAgent

    _DEVICE = device
    _RL_AGENT = PPOAgent(
        obs_dim=OBS_SIZE,
        act_dim=MAX_ACTIONS,
        hidden=hidden,
        device=device,
    )
    _RL_AGENT.load(model_path)
    print(f"RL policy loaded from {model_path}")


def _get_rl_agent():
    if _RL_AGENT is None:
        raise RuntimeError(
            "RL agent not initialised. Call build_rl_policy(path) first."
        )
    return _RL_AGENT


# ---------------------------------------------------------------------------
# State encoding (mirrors rl_env._encode_state but takes live data)
# ---------------------------------------------------------------------------

def _encode_live_state(
    request:      Request,
    vehicles:     dict[str, Vehicle],
    current_time: float,
    system_state: dict,
    cfg,
) -> np.ndarray:
    """
    Encode the current system state for the RL agent.
    Mirrors the encoding in DARPEnv._encode_state().
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

    # Vehicle features
    vids = sorted(vehicles.keys())
    for i, vid in enumerate(vids[:MAX_VEHICLES]):
        v = vehicles[vid]
        base = i * OBS_PER_VEHICLE
        x, y = norm_coord(v.location)
        obs[base + 0] = x
        obs[base + 1] = y
        obs[base + 2] = len(v.onboard) / max(v.capacity, 1)
        obs[base + 3] = min(len(v.plan) / 20.0, 1.0)
        obs[base + 4] = (
            1.0 if not v.plan and v.in_transit_stop is None else 0.0
        )
        if v.plan:
            t_next = travel_fn(v.location, v.plan[0].node, current_time)
            obs[base + 5] = min(t_next / 30.0, 1.0)

    # Request features
    req_base = MAX_VEHICLES * OBS_PER_VEHICLE
    px, py = norm_coord(request.pickup_node)
    dx, dy = norm_coord(request.dropoff_node)
    obs[req_base + 0] = px
    obs[req_base + 1] = py
    obs[req_base + 2] = dx
    obs[req_base + 3] = dy
    obs[req_base + 4] = min((request.direct_time or 0) / 30.0, 1.0)

    # Global features
    g_base = req_base + 5
    obs[g_base + 0] = current_time / max(cfg.service_end, 1)
    n_busy = sum(
        1 for v in vehicles.values()
        if v.plan or v.in_transit_stop is not None
    )
    obs[g_base + 1] = n_busy / max(len(vehicles), 1)
    obs[g_base + 2] = 0.0  # not easily available in live sim
    obs[g_base + 3] = 0.0
    obs[g_base + 4] = (congestion_factor(current_time) - 0.5) * 2

    return obs


# ---------------------------------------------------------------------------
# Action mask and insertion (mirrors rl_env._build_action_mask)
# ---------------------------------------------------------------------------

def _build_live_action_mask(
    request:      Request,
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
) -> tuple[np.ndarray, dict]:
    """
    Build action mask for the live simulation state.
    Returns (mask, action_map) where action_map[idx] = (vid, plan, n_committed).
    """
    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    action_map = {}

    mask[0] = 1  # reject always available
    max_wait = system_state.get("max_wait", float("inf"))
    action_idx = 1

    for vid, vehicle in vehicles.items():
        v_state = vehicle.to_state_dict(current_time)
        full_plan = v_state["plan_snapshot"]
        n_committed = 1 if vehicle.in_transit_stop is not None else 0
        insertable = full_plan[n_committed:]
        n = len(insertable)

        for i in range(n + 1):
            for j in range(i + 1, n + 2):
                if action_idx >= MAX_ACTIONS:
                    break

                candidate_tail = list(insertable)
                pu = Stop(
                    node=request.pickup_node,
                    kind="PU",
                    req_id=request.id,
                    earliest=request.earliest,
                    latest=request.request_time + max_wait,
                    service=1.0,
                    request_time=request.request_time,
                )
                do = Stop(
                    node=request.dropoff_node,
                    kind="DO",
                    req_id=request.id,
                    earliest=None,
                    latest=None,
                    service=1.0,
                    request_time=request.request_time,
                )
                candidate_tail.insert(i, pu)
                candidate_tail.insert(j, do)
                candidate = full_plan[:n_committed] + candidate_tail

                if check_feasibility(candidate, v_state, system_state):
                    mask[action_idx] = 1
                    action_map[action_idx] = (vid, candidate, n_committed)

                action_idx += 1

            if action_idx >= MAX_ACTIONS:
                break
        if action_idx >= MAX_ACTIONS:
            break

    return mask, action_map


# ---------------------------------------------------------------------------
# Public interface: rl_insert (drop-in for greedy_insert)
# ---------------------------------------------------------------------------

def rl_insert(
    request:      Request,
    vehicles:     dict[str, Vehicle],
    system_state: dict,
    current_time: float,
    weights:      tuple[float, float, float],
    metrics=None,
    deterministic: bool = True,
) -> bool:
    """
    Insert request using the trained RL policy.

    Same signature as greedy_insert() for drop-in compatibility.
    Returns True if inserted, False if rejected.
    """
    import time as _time
    t0 = _time.time()

    agent = _get_rl_agent()

    # Build observation and action mask
    from config import SimulationConfig
    cfg = SimulationConfig()  # for normalisation constants

    obs = _encode_live_state(
        request, vehicles, current_time, system_state, cfg,
    )
    mask, action_map = _build_live_action_mask(
        request, vehicles, system_state, current_time,
    )

    # Check if any feasible insertion exists
    if mask.sum() <= 1:  # only reject available
        elapsed = _time.time() - t0
        if metrics:
            metrics.log_decision_latency(elapsed)
            metrics.mark_rejected(request.id)
        print(f"  -> RL rejected {request.id} (no feasible insertion)")
        return False

    # Agent selects action
    action, log_prob, value = agent.select_action(
        obs, mask, deterministic=deterministic,
    )

    elapsed = _time.time() - t0
    if metrics:
        metrics.log_decision_latency(elapsed)

    if action == 0 or action not in action_map:
        if metrics:
            metrics.mark_rejected(request.id)
        print(f"  -> RL rejected {request.id} (agent chose reject)")
        return False

    # Apply insertion
    vid, candidate_plan, n_committed = action_map[action]
    vehicle = vehicles[vid]
    vehicle.plan = candidate_plan[n_committed:]

    if (vehicle.wake_event is not None
            and not vehicle.wake_event.triggered
            and vehicle.plan):
        vehicle.wake_event.succeed()

    # Compute cost for logging
    v_state = vehicle.to_state_dict(current_time)
    cost = evaluate_plan(candidate_plan, v_state, system_state, weights)

    print(f"  -> RL inserted {request.id} into {vid}  cost={cost:.2f}")
    return True
