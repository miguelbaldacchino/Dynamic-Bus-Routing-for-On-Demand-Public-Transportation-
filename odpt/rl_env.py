#!/usr/bin/env python3
# rl_env.py
# Gymnasium environment for Dynamic DARP dispatch.
#
# Hierarchical decision: the agent selects a VEHICLE (or reject).
# Greedy best-position insertion handles where in the plan to put
# the PU/DO pair.  This is a standalone RL policy — greedy is used
# only for within-vehicle positioning, not for vehicle selection.
#
# Action masking: infeasible vehicles (no valid insertion exists)
# are masked out.  The agent can only pick feasible vehicles.
# Constraint satisfaction is guaranteed by check_feasibility().
#
# Designed for sb3-contrib MaskablePPO.
#
# Matches documented config:
#   6 buses, 16 capacity, depot node 40, 300 requests, Malta profile.

from __future__ import annotations

import random
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import SimulationConfig, arrival_rate
from models import Request, Stop, Vehicle
from feasibility import check_feasibility, evaluate_plan
from malta_travel import DEFAULT_COORDS, make_travel_fn


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------
MAX_VEHICLES = 8          # padded (fleet <= 6, room for sensitivity)
OBS_PER_VEHICLE = 8       # see _encode_state
OBS_REQUEST = 6

# Anticipatory features flag — overridden per-run by benchmark.py execute_run().
# When False: 4 global features (74 dims total) — baseline models (rl_v4 etc.)
# When True:  8 global features (78 dims total) — rl_v3ant only.
USE_ANTICIPATORY_FEATURES = False


def get_obs_size() -> int:
    """Compute obs size dynamically so benchmark per-run flag overrides work."""
    global_feats = 8 if USE_ANTICIPATORY_FEATURES else 4
    return MAX_VEHICLES * OBS_PER_VEHICLE + OBS_REQUEST + global_feats


class DARPEnv(gym.Env):
    """
    Gymnasium environment for online DARP dispatch.

    At each step a new request has arrived.  The agent selects one of
    K vehicles to assign it to, or action 0 to reject.  For the chosen
    vehicle, greedy best-position insertion finds the optimal (PU, DO)
    positions within that vehicle's plan.

    The action space is Discrete(K+1) where K = fleet_size.
    Action 0 = reject.  Actions 1..K = assign to vehicle 1..K.

    Parameters
    ----------
    cfg : SimulationConfig
    reward_mode : str
        "composite" (default) - acceptance bonus + cost + wait + ride
        "cost" - pure insertion cost
        "wait" - wait-time focused
    w_acceptance : float
        Reward for accepting any request (default 1.0).
        Higher = agent more aggressively accepts requests.
    w_wait : float
        Penalty weight on estimated wait time (default 2.0).
        Higher = agent prioritises reducing passenger wait.
    w_ride : float
        Linear penalty weight on estimated ride time (default 1.0).
        Higher = agent prioritises reducing in-vehicle time.
    w_ride_sq : float
        Quadratic penalty weight on estimated ride time (default 0.5).
        Acts as a p95_ride proxy — penalises outlier ride times
        disproportionately. A passenger at 90% of max_ride gets
        0.81*w_ride_sq penalty vs 0.25*w_ride_sq at 50% of max_ride.
    w_detour : float
        Penalty weight on excess detour ratio (default 0.5).
        Penalises est_ride / direct_time above 1.0x.
        e.g. a 3x detour gets 2.0*w_detour penalty.
        Zero penalty when est_ride <= direct_time (no detour).
    w_cost : float
        Penalty weight on insertion cost delta (default 0.5).
        Higher = agent prioritises operational efficiency.
    w_rejection : float
        Penalty applied when a request is rejected (default 5.0).
        Higher = agent more reluctant to reject any request.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: SimulationConfig = None,
        reward_mode: str = "composite",
        w_acceptance: float = 1.0,
        w_wait: float = 2.0,
        w_ride: float = 1.0,
        w_ride_sq: float = 0.5,
        w_detour: float = 0.5,
        w_cost: float = 0.5,
        w_rejection: float = 5.0,
    ):
        super().__init__()

        self.cfg         = cfg or SimulationConfig()
        self.reward_mode = reward_mode

        # Reward weights — searchable by Optuna, defaults match original
        self.w_acceptance = w_acceptance
        self.w_wait       = w_wait
        self.w_ride       = w_ride
        self.w_ride_sq    = w_ride_sq
        self.w_detour     = w_detour
        self.w_cost       = w_cost
        self.w_rejection  = w_rejection

        self.n_actions = self.cfg.fleet_size + 1  # 0=reject, 1..K=vehicles
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(get_obs_size(),), dtype=np.float32,
        )

        # Internal state
        self._travel_fn = None
        self._vehicles: dict[str, Vehicle] = {}
        self._vehicle_ids: list[str] = []   # ordered, stable
        self._system_state: dict = {}
        self._direct_times: dict = {}
        self._requests: dict = {}
        self._request_queue: list[Request] = []
        self._current_request: Optional[Request] = None
        self._sim_time: float = 0.0
        self._step_count: int = 0
        self._rng = random.Random(self.cfg.seed)

        # Per-step: best insertion for each vehicle (computed in _build_mask)
        # Maps vehicle_id -> (candidate_plan, n_committed, cost)
        self._vehicle_insertions: dict[str, tuple] = {}
        self._action_mask_array: np.ndarray = np.zeros(self.n_actions, dtype=np.int8)

        # Anticipatory features — tracked per-episode
        self._n_rejections: int = 0

    # ------------------------------------------------------------------
    # sb3-contrib MaskablePPO interface
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return current action mask. Called by MaskablePPO."""
        return self._action_mask_array.copy()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random(self.cfg.seed)

        self._travel_fn = make_travel_fn(DEFAULT_COORDS)
        self._direct_times = {}
        self._requests = {}
        self._step_count = 0
        self._sim_time = 0.0
        self._n_rejections = 0

        self._vehicles = {}
        self._vehicle_ids = []
        for k in range(self.cfg.fleet_size):
            vid = f"Bus-{k+1}"
            self._vehicles[vid] = Vehicle(
                id=vid,
                capacity=self.cfg.vehicle_capacity,
                location=self.cfg.depot_node,
            )
            self._vehicle_ids.append(vid)

        self._system_state = {
            "travel_time":      self._travel_fn,
            "ride_factor":      self.cfg.ride_factor,
            "direct_times":     self._direct_times,
            "coords":           DEFAULT_COORDS,
            "max_wait":         self.cfg.max_wait,
            "ride_time_margin": self.cfg.ride_time_margin,
        }

        self._request_queue = self._generate_requests()

        if self._request_queue:
            self._current_request = self._request_queue.pop(0)
            self._sim_time = self._current_request.request_time
            self._advance_vehicles_to(self._sim_time)
        else:
            self._current_request = None

        obs = self._encode_state()
        self._build_action_mask()
        info = {"action_mask": self._action_mask_array.copy()}
        return obs, info

    def step(self, action: int):
        assert self._current_request is not None

        req = self._current_request
        reward = 0.0
        info = {}

        if action == 0:
            # REJECT
            req.status = "REJECTED"
            reward = self._rejection_penalty(req)
            info["rejected"] = True
            self._n_rejections += 1
        else:
            # ASSIGN to vehicle (1-indexed action -> vehicle_ids[action-1])
            vid = self._vehicle_ids[action - 1]

            if vid not in self._vehicle_insertions:
                # Masked action was somehow selected - treat as reject
                req.status = "REJECTED"
                reward = self._rejection_penalty(req)
                info["rejected"] = True
            else:
                candidate_plan, n_committed, new_cost = self._vehicle_insertions[vid]
                vehicle = self._vehicles[vid]

                # Compute old cost for reward
                v_state = vehicle.to_state_dict(self._sim_time)
                old_cost = evaluate_plan(
                    v_state["plan_snapshot"], v_state,
                    self._system_state, self.cfg.weights,
                )

                # Apply insertion
                vehicle.plan = candidate_plan[n_committed:]
                req.status = "ASSIGNED"
                req.assignment_time = self._sim_time

                # Compute reward with wait and ride estimates
                est_wait = self._estimate_wait(req, candidate_plan, v_state)
                est_ride = self._estimate_ride(req, candidate_plan, v_state)
                reward = self._compute_reward(
                    req, old_cost, new_cost, est_wait, est_ride,
                )
                info["rejected"] = False
                info["vehicle"] = vid
                info["insertion_cost"] = new_cost - old_cost
                info["est_wait"] = est_wait

        self._requests[req.id] = req
        self._step_count += 1

        # Advance to next request
        terminated = False
        if self._request_queue:
            self._current_request = self._request_queue.pop(0)
            self._sim_time = self._current_request.request_time
            self._advance_vehicles_to(self._sim_time)
        else:
            self._current_request = None
            terminated = True

        obs = self._encode_state()
        self._build_action_mask()
        info["action_mask"] = self._action_mask_array.copy()
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Action mask: find best insertion per vehicle
    # ------------------------------------------------------------------

    def _build_action_mask(self):
        """
        For each vehicle, find the greedy best-position insertion.
        If a feasible insertion exists, the vehicle is unmasked.
        Store the best plan for use when the agent picks that vehicle.
        """
        self._action_mask_array = np.zeros(self.n_actions, dtype=np.int8)
        self._vehicle_insertions = {}

        # Reject always available
        self._action_mask_array[0] = 1

        if self._current_request is None:
            return

        req = self._current_request
        max_wait = self._system_state.get("max_wait", float("inf"))

        for idx, vid in enumerate(self._vehicle_ids):
            vehicle = self._vehicles[vid]
            v_state = vehicle.to_state_dict(self._sim_time)
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
                        node=req.pickup_node, kind="PU", req_id=req.id,
                        earliest=req.earliest,
                        latest=req.request_time + max_wait,
                        service=1.0, request_time=req.request_time,
                    )
                    do = Stop(
                        node=req.dropoff_node, kind="DO", req_id=req.id,
                        earliest=None, latest=None,
                        service=1.0, request_time=req.request_time,
                    )
                    candidate_tail.insert(i, pu)
                    candidate_tail.insert(j, do)
                    candidate = full_plan[:n_committed] + candidate_tail

                    if not check_feasibility(candidate, v_state, self._system_state):
                        continue

                    cost = evaluate_plan(
                        candidate, v_state, self._system_state, self.cfg.weights,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = candidate

            if best_candidate is not None:
                action_idx = idx + 1  # 1-indexed
                self._action_mask_array[action_idx] = 1
                self._vehicle_insertions[vid] = (
                    best_candidate, n_committed, best_cost,
                )

    # ------------------------------------------------------------------
    # Wait / ride estimation (for reward shaping)
    # ------------------------------------------------------------------

    def _estimate_wait(self, req, plan, v_state) -> float:
        current_node = v_state["location"]
        current_time = v_state["time"]
        travel_fn = self._system_state["travel_time"]

        for stop in plan:
            current_time += travel_fn(current_node, stop.node, current_time)
            if stop.kind == "PU" and stop.earliest and current_time < stop.earliest:
                current_time = stop.earliest
            if stop.kind == "PU" and stop.req_id == req.id:
                return max(0.0, current_time - req.request_time)
            current_time += stop.service
            current_node = stop.node
        return self.cfg.max_wait

    def _estimate_ride(self, req, plan, v_state) -> float:
        current_node = v_state["location"]
        current_time = v_state["time"]
        travel_fn = self._system_state["travel_time"]
        pu_time = None

        for stop in plan:
            current_time += travel_fn(current_node, stop.node, current_time)
            if stop.kind == "PU" and stop.earliest and current_time < stop.earliest:
                current_time = stop.earliest
            if stop.kind == "PU" and stop.req_id == req.id:
                pu_time = current_time
            if stop.kind == "DO" and stop.req_id == req.id and pu_time is not None:
                return current_time - pu_time
            current_time += stop.service
            current_node = stop.node
        return (req.direct_time or 5.0) * self.cfg.ride_factor

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, req, old_cost, new_cost, est_wait, est_ride) -> float:
        """
        Composite reward targeting all thesis metrics:
          +1.0  acceptance bonus (service rate)
          -w    wait penalty (wait time metric)
          -r    ride penalty (ride time metric)
          -c    cost delta penalty (distance/efficiency)
        """
        cost_delta = new_cost - old_cost

        if self.reward_mode == "cost":
            norm = max(req.direct_time or 1.0, 1.0) * 5
            return -cost_delta / norm

        elif self.reward_mode == "wait":
            return self.w_acceptance - self.w_wait * (est_wait / self.cfg.max_wait)

        else:  # composite
            direct   = max(req.direct_time or 1.0, 1.0)
            max_ride = direct * self.cfg.ride_factor
            norm     = direct * 5

            # Normalised ride ratio in [0, 1] — used for both ride terms
            ride_ratio = est_ride / max(max_ride, 1.0)

            # Wait penalty — linear, normalised by max_wait
            wait_pen = -self.w_wait * (est_wait / self.cfg.max_wait)

            # Ride penalty — linear term penalises all long rides
            ride_pen = -self.w_ride * ride_ratio

            # Ride penalty — quadratic term penalises outliers harder.
            # Proxy for p95_ride: a ride at 90% of max gets 0.81*w_ride_sq
            # vs 0.25*w_ride_sq at 50% of max. Suppresses the tail.
            ride_sq_pen = -self.w_ride_sq * (ride_ratio ** 2)

            # Detour penalty — penalises excess routing above direct time.
            # Only activates when est_ride > direct_time (actual detour).
            # A 3x detour on a 5-min trip gets 2.0*w_detour penalty.
            detour      = est_ride / direct
            detour_pen  = -self.w_detour * max(detour - 1.0, 0.0)

            # Cost penalty — insertion efficiency
            cost_pen = -self.w_cost * (cost_delta / norm)

            return (self.w_acceptance
                    + wait_pen
                    + ride_pen
                    + ride_sq_pen
                    + detour_pen
                    + cost_pen)

    def _rejection_penalty(self, req) -> float:
        return -self.w_rejection

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _encode_state(self) -> np.ndarray:
        obs = np.zeros(get_obs_size(), dtype=np.float32)

        lon_min, lon_max = 14.35, 14.55
        lat_min, lat_max = 35.85, 35.95

        def norm_coord(node_id):
            if node_id in DEFAULT_COORDS:
                lon, lat = DEFAULT_COORDS[node_id]
                x = (lon - lon_min) / (lon_max - lon_min) * 2 - 1
                y = (lat - lat_min) / (lat_max - lat_min) * 2 - 1
                return np.clip(x, -1, 1), np.clip(y, -1, 1)
            return 0.0, 0.0

        for i, vid in enumerate(self._vehicle_ids[:MAX_VEHICLES]):
            v = self._vehicles[vid]
            base = i * OBS_PER_VEHICLE
            x, y = norm_coord(v.location)
            obs[base + 0] = x
            obs[base + 1] = y
            obs[base + 2] = len(v.onboard) / max(v.capacity, 1)
            obs[base + 3] = min(len(v.plan) / 20.0, 1.0)
            obs[base + 4] = 1.0 if not v.plan and v.in_transit_stop is None else 0.0
            if v.plan:
                t_next = self._travel_fn(v.location, v.plan[0].node, self._sim_time)
                obs[base + 5] = min(t_next / 30.0, 1.0)
            if self._current_request is not None:
                req = self._current_request
                d_pu = self._travel_fn(v.location, req.pickup_node, self._sim_time)
                d_do = self._travel_fn(v.location, req.dropoff_node, self._sim_time)
                obs[base + 6] = min(d_pu / 30.0, 1.0)
                obs[base + 7] = min(d_do / 30.0, 1.0)

        req_base = MAX_VEHICLES * OBS_PER_VEHICLE
        if self._current_request is not None:
            req = self._current_request
            px, py = norm_coord(req.pickup_node)
            dx, dy = norm_coord(req.dropoff_node)
            obs[req_base + 0] = px
            obs[req_base + 1] = py
            obs[req_base + 2] = dx
            obs[req_base + 3] = dy
            obs[req_base + 4] = min((req.direct_time or 0) / 30.0, 1.0)
            obs[req_base + 5] = 0.0

        g_base = req_base + OBS_REQUEST

        # --- Original global features (4) ---
        obs[g_base + 0] = self._sim_time / max(self.cfg.service_end, 1)
        n_busy = sum(1 for v in self._vehicles.values()
                     if v.plan or v.in_transit_stop is not None)
        obs[g_base + 1] = n_busy / max(len(self._vehicles), 1)
        n_served = sum(1 for r in self._requests.values() if r.status == "COMPLETED")
        obs[g_base + 2] = n_served / max(self.cfg.n_requests, 1)
        from malta_travel import congestion_factor
        obs[g_base + 3] = (congestion_factor(self._sim_time) - 0.5) * 2

        # --- Anticipatory features (4 new — gated by USE_ANTICIPATORY_FEATURES) ---
        if USE_ANTICIPATORY_FEATURES:
            # Feature 5: Current demand intensity (higher = busier)
            current_rate = arrival_rate(self._sim_time, self.cfg)
            base_rate    = getattr(self.cfg, 'inter_arrival', 3.0)
            obs[g_base + 4] = min((base_rate / max(current_rate, 0.1)) / 2.5, 1.0)

            # Feature 6: Demand lookahead — intensity 30 min from now
            future_t    = min(self._sim_time + 30, self.cfg.service_end)
            future_rate = arrival_rate(future_t, self.cfg)
            obs[g_base + 5] = min((base_rate / max(future_rate, 0.1)) / 2.5, 1.0)

            # Feature 7: Fleet spare capacity (total free seats / total seats)
            total_onboard  = sum(len(v.onboard) for v in self._vehicles.values())
            total_capacity = sum(v.capacity for v in self._vehicles.values())
            obs[g_base + 6] = 1.0 - (total_onboard / max(total_capacity, 1))

            # Feature 8: Recent rejection pressure (rejections / decisions so far)
            obs[g_base + 7] = min(
                self._n_rejections / max(self._step_count, 1), 1.0
            )

        return obs

    # ------------------------------------------------------------------
    # Request generation
    # ------------------------------------------------------------------

    def _generate_requests(self) -> list[Request]:
        requests = []
        t = 0.0
        for i in range(1, self.cfg.n_requests + 1):
            mean_gap = arrival_rate(t, self.cfg)
            if self.cfg.stochastic_arrivals:
                gap = self._rng.expovariate(1.0 / mean_gap)
            else:
                gap = mean_gap
            t += gap
            if t >= self.cfg.service_end:
                break
            pu = self._rng.randint(1, self.cfg.n_nodes)
            do = self._rng.randint(1, self.cfg.n_nodes)
            while do == pu:
                do = self._rng.randint(1, self.cfg.n_nodes)
            req = Request(id=f"R{i}", pickup_node=pu, dropoff_node=do,
                          earliest=t, request_time=t)
            req.direct_time = self._travel_fn(pu, do, t)
            self._direct_times[req.id] = req.direct_time
            requests.append(req)
        return requests

    # ------------------------------------------------------------------
    # Vehicle advancement
    # ------------------------------------------------------------------

    def _advance_vehicles_to(self, target_time: float):
        for vid, vehicle in self._vehicles.items():
            v_time = getattr(vehicle, '_local_time', 0.0)
            while vehicle.plan and v_time < target_time:
                stop = vehicle.plan[0]
                travel = self._travel_fn(vehicle.location, stop.node, v_time)
                arrival = v_time + travel
                if arrival > target_time:
                    vehicle.in_transit_stop = stop
                    vehicle.in_transit_depart_time = v_time
                    vehicle.in_transit_eta = arrival
                    break
                v_time = arrival
                vehicle.location = stop.node
                vehicle.in_transit_stop = None
                vehicle.in_transit_depart_time = None
                vehicle.in_transit_eta = None
                if stop.kind == "PU" and stop.earliest and v_time < stop.earliest:
                    v_time = stop.earliest
                if stop.kind == "PU":
                    vehicle.onboard.add(stop.req_id)
                    vehicle.onboard_pickup_times[stop.req_id] = v_time
                    if stop.req_id in self._requests:
                        self._requests[stop.req_id].pickup_time = v_time
                        self._requests[stop.req_id].status = "ONBOARD"
                elif stop.kind == "DO":
                    vehicle.onboard.discard(stop.req_id)
                    vehicle.onboard_pickup_times.pop(stop.req_id, None)
                    if stop.req_id in self._requests:
                        self._requests[stop.req_id].dropoff_time = v_time
                        self._requests[stop.req_id].status = "COMPLETED"
                v_time += stop.service
                vehicle.plan.pop(0)
            vehicle._local_time = v_time

    # ------------------------------------------------------------------
    # Episode summary
    # ------------------------------------------------------------------

    def episode_summary(self) -> dict:
        served = sum(1 for r in self._requests.values() if r.status == "COMPLETED")
        rejected = sum(1 for r in self._requests.values() if r.status == "REJECTED")
        in_prog = sum(1 for r in self._requests.values()
                      if r.status in ("ASSIGNED", "ONBOARD"))
        total = len(self._requests)
        wait_times = [r.pickup_time - r.request_time
                      for r in self._requests.values() if r.pickup_time is not None]
        ride_times = [r.dropoff_time - r.pickup_time
                      for r in self._requests.values()
                      if r.pickup_time is not None and r.dropoff_time is not None]
        return {
            "total": total, "served": served, "rejected": rejected,
            "in_progress": in_prog,
            "service_rate": served / max(total, 1),
            "mean_wait": float(np.mean(wait_times)) if wait_times else None,
            "mean_ride": float(np.mean(ride_times)) if ride_times else None,
            "p95_wait": float(np.percentile(wait_times, 95)) if len(wait_times) > 1 else None,
            "p95_ride": float(np.percentile(ride_times, 95)) if len(ride_times) > 1 else None,
        }