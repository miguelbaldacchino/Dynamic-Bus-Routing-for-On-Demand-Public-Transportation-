# rl_env.py
# Gymnasium environment for the Dynamic Dial-a-Ride Problem.
#
# The environment wraps the DARP simulation and presents it as a
# standard RL interface.  At each step, a new request has arrived
# and the agent must choose a (vehicle, PU_position, DO_position)
# triple from the set of feasible insertions.
#
# Key design decisions:
# ---------------------
# 1. Action masking: The environment pre-computes ALL feasible
#    insertions and exposes them as a mask.  The agent can ONLY
#    select feasible actions, guaranteeing 100% constraint
#    satisfaction.  This eliminates the need for penalty shaping
#    or post-hoc repair.
#
# 2. Flat action space: We enumerate all (vehicle, i, j) triples
#    into a single discrete action space with a fixed maximum size.
#    Invalid/infeasible indices are masked out.
#
# 3. State representation: Fixed-size vector encoding vehicle states
#    (location, load, plan length, idle time), the new request
#    (pickup, dropoff, direct time), and global context (time of day,
#    fleet utilisation).
#
# 4. Reward: Negative insertion cost (same objective as the greedy
#    heuristic) with a large penalty for rejection.  This aligns
#    RL training with the thesis evaluation metrics.

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
# Constants
# ---------------------------------------------------------------------------

# Maximum number of enumerated actions.
# With 10 vehicles and plans up to ~20 stops each, the theoretical max
# is 10 * 21 * 22 / 2 ≈ 2310.  We pad to 3000 for safety.
# Action index 0 is always the REJECT action.
MAX_ACTIONS = 3000

# Observation vector size (see _encode_state for layout)
# Per-vehicle: 5 features x max_vehicles
# Request: 4 features
# Global: 4 features
MAX_VEHICLES = 12  # padded; actual fleet ≤ 10
OBS_PER_VEHICLE = 6
OBS_REQUEST = 5
OBS_GLOBAL = 5
OBS_SIZE = MAX_VEHICLES * OBS_PER_VEHICLE + OBS_REQUEST + OBS_GLOBAL


class DARPEnv(gym.Env):
    """
    Gymnasium environment for online DARP dispatch.

    Each episode simulates one service day.  At each step, the
    environment advances the simulation clock to the next request
    arrival and presents the agent with the current state and a
    mask of feasible insertion actions.

    The agent selects one action (an insertion or rejection), the
    environment applies it, optionally runs SA improvement, and
    returns the reward.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation parameters.  The RL policy field is ignored;
        this env always uses the agent's action.
    sa_improve_after : bool
        If True, run SA improvement after each RL insertion.
        This tests "RL insertion + SA refinement" as a hybrid.
    reward_mode : str
        "cost"    — reward = -insertion_cost (normalised)
        "wait"    — reward = -wait_time_estimate
        "composite" — weighted combination (default)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: SimulationConfig = None,
        sa_improve_after: bool = False,
        reward_mode: str = "composite",
    ):
        super().__init__()

        self.cfg = cfg or SimulationConfig()
        self.sa_improve_after = sa_improve_after
        self.reward_mode = reward_mode

        # Spaces
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32,
        )

        # Internal state (initialised in reset)
        self._travel_fn = None
        self._vehicles: dict[str, Vehicle] = {}
        self._system_state: dict = {}
        self._direct_times: dict = {}
        self._requests: dict = {}
        self._request_queue: list[Request] = []
        self._current_request: Optional[Request] = None
        self._sim_time: float = 0.0
        self._step_count: int = 0
        self._rng = random.Random(self.cfg.seed)

        # Action index -> (vehicle_id, pu_pos, do_pos) mapping
        # Rebuilt at each step based on current feasible insertions.
        self._action_map: dict[int, tuple] = {}
        self._action_mask: np.ndarray = np.zeros(MAX_ACTIONS, dtype=np.int8)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset for a new episode (one service day)."""
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random(self.cfg.seed)

        # Rebuild travel function
        self._travel_fn = make_travel_fn(DEFAULT_COORDS)
        self._direct_times = {}
        self._requests = {}
        self._step_count = 0
        self._sim_time = 0.0

        # Create fleet
        self._vehicles = {
            f"Bus-{k+1}": Vehicle(
                id=f"Bus-{k+1}",
                capacity=self.cfg.vehicle_capacity,
                location=self.cfg.depot_node,
            )
            for k in range(self.cfg.fleet_size)
        }

        # System state dict (shared with feasibility checker)
        self._system_state = {
            "travel_time":      self._travel_fn,
            "ride_factor":      self.cfg.ride_factor,
            "direct_times":     self._direct_times,
            "coords":           DEFAULT_COORDS,
            "max_wait":         self.cfg.max_wait,
            "ride_time_margin": self.cfg.ride_time_margin,
        }

        # Pre-generate the full request stream for this episode.
        # This gives us deterministic, reproducible episodes.
        self._request_queue = self._generate_requests()

        # Advance to first request
        if self._request_queue:
            self._current_request = self._request_queue.pop(0)
            self._sim_time = self._current_request.request_time
            self._advance_vehicles_to(self._sim_time)
        else:
            self._current_request = None

        obs = self._encode_state()
        self._build_action_mask()
        info = {"action_mask": self._action_mask.copy()}

        return obs, info

    def step(self, action: int):
        """
        Execute one dispatch decision.

        Parameters
        ----------
        action : int
            Index into the current action map.
            0 = reject the request.
            1..N = feasible insertion.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert self._current_request is not None, "No current request"

        req = self._current_request
        reward = 0.0
        info = {}

        if action == 0 or action not in self._action_map:
            # REJECT
            req.status = "REJECTED"
            reward = self._rejection_penalty(req)
            info["rejected"] = True
        else:
            # INSERT
            vid, candidate_plan, n_committed = self._action_map[action]
            vehicle = self._vehicles[vid]

            # Compute insertion cost for reward
            v_state = vehicle.to_state_dict(self._sim_time)
            old_cost = evaluate_plan(
                v_state["plan_snapshot"], v_state,
                self._system_state, self.cfg.weights,
            )
            new_cost = evaluate_plan(
                candidate_plan, v_state,
                self._system_state, self.cfg.weights,
            )

            # Apply the insertion
            vehicle.plan = candidate_plan[n_committed:]
            req.status = "ASSIGNED"
            req.assignment_time = self._sim_time

            reward = self._compute_reward(req, old_cost, new_cost)
            info["rejected"] = False
            info["insertion_cost"] = new_cost - old_cost

        self._requests[req.id] = req
        self._step_count += 1

        # Advance to next request
        terminated = False
        truncated = False

        if self._request_queue:
            self._current_request = self._request_queue.pop(0)
            self._sim_time = self._current_request.request_time
            self._advance_vehicles_to(self._sim_time)
        else:
            self._current_request = None
            terminated = True

        obs = self._encode_state()
        self._build_action_mask()
        info["action_mask"] = self._action_mask.copy()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Request generation (mirrors main.py logic, but pre-generated)
    # ------------------------------------------------------------------

    def _generate_requests(self) -> list[Request]:
        """Pre-generate the full request stream for one episode."""
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

            req = Request(
                id=f"R{i}",
                pickup_node=pu,
                dropoff_node=do,
                earliest=t,
                request_time=t,
            )
            req.direct_time = self._travel_fn(pu, do, t)
            self._direct_times[req.id] = req.direct_time
            requests.append(req)

        return requests

    # ------------------------------------------------------------------
    # Vehicle advancement (simplified execution model)
    # ------------------------------------------------------------------

    def _advance_vehicles_to(self, target_time: float):
        """
        Advance all vehicles' execution up to target_time.

        This is a simplified version of vehicle_process that executes
        deterministically (no stochastic noise) for training stability.
        Vehicles process stops in order, consuming travel + service time.
        """
        for vid, vehicle in self._vehicles.items():
            while vehicle.plan:
                stop = vehicle.plan[0]
                travel = self._travel_fn(
                    vehicle.location, stop.node, self._sim_time
                )

                # Time to complete this stop
                arrival = self._sim_time + travel  # simplified: uses current sim_time
                # Actually we need to track vehicle's own clock
                break  # simplified: we use a vehicle-local clock below

        # More accurate: track each vehicle's clock independently
        for vid, vehicle in self._vehicles.items():
            v_time = getattr(vehicle, '_local_time', 0.0)

            while vehicle.plan and v_time < target_time:
                stop = vehicle.plan[0]
                travel = self._travel_fn(vehicle.location, stop.node, v_time)
                arrival = v_time + travel

                if arrival > target_time:
                    # Vehicle is mid-travel; set in-transit state
                    vehicle.in_transit_stop = stop
                    vehicle.in_transit_depart_time = v_time
                    vehicle.in_transit_eta = arrival
                    break

                # Vehicle arrives at stop
                v_time = arrival
                vehicle.location = stop.node
                vehicle.in_transit_stop = None
                vehicle.in_transit_depart_time = None
                vehicle.in_transit_eta = None

                # Wait if early for pickup
                if stop.kind == "PU" and stop.earliest and v_time < stop.earliest:
                    v_time = stop.earliest

                # Process stop
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
    # State encoding
    # ------------------------------------------------------------------

    def _encode_state(self) -> np.ndarray:
        """
        Encode the current system state as a fixed-size float vector.

        Layout (all values normalised to [-1, 1] or [0, 1]):
        ┌──────────────────────────────────────────────┐
        │ Per-vehicle block (MAX_VEHICLES × 6):        │
        │   [0] location_x (normalised)                │
        │   [1] location_y (normalised)                │
        │   [2] load / capacity                        │
        │   [3] plan_length / 20 (clipped)             │
        │   [4] is_idle (0 or 1)                       │
        │   [5] time_to_next_stop / 30 (clipped)       │
        │                                              │
        │ Request block (5):                           │
        │   [0] pickup_x (normalised)                  │
        │   [1] pickup_y (normalised)                  │
        │   [2] dropoff_x (normalised)                 │
        │   [3] dropoff_y (normalised)                 │
        │   [4] direct_time / 30 (clipped)             │
        │                                              │
        │ Global block (5):                            │
        │   [0] time_of_day / service_end              │
        │   [1] fleet_utilisation (busy / total)       │
        │   [2] requests_served_so_far / n_requests    │
        │   [3] requests_rejected_so_far / n_requests  │
        │   [4] congestion_factor (0.65-1.0 scaled)    │
        └──────────────────────────────────────────────┘
        """
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # Coordinate normalisation bounds (Malta bounding box, approx)
        lon_min, lon_max = 14.35, 14.55
        lat_min, lat_max = 35.85, 35.95

        def norm_coord(node_id):
            if node_id in DEFAULT_COORDS:
                lon, lat = DEFAULT_COORDS[node_id]
                x = (lon - lon_min) / (lon_max - lon_min) * 2 - 1
                y = (lat - lat_min) / (lat_max - lat_min) * 2 - 1
                return np.clip(x, -1, 1), np.clip(y, -1, 1)
            return 0.0, 0.0

        # Encode vehicles
        vids = sorted(self._vehicles.keys())
        for i, vid in enumerate(vids[:MAX_VEHICLES]):
            v = self._vehicles[vid]
            base = i * OBS_PER_VEHICLE
            x, y = norm_coord(v.location)
            obs[base + 0] = x
            obs[base + 1] = y
            obs[base + 2] = len(v.onboard) / max(v.capacity, 1)
            obs[base + 3] = min(len(v.plan) / 20.0, 1.0)
            obs[base + 4] = 1.0 if not v.plan and v.in_transit_stop is None else 0.0

            # Time to next stop
            if v.plan:
                t_next = self._travel_fn(v.location, v.plan[0].node, self._sim_time)
                obs[base + 5] = min(t_next / 30.0, 1.0)
            else:
                obs[base + 5] = 0.0

        # Encode current request
        req_base = MAX_VEHICLES * OBS_PER_VEHICLE
        if self._current_request is not None:
            req = self._current_request
            px, py = norm_coord(req.pickup_node)
            dx, dy = norm_coord(req.dropoff_node)
            obs[req_base + 0] = px
            obs[req_base + 1] = py
            obs[req_base + 2] = dx
            obs[req_base + 3] = dy
            obs[req_base + 4] = min(
                (req.direct_time or 0) / 30.0, 1.0
            )

        # Global features
        g_base = req_base + OBS_REQUEST
        obs[g_base + 0] = self._sim_time / max(self.cfg.service_end, 1)
        n_busy = sum(
            1 for v in self._vehicles.values()
            if v.plan or v.in_transit_stop is not None
        )
        obs[g_base + 1] = n_busy / max(len(self._vehicles), 1)

        n_served = sum(
            1 for r in self._requests.values()
            if r.status == "COMPLETED"
        )
        n_rejected = sum(
            1 for r in self._requests.values()
            if r.status == "REJECTED"
        )
        total = max(len(self._requests) + 1, 1)
        obs[g_base + 2] = n_served / self.cfg.n_requests
        obs[g_base + 3] = n_rejected / self.cfg.n_requests

        from malta_travel import congestion_factor
        obs[g_base + 4] = (congestion_factor(self._sim_time) - 0.5) * 2

        return obs

    # ------------------------------------------------------------------
    # Action mask construction
    # ------------------------------------------------------------------

    def _build_action_mask(self):
        """
        Enumerate all feasible (vehicle, PU_pos, DO_pos) insertions
        for the current request, and build the action mask.

        Action 0 = reject (always available).
        Actions 1..N = feasible insertions.
        """
        self._action_mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        self._action_map = {}

        # Reject is always available
        self._action_mask[0] = 1

        if self._current_request is None:
            return

        req = self._current_request
        max_wait = self._system_state.get("max_wait", float("inf"))
        action_idx = 1

        for vid, vehicle in self._vehicles.items():
            v_state = vehicle.to_state_dict(self._sim_time)
            full_plan = v_state["plan_snapshot"]
            n_committed = 1 if vehicle.in_transit_stop is not None else 0
            insertable = full_plan[n_committed:]
            n = len(insertable)

            for i in range(n + 1):
                for j in range(i + 1, n + 2):
                    if action_idx >= MAX_ACTIONS:
                        break

                    # Build candidate plan
                    candidate_tail = list(insertable)
                    pu = Stop(
                        node=req.pickup_node,
                        kind="PU",
                        req_id=req.id,
                        earliest=req.earliest,
                        latest=req.request_time + max_wait,
                        service=1.0,
                        request_time=req.request_time,
                    )
                    do = Stop(
                        node=req.dropoff_node,
                        kind="DO",
                        req_id=req.id,
                        earliest=None,
                        latest=None,
                        service=1.0,
                        request_time=req.request_time,
                    )
                    candidate_tail.insert(i, pu)
                    candidate_tail.insert(j, do)
                    candidate = full_plan[:n_committed] + candidate_tail

                    if check_feasibility(candidate, v_state, self._system_state):
                        self._action_mask[action_idx] = 1
                        self._action_map[action_idx] = (
                            vid, candidate, n_committed,
                        )

                    action_idx += 1

                if action_idx >= MAX_ACTIONS:
                    break
            if action_idx >= MAX_ACTIONS:
                break

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _compute_reward(
        self, req: Request, old_cost: float, new_cost: float
    ) -> float:
        """Compute reward for a successful insertion."""
        cost_delta = new_cost - old_cost

        if self.reward_mode == "cost":
            # Normalise by direct time to make rewards comparable
            norm = max(req.direct_time or 1.0, 1.0) * 10
            return -cost_delta / norm

        elif self.reward_mode == "wait":
            # Estimate wait time penalty (direct proxy for passenger QoS)
            return -cost_delta / 100.0

        else:  # "composite" (default)
            # Small positive reward for acceptance + scaled cost
            acceptance_bonus = 0.5
            norm = max(req.direct_time or 1.0, 1.0) * 10
            return acceptance_bonus - cost_delta / norm

    def _rejection_penalty(self, req: Request) -> float:
        """Large negative reward for rejecting a request."""
        return -5.0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_action_mask(self) -> np.ndarray:
        """Return the current action mask (for external use by the agent)."""
        return self._action_mask.copy()

    def n_feasible_actions(self) -> int:
        """Number of currently feasible actions (including reject)."""
        return int(self._action_mask.sum())

    def episode_summary(self) -> dict:
        """Return end-of-episode statistics."""
        served = sum(
            1 for r in self._requests.values()
            if r.status == "COMPLETED"
        )
        rejected = sum(
            1 for r in self._requests.values()
            if r.status == "REJECTED"
        )
        assigned = sum(
            1 for r in self._requests.values()
            if r.status == "ASSIGNED" or r.status == "ONBOARD"
        )

        wait_times = [
            r.pickup_time - r.request_time
            for r in self._requests.values()
            if r.pickup_time is not None
        ]
        ride_times = [
            r.dropoff_time - r.pickup_time
            for r in self._requests.values()
            if r.pickup_time is not None and r.dropoff_time is not None
        ]

        return {
            "total": len(self._requests),
            "served": served,
            "rejected": rejected,
            "in_progress": assigned,
            "mean_wait": float(np.mean(wait_times)) if wait_times else None,
            "mean_ride": float(np.mean(ride_times)) if ride_times else None,
            "service_rate": served / max(len(self._requests), 1),
        }
