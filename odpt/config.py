# config.py
# All time values are in MINUTES.
# Simulation time 0 = 07:00.
# service_end = 660 = 18:00 (last request accepted).
# horizon = 840 = 21:00 (sim clock stops; vehicles finish plans).
#
# Fixes applied
# -------------
# - horizon corrected: 660 min (was 800, which gave 20:20 not 18:00)
# - stochastic_arrivals flag: enables exponential inter-arrival times
# - SA time limit is now per-vehicle, not shared across fleet
# - SA iterations tuned down (vehicles have small plans; fewer wasted iters)

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # ---- Reproducibility ----
    seed:             int   = 42

    # ---- Simulation horizon (minutes, 0=07:00) ----
    service_end:      float = 660.0      # 18:00 — stop accepting new requests
    horizon:          float = 840.0      # 21:00 — sim ends; 3h buffer for completion

    # ---- Demand ----
    n_requests:       int   = 220
    inter_arrival:    float = 3.0        # minutes between requests (baseline mean)
    demand_profile:   str   = "bimodal"  # "uniform" | "peak" | "bimodal"
    stochastic_arrivals: bool = True     # True = Poisson (exponential gaps)
    n_nodes:          int   = 15         # nodes 1-15 available (0 = depot)

    # ---- Fleet ----
    fleet_size:       int   = 8
    vehicle_capacity: int   = 12
    depot_node:       int   = 0

    # ---- DARP constraints (minutes) ----
    ride_factor:      float = 2.5    # max ride = 2.5 x direct travel time
    max_wait:         float = 30.0   # latest pickup = request_time + 30 min
    ride_time_margin: float = 0.0    # minutes subtracted from max ride in planning
                                     # absorbs timing drift from congestion transitions

    # ---- Objective weights (alpha, beta, gamma) ----
    weights:          tuple = (1.0, 2.0, 2.5)  # distance, wait_time, ride_time

    # ---- Policy ----
    policy:           str   = "greedy+sa"

    # ---- SA hyperparameters ----
    sa_initial_temp:  float = 5_000.0
    sa_cooling_rate:  float = 0.995
    sa_iterations:    int   = 5_000      # per vehicle (was 20k shared)
    sa_time_limit:    float = 0.3        # seconds per vehicle (was 1.0 shared)


def arrival_rate(t: float, cfg: SimulationConfig) -> float:
    """
    Mean inter-arrival time (minutes) at simulation time t.

    When cfg.stochastic_arrivals is True, the caller should sample
    from Exponential(1/rate) rather than using this value directly.

    Bimodal (default):
      t=0-120    07:00-09:00  morning peak      interval / 2.0
      t=120-480  09:00-15:00  off-peak          interval * 1.5
      t=480-600  15:00-17:00  afternoon peak    interval / 1.8
      t=600-660  17:00-18:00  evening           interval
    """
    if cfg.demand_profile == "uniform":
        return cfg.inter_arrival

    if cfg.demand_profile == "peak":
        if 0 <= t < 120:
            return cfg.inter_arrival / 2.5
        return cfg.inter_arrival

    if cfg.demand_profile == "bimodal":
        if t < 120:
            return cfg.inter_arrival / 2.0
        if t < 480:
            return cfg.inter_arrival * 1.5
        if t < 600:
            return cfg.inter_arrival / 1.8
        return cfg.inter_arrival

    raise ValueError(f"Unknown demand profile: {cfg.demand_profile!r}")