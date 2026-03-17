# config.py
# All time values are in MINUTES.
# Simulation time 0 = 07:00.  Horizon 660 = 18:00.

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # ---- Reproducibility ----
    seed:             int   = 42

    # ---- Simulation horizon (minutes, 0=07:00) ----
    horizon:          float = 800.0      # 18:00

    # ---- Demand ----
    # 220 requests over 660 min with bimodal profile gives realistic load
    n_requests:       int   = 220
    inter_arrival:    float = 3.0        # minutes between requests (baseline)
    demand_profile:   str   = "bimodal"  # "uniform" | "peak" | "bimodal"
    n_nodes:          int   = 15         # nodes 1-15 available (0 = depot)

    # ---- Fleet ----
    fleet_size:       int   = 8
    vehicle_capacity: int   = 12
    depot_node:       int   = 0

    # ---- DARP constraints (minutes) ----
    ride_factor:      float = 2.5    # max ride = 2.5 x direct travel time
    max_wait:         float = 30.0   # latest pickup = request_time + 30 min

    # ---- Objective weights (alpha, beta, gamma) ----
    weights:          tuple = (1.0, 2.0, 2.5)  # distance, wait_time, ride_time

    # ---- Policy ----
    policy:           str   = "greedy+sa"

    # ---- SA hyperparameters ----
    sa_initial_temp:  float = 10_000.0
    sa_cooling_rate:  float = 0.997
    sa_iterations:    int   = 20_000
    sa_time_limit:    float = 3.0  # seconds per decision


def arrival_rate(t: float, cfg: SimulationConfig) -> float:
    """
    Inter-arrival time (minutes) at simulation time t.

    Bimodal (default):
      t=0-120    07:00-09:00  morning peak      interval / 2.0   (2x requests)
      t=120-480  09:00-15:00  off-peak          interval * 1.5   (quiet)
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