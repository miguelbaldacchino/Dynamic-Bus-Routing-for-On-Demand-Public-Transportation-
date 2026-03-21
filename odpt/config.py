# config.py
# All time values are in MINUTES.
# Simulation time 0 = 05:30.
# service_end = 1020 = 22:30 (last request accepted).
# horizon = 1140 = 00:30 next day (sim clock stops; vehicles finish).
#
# Calibrated for the tallinja On Demand service (2019-2024 daily operation).
# Fleet: 6 minibuses, 16 passengers each.
# Zone: Sliema, St Julian's, San Gwann, Swieqi, Birkirkara, Msida,
#        Gzira, Ta' Xbiex, Pembroke, Balzan, Santa Venera.
# 72 stops, real OSRM travel times from Malta road network.

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # ---- Reproducibility ----
    seed:             int   = 42

    # ---- Simulation horizon (minutes, 0=05:30) ----
    service_end:      float = 1020.0     # 22:30 — stop accepting new requests
    horizon:          float = 1140.0     # 00:30 — sim ends; 2h buffer for completion

    # ---- Demand ----
    n_requests:       int   = 400        # 17h service day, ~18 requests/hour
    inter_arrival:    float = 3.0        # minutes between requests (baseline mean)
    demand_profile:   str   = "malta"    # "uniform" | "peak" | "bimodal" | "malta"
    stochastic_arrivals: bool = True     # True = Poisson (exponential gaps)
    n_nodes:          int   = 71         # nodes 1-71 available (0 = depot)

    # ---- Fleet ----
    fleet_size:       int   = 6          # MPT On Demand: 6 minibuses
    vehicle_capacity: int   = 16         # MPT minibuses: 16 passengers
    depot_node:       int   = 0          # Hal Qormi - Bankieri

    # ---- DARP constraints (minutes) ----
    ride_factor:      float = 2.5        # max ride = 2.5 x direct travel time
    max_wait:         float = 30.0       # latest pickup = request_time + 30 min
    ride_time_margin: float = 3.0        # minutes subtracted from max ride in planning

    # ---- Execution noise ----
    travel_noise:     float = 0.15       # lognormal sigma for travel time variability

    # ---- Objective weights (alpha, beta, gamma) ----
    weights:          tuple = (1.0, 2.0, 2.5)  # distance, wait_time, ride_time

    # ---- Policy ----
    policy:           str   = "greedy+sa"

    # ---- SA hyperparameters ----
    sa_initial_temp:  float = 5_000.0
    sa_cooling_rate:  float = 0.995
    sa_iterations:    int   = 5_000      # per vehicle
    sa_time_limit:    float = 0.3        # seconds per vehicle


def arrival_rate(t: float, cfg: SimulationConfig) -> float:
    """
    Mean inter-arrival time (minutes) at simulation time t.
    Simulation t=0 corresponds to 05:30.

    Malta profile (default):
      t=0-90      05:30-07:00  early morning    interval * 3.0   (sparse)
      t=90-210    07:00-09:00  morning peak     interval / 2.0   (heavy)
      t=210-390   09:00-12:00  mid-morning      interval * 1.2   (moderate)
      t=390-570   12:00-15:00  afternoon        interval * 1.5   (quiet)
      t=570-750   15:00-18:00  evening peak     interval / 1.8   (heavy)
      t=750-870   18:00-20:00  evening          interval * 1.0   (moderate)
      t=870-1020  20:00-22:30  night            interval * 2.0   (sparse)
    """
    if cfg.demand_profile == "uniform":
        return cfg.inter_arrival

    if cfg.demand_profile == "peak":
        if 90 <= t < 210:
            return cfg.inter_arrival / 2.5
        return cfg.inter_arrival

    if cfg.demand_profile == "bimodal":
        if t < 90:
            return cfg.inter_arrival * 3.0
        if t < 210:
            return cfg.inter_arrival / 2.0
        if t < 570:
            return cfg.inter_arrival * 1.5
        if t < 750:
            return cfg.inter_arrival / 1.8
        return cfg.inter_arrival

    if cfg.demand_profile == "malta":
        if t < 90:                 # 05:30-07:00 early morning
            return cfg.inter_arrival * 3.0
        if t < 210:                # 07:00-09:00 morning peak
            return cfg.inter_arrival / 2.0
        if t < 390:                # 09:00-12:00 mid-morning
            return cfg.inter_arrival * 1.2
        if t < 570:                # 12:00-15:00 afternoon
            return cfg.inter_arrival * 1.5
        if t < 750:                # 15:00-18:00 evening peak
            return cfg.inter_arrival / 1.8
        if t < 870:                # 18:00-20:00 evening
            return cfg.inter_arrival * 1.0
        return cfg.inter_arrival * 2.0  # 20:00-22:30 night

    raise ValueError(f"Unknown demand profile: {cfg.demand_profile!r}")