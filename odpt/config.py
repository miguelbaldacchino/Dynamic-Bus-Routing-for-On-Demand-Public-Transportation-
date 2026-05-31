# config.py
# Centralised simulation parameters: fleet size, capacity, time windows,
# algorithm hyperparameters, and demand profiles.
# Not runnable — imported by all other modules.


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

    # ---- TS hyperparameters ----
    ts_tabu_tenure:    int   = 7          # iterations a move stays tabu
    ts_max_neighbours: int   = 50         # neighbour plans evaluated per iteration
    ts_iterations:     int   = 200        # max TS iterations per vehicle
    ts_patience:       int   = 30         # iterations without improvement before restart
    ts_time_limit:     float = 0.3        # seconds per vehicle (matches SA/GA)

    # ---- ALNS hyperparameters ----
    alns_iterations:    int   = 150       # max ALNS iterations per vehicle
    alns_q_min:         int   = 1         # min requests removed per destroy
    alns_q_max:         int   = 6         # max requests removed per destroy
    alns_reaction:      float = 0.1       # operator weight reaction factor
    alns_temp_factor:   float = 0.5       # initial_temp = factor * initial_cost
    alns_cooling:       float = 0.992     # temperature cooling per iteration
    alns_time_limit:    float = 0.3       # seconds per vehicle (matches SA/GA/TS)

    # ---- GA hyperparameters ----
    ga_population:    int   = 30         # individuals per generation
    ga_generations:   int   = 200        # max generations per vehicle
    ga_crossover:     float = 0.85       # OX crossover probability
    ga_mutation:      float = 0.40       # mutation probability per offspring
    ga_tournament:    int   = 3          # tournament selection size
    ga_elite:         int   = 2          # elites carried forward unchanged
    ga_time_limit:    float = 0.3        # seconds per vehicle (matches SA)


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