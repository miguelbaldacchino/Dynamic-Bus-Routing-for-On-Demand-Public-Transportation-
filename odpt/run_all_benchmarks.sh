#!/bin/bash
# =============================================================================
# THESIS BENCHMARK — ALL RUNS
# =============================================================================
#
# BEFORE RUNNING — edit MODEL_REGISTRY at the top of benchmark.py:
#
#   "rl_base": "rl_outputs/run_006/model.zip"           <- already correct
#   "rl_v3":   "rl_outputs/YOUR_V3_RUN/model.zip"       <- fill in
#   "rl_v4":   "rl_outputs/YOUR_V4_RUN/model_final.zip" <- fill in
#
# Recommended: run in tmux so it survives disconnection
#   tmux new-session -s bench
#   bash run_all_benchmarks.sh 2>&1 | tee benchmark_log.txt
#
# =============================================================================
# WHAT EVERY RUN CONTAINS
#
# Each run = N seeds x M policies individual simulations.
# Each simulation = one complete Malta day (05:30-22:30) for one policy.
#
# Policies:
#   Greedy family (5):  greedy | greedy+sa | greedy+ts | greedy+ga | greedy+alns
#   RL base     (5):  rl | rl+sa | rl+ts | rl+ga | rl+alns  (untuned)
#   RL v3       (5):  rl | rl+sa | rl+ts | rl+ga | rl+alns  (standalone tune)
#   RL v4       (5):  rl | rl+sa | rl+ts | rl+ga | rl+alns  (TS-initialiser)
#   = 20 policies total when all three RL models run
#   = 10 policies when --rl-model rl_v4 (greedy+rl_v4 only)
#
# RNG fairness: every policy on seed=42 sees the EXACT same request stream.
# _sim_rng, _noise_rng, _algo_rng are all seeded from cfg.seed, not the policy.
#
# =============================================================================
# CONFIRMED FINAL RUN LIST
#
#  #  Name              Seeds  Policies  Runs  Est.hrs  What varies
#  1  baseline            5      20       100   ~4.0    Nothing — all defaults
#  2  fleet_4             3      20        60   ~2.5    fleet_size=4
#  3  fleet_8             3      20        60   ~2.5    fleet_size=8
#  4  demand_quiet        3      10        30   ~0.5    inter_arrival=5.0 (~212 req)
#  5  demand_busy         3      20        60   ~5.0    inter_arrival=2.0 (~520 req)
#  6  maxwait_15          3      20        60   ~2.5    max_wait=15 min (OOD test)
#  7  maxwait_45          3      10        30   ~1.2    max_wait=45 min
#  8  capacity_8          3      10        30   ~1.2    vehicle_capacity=8
#  9  capacity_24         3      10        30   ~1.2    vehicle_capacity=24
# 10  ridefactor_15       3      10        30   ~1.2    ride_factor=1.5
# 11  ridefactor_30       3      10        30   ~1.2    ride_factor=3.0
# --- Tier 3: only if time allows ---
# 12  demand_peak         3      10        30   ~4.5    inter_arrival=1.5 (~705 req)
# 13  profile_uniform     3      10        30   ~1.2    demand_profile=uniform
#
#  Tier 1 (runs 1-5):     ~14.5 hrs
#  Tier 1+2 (runs 1-11):  ~21.8 hrs
#  All 13:                ~27.5 hrs
#
# DEFAULT SimulationConfig (what baseline uses, what sensitivity varies FROM):
#   fleet_size=6  vehicle_capacity=16  inter_arrival=3.0  n_requests=400(cap)
#   max_wait=30   ride_factor=2.5      demand_profile=malta  seed=42-46
# =============================================================================

set -e   # stop on first error

mkdir -p results

# =============================================================================
# TIER 1 — MUST RUN
# =============================================================================

# 1. BASELINE
# All 20 policies. All SimulationConfig defaults. 5 seeds (42-46).
# This is the primary results table for the thesis.
# 100 total simulation runs. ~4 hours.
python benchmark.py \
  --n-seeds 5 \
  --out results/baseline

# 2. FLEET SIZE: 4 vehicles
# All 20 policies. 3 seeds.
# Why all models: showing v4 > v3 > base holding under fleet stress
# is a direct thesis claim. 60 total runs. ~2.5 hours.
python benchmark.py \
  --n-seeds 3 \
  --fleet-size 4 \
  --out results/fleet_4

# 3. FLEET SIZE: 8 vehicles
# All 20 policies. 3 seeds.
# Upper bound: confirms advantage is structural not capacity-driven.
# 60 total runs. ~2.5 hours.
python benchmark.py \
  --n-seeds 3 \
  --fleet-size 8 \
  --out results/fleet_8

# 4. DEMAND: quiet day (inter_arrival=5.0, ~212 requests)
# Greedy + rl_v4 only. 3 seeds.
# Why --rl-model rl_v4: low load is less revealing for model comparison.
# Why inter_arrival not n_requests: at ia=3.0 the 17-hr service window
# physically fits ~353 req regardless of n_requests — the horizon binds
# first. inter_arrival is the only effective demand volume knob.
# Why n_requests=9999: removes the 400-request cap so ia alone drives volume.
# 30 total runs. ~0.5 hours.
python benchmark.py \
  --n-seeds 3 \
  --rl-model rl_v4 \
  --inter-arrival 5.0 \
  --n-requests 9999 \
  --out results/demand_quiet

# 5. DEMAND: busy day (inter_arrival=2.0, ~520 requests)
# All 20 policies. 3 seeds.
# Most revealing sensitivity run. Under higher load rejection rates rise —
# does v4's growing rejection penalty hold service rate where v3 and base
# degrade? This is the key stress-test graph.
# Longer episodes = ~5 hours.
python benchmark.py \
  --n-seeds 3 \
  --inter-arrival 2.0 \
  --n-requests 9999 \
  --out results/demand_busy

# =============================================================================
# TIER 2 — STRONGLY RECOMMENDED
# =============================================================================

# 6. MAX WAIT: 15 minutes (strict)
# All 20 policies. 3 seeds.
# Most academically interesting tier-2 run. RL was trained at max_wait=30.
# Testing at 15 is an OOD generalisation test across all three models.
# Real-world: Helsinki on-demand targets 10-15 min wait.
# 60 total runs. ~2.5 hours.
python benchmark.py \
  --n-seeds 3 \
  --max-wait 15 \
  --out results/maxwait_15

# 7. MAX WAIT: 45 minutes (relaxed)
# Greedy + rl_v4 only. 3 seeds.
# Confirms RL+TS gap narrows as constraint loosens — honest, expected result.
# 30 total runs. ~1.2 hours.
python benchmark.py \
  --n-seeds 3 \
  --rl-model rl_v4 \
  --max-wait 45 \
  --out results/maxwait_45

# 8. CAPACITY: 8 passengers (minivan)
# Greedy + rl_v4 only. 3 seeds.
# Grounded in Malta's real fleet mix post-2020.
# 30 total runs. ~1.2 hours.
python benchmark.py \
  --n-seeds 3 \
  --rl-model rl_v4 \
  --capacity 8 \
  --out results/capacity_8

# 9. CAPACITY: 24 passengers
# Greedy + rl_v4 only. 3 seeds.
# Upper bound — confirms advantage not artefact of 16-seat config.
# 30 total runs. ~1.2 hours.
python benchmark.py \
  --n-seeds 3 \
  --rl-model rl_v4 \
  --capacity 24 \
  --out results/capacity_24

# 10. RIDE FACTOR: 1.5 (very tight)
# Greedy + rl_v4 only. 3 seeds.
# Orthogonal to max_wait. A 10-min direct trip can only take 15 min max
# (5 min detour tolerance after the 3-min planning margin).
# Tests whether slack-preserving insertion helps when ride feasibility
# — not wait — is the binding constraint.
# Do NOT combine with --max-wait 15 (double-tightening = uninterpretable).
# 30 total runs. ~1.2 hours.
python benchmark.py \
  --n-seeds 3 \
  --rl-model rl_v4 \
  --ride-factor 1.5 \
  --out results/ridefactor_15

# 11. RIDE FACTOR: 3.0 (loose)
# Greedy + rl_v4 only. 3 seeds.
# Completes the 1.5 / 2.5 / 3.0 axis for the ride-factor sensitivity table.
# 30 total runs. ~1.2 hours.
python benchmark.py \
  --n-seeds 3 \
  --rl-model rl_v4 \
  --ride-factor 3.0 \
  --out results/ridefactor_30

# =============================================================================
# TIER 3 — ONLY IF COMPUTE BUDGET ALLOWS
# =============================================================================

# 12. DEMAND: peak event (inter_arrival=1.5, ~705 requests)
# Greedy + rl_v4 only. 3 seeds.
# Extreme overload — nearly double baseline. Shows algorithmic stress limits.
# Very long episodes. ~4.5 hours.
python benchmark.py \
  --n-seeds 3 \
  --rl-model rl_v4 \
  --inter-arrival 1.5 \
  --n-requests 9999 \
  --out results/demand_peak

# 13. DEMAND PROFILE: uniform (flat arrivals, no peaks)
# Greedy + rl_v4 only. 3 seeds.
# inter_arrival = how many requests arrive. demand_profile = when they arrive.
# These are orthogonal dimensions. Uniform removes all peak structure.
# Tests whether v4 fleet-balance reward (designed for peak clustering)
# transfers to demand without temporal structure.
# 30 total runs. ~1.2 hours.
python benchmark.py \
  --n-seeds 3 \
  --rl-model rl_v4 \
  --demand-profile uniform \
  --out results/profile_uniform

echo ""
echo "All benchmark runs complete."
echo "Results in: results/"
echo "Each subdirectory contains: aggregated.json, aggregated.csv, report.txt, runs/"
