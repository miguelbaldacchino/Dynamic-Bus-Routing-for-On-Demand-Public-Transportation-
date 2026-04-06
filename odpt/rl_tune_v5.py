#!/usr/bin/env python3
# rl_tune_v5.py
# Optuna hyperparameter search — v5 "TS-initialiser, corrected".
#
# =====================================================================
# PROGRESSION SUMMARY: v1 → v3 → v4 → v5
# =====================================================================
#
# v1: Standalone RL objective.  Discovered rejection gaming, norm_obs
#     mismatch.  10 search dims, 27 trials.
#
# v3: Refined standalone RL.  Fixed norm_obs.  7 search dims, 30 trials.
#     Best: wait_served=10.39, svc=89.5%.  Strong standalone baseline.
#
# v4: TS-initialiser objective — conceptually correct, four execution
#     errors made results unreliable (see FIXES below).
#
# v5: v4 concepts + four targeted fixes.  Results are directly comparable
#     to baseline_aggregated.csv (the benchmark ground truth).
#
# =====================================================================
# FIXES FROM v4
# =====================================================================
#
# FIX 1 — SEED MISMATCH
#   v4 calibrated GREEDY_TS_WAIT=7.41 from a single-seed run, and tuned
#   on seeds 2000–2009.  The benchmark uses 20 different seeds where
#   greedy+ts averages mean_wait_all=11.63.  v4 was beating an easy
#   target on easy instances — gains evaporated on the benchmark seeds.
#   v5: eval seeds = 3000+i (distinct from training: 42+i, v3/v4: 2000+i).
#       All baselines sourced from baseline_aggregated.csv (20 seeds).
#
# FIX 2 — w_wait FIXED AT 1.0
#   v3 found w_wait in [1.5, 2.8] is necessary for competitive wait times.
#   v4 fixed it at 1.0 as "less wait obsession", but the benchmark measures
#   wait time.  This was the single largest cause of v4's underperformance
#   vs v3 hybrids.
#   v5: w_wait restored to search space (1.0–3.0).
#
# FIX 3 — SLACK BONUS WAS MATHEMATICALLY REDUNDANT
#   slack_norm = (max_wait - est_wait) / max_wait = 1 - est_wait/max_wait
#   Combined with wait penalty this simplifies to:
#     w_slack - (w_wait + w_slack) * (est_wait/max_wait)
#   w_slack added no new information — it merely inflated w_acceptance and
#   w_wait through the back door.  This is why v4 (w_wait=1.0, w_slack=0.43)
#   converged to an effective w_wait of 1.43 — sneaking back toward v3's
#   range.
#   v5: w_slack removed entirely.  Search reduced from 7 to 6 dimensions.
#       Faster TPE convergence, no redundant axes.
#
# FIX 4 — REJECTION PENALTY IN OBJECTIVE TOO WEAK
#   v4: score = mean_wait + 0.1*rejected → 42 rejections costs 4.2 pts
#   vs wait of 9.17.  Service rate could drop freely.
#   v5: MAX_WAIT_PENALTY=60 (2× the system constraint of 30).  A rejected
#   passenger didn't just wait — they got no service.  The 2× multiplier
#   encodes this asymmetry and ensures Optuna cannot improve score by
#   rejecting unless the wait savings genuinely outweigh the cost.
#
# ADDITIONAL FIX — BALANCE METRIC WAS BLIND TO QUEUED ROUTES
#   v4 used len(v.onboard) — only current passengers.  A vehicle with 0
#   onboard but 15 queued pickups looked "empty", causing over-assignment.
#   v5: uses len(v.onboard) + len(v.plan)//2 — total assigned workload.
#   (v.plan has 2 stops per request: PU + DO, so //2 gives queued count.)
#
# =====================================================================
# WHAT v5 KEEPS FROM v4 (correct concepts)
# =====================================================================
#
# CONCEPT 1 — GROWING REJECTION PENALTY (reward-level, not objective)
#   -w_rejection * (1 + n_rejected_so_far / n_requests)
#   Prevents "reject the hard ones early" gaming.  Proved effective:
#   v4 standalone svc=85.4% vs v3's 81.8% despite lower w_rejection.
#
# CONCEPT 2 — FLEET BALANCE BONUS (w_imbalance)
#   Rewards even workload distribution across vehicles.  Gives TS more
#   inter-vehicle move options.  Now uses correct workload metric.
#
# CONCEPT 3 — TS-INITIALISER OBJECTIVE
#   Optuna scores RL+TS combo, not standalone RL.  The fundamental v4
#   insight was correct.
#
# =====================================================================
# OBJECTIVE FUNCTION
# =====================================================================
#
# Score = mean_wait_all_penalised + OBJ_W_P95 * p95_wait   (MINIMISE)
#
# mean_wait_all_penalised = (served*mean_wait + rejected*MAX_WAIT_PENALTY)
#                           / n_requests
#
# MAX_WAIT_PENALTY = 60 min  (2× system MAX_WAIT — academic design choice)
# OBJ_W_P95       = 0.05    (tail penalty — dimensionally consistent, minutes)
#
# Calibration (baseline_aggregated.csv, 20 seeds, greedy+ts):
#   served=313, mean_wait=9.17, rejected=42, p95=23.03, n_total=355
#   mean_wait_all = (313*9.17 + 42*60) / 355 = 15.88
#   score         = 15.88 + 0.05*23.03 = 17.03  ← target to beat
#
# =====================================================================
# SEARCH SPACE (6 parameters)
# =====================================================================
#
# REWARD WEIGHTS (3 searched):
#   w_wait       1.0–3.0   RESTORED (v3 found ~2.4; critical fix from v4)
#   w_rejection  4.0–9.0   informed by v3 best (8.46) and v4 best (3.22)
#   w_imbalance  0.1–0.8   narrowed from v4; v4 best was 0.22
#
# PPO HYPERPARAMETERS (3 searched):
#   lr_start    1e-4–4e-4 (log)
#   gamma       0.990–0.997
#   ent_coef    0.01–0.05 (log)
#
# FIXED reward weights:
#   w_acceptance 1.0   (unchanged all versions)
#   w_ride       0.8   (moderate — ride matters, less than wait)
#   w_ride_sq    0.0   (removed in v4; stay removed)
#   w_detour     0.0   (redundant with ride penalty)
#   w_cost       0.1   (low; balance > cost minimisation)
#   w_slack      —     (REMOVED: mathematically redundant with w_wait)
#
# FIXED PPO (proven from v1–v3):
#   lr_schedule  linear   (all v3 best trials used linear)
#   n_steps      1024
#   n_epochs     5
#   batch_size   128
#   vf_coef      1.0
#   gae_lambda   0.95
#   clip_range   0.2
#   net_arch     [128, 128]
#   norm_obs     False   (obs already in [-1,1])
#   norm_reward  True    (critical from v1 run005)
#
# =====================================================================
# Usage:
#   python rl_tune_v5.py                   # 30 trials
#   python rl_tune_v5.py --resume          # safe resume
#   python rl_tune_v5.py --samples 5       # quick smoke test
# =====================================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import optuna
from optuna.samplers import TPESampler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_WAIT = 30.0          # system constraint — actual maximum passenger wait

# Rejection penalty for the OBJECTIVE only (not the reward).
# Set to 2× MAX_WAIT: a rejected passenger didn't just wait — they got
# nothing.  This asymmetry must be reflected in the search objective.
MAX_WAIT_PENALTY = 60.0

OBJ_W_P95 = 0.05  # p95_wait weight — in minutes (dimensionally consistent)

# Baselines from baseline_aggregated.csv (20 seeds, greedy+ts).
# These are the numbers your benchmark comparison is evaluated against.
GREEDY_TS_MEAN_WAIT  = 9.1672
GREEDY_TS_REJECTED   = 42.0
GREEDY_TS_N_SERVED   = 313.0
GREEDY_TS_N_REQUESTS = 355
GREEDY_TS_P95_WAIT   = 23.0255

_gts_mw_all = (
    GREEDY_TS_N_SERVED * GREEDY_TS_MEAN_WAIT + GREEDY_TS_REJECTED * MAX_WAIT_PENALTY
) / GREEDY_TS_N_REQUESTS
GREEDY_TS_SCORE = _gts_mw_all + OBJ_W_P95 * GREEDY_TS_P95_WAIT
# = (313*9.17 + 42*60)/355 + 0.05*23.03 ≈ 17.03


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def compute_objective(rl_ts_metrics: dict, n_requests: int) -> tuple[float, dict]:
    """
    Score = mean_wait_all_penalised + OBJ_W_P95 * p95_wait   (MINIMISE)

    Both terms are in minutes.  mean_wait_all_penalised encodes both
    service rate and wait time in a single number — Optuna cannot improve
    it by sacrificing service rate unless wait savings genuinely exceed the
    MAX_WAIT_PENALTY cost of each extra rejection.
    """
    svc    = rl_ts_metrics.get("service_rate") or 0.0
    mw     = rl_ts_metrics.get("mean_wait")    or MAX_WAIT
    rej    = rl_ts_metrics.get("rejected")     or float(n_requests)
    p95w   = rl_ts_metrics.get("p95_wait")     or MAX_WAIT
    mr     = rl_ts_metrics.get("mean_ride")    or 0.0
    ts_imp = rl_ts_metrics.get("ts_improvements") or 0.0

    n_served = svc * n_requests
    mw_all   = (n_served * mw + rej * MAX_WAIT_PENALTY) / max(n_served + rej, 1)
    score    = mw_all + OBJ_W_P95 * p95w

    return score, {
        "rl_ts_service_rate":    round(svc,    4),
        "rl_ts_mean_wait":       round(mw,     2),
        "rl_ts_mean_wait_all":   round(mw_all, 2),
        "rl_ts_p95_wait":        round(p95w,   2),
        "rl_ts_mean_ride":       round(mr,     2),
        "rl_ts_rejected":        round(rej,    1),
        "rl_ts_improvements":    round(ts_imp, 1),
        "score":                 round(score,  3),
        "vs_greedy_ts_score":    round(score - GREEDY_TS_SCORE,    3),
        "vs_greedy_ts_wait":     round(mw    - GREEDY_TS_MEAN_WAIT, 2),
        "vs_greedy_ts_wait_all": round(mw_all - _gts_mw_all,        2),
    }


# ---------------------------------------------------------------------------
# DARPEnvV5 — monkey-patched reward
# ---------------------------------------------------------------------------

class DARPEnvV5:
    """
    Patches DARPEnv with the v5 reward.

    Changes from v4:
      - w_wait is now a searched parameter (restored from fixed 1.0)
      - w_slack removed (collinear with w_wait — see header)
      - balance metric uses total assigned workload: onboard + queued pickups
        (v.plan has 2 stops per request — PU and DO — so queued = plan//2)

    Per-step reward (accepted insertion):
      + w_acceptance                             (fixed 1.0)
      - w_wait    * (est_wait / max_wait)        (SEARCHED 1.0–3.0)
      - w_ride    * (est_ride / max_ride)        (fixed 0.8)
      - w_cost    * (delta_cost / norm)          (fixed 0.1)
      + w_imbalance * balance_bonus              (SEARCHED 0.1–0.8)

    Rejection penalty (growing):
      - w_rejection * (1 + n_rejected_so_far / n_requests)
    """

    @staticmethod
    def make(cfg, w_wait, w_rejection, w_imbalance):
        from rl_env import DARPEnv
        import types

        env = DARPEnv(
            cfg          = cfg,
            reward_mode  = "composite",
            w_acceptance = 1.0,
            w_wait       = w_wait,
            w_ride       = 0.8,
            w_ride_sq    = 0.0,
            w_detour     = 0.0,
            w_cost       = 0.1,
            w_rejection  = w_rejection,
        )
        env._v5_w_imbalance = w_imbalance
        env._v5_n_requests  = cfg.n_requests

        def _v5_compute_reward(self, req, old_cost, new_cost, est_wait, est_ride):
            direct   = max(req.direct_time or 1.0, 1.0)
            max_ride = direct * self.cfg.ride_factor
            norm     = direct * 5.0

            wait_pen = -self.w_wait * (est_wait / max(self.cfg.max_wait, 1.0))
            ride_pen = -self.w_ride * (est_ride / max(max_ride, 1.0))
            cost_pen = -self.w_cost * ((new_cost - old_cost) / max(norm, 1.0))

            # Fleet balance bonus — rewards even WORKLOAD distribution.
            # Uses onboard + queued (plan//2) to avoid blind spot for
            # vehicles with empty seats but full route plans.
            loads = [
                len(v.onboard) + len(v.plan) // 2
                for v in self._vehicles.values()
            ]
            load_std      = float(np.std(loads)) if loads else 0.0
            balance_bonus = self._v5_w_imbalance * (
                1.0 - min(load_std / max(self.cfg.vehicle_capacity, 1), 1.0)
            )

            return (self.w_acceptance
                    + wait_pen + ride_pen + cost_pen
                    + balance_bonus)

        def _v5_rejection_penalty(self, req):
            n_rej  = self._n_rejections   # already incremented before call
            growth = 1.0 + (n_rej / max(self._v5_n_requests, 1))
            return -self.w_rejection * growth

        env._compute_reward    = types.MethodType(_v5_compute_reward,    env)
        env._rejection_penalty = types.MethodType(_v5_rejection_penalty, env)
        return env


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_standalone(model, cfg, n_episodes: int) -> dict:
    """Standalone RL — logged for graphs/context, NOT the Optuna objective."""
    from rl_env import DARPEnv
    from config import SimulationConfig

    results = []
    for i in range(n_episodes):
        eval_cfg = SimulationConfig(
            seed=3000 + i,
            fleet_size=cfg.fleet_size, vehicle_capacity=cfg.vehicle_capacity,
            depot_node=cfg.depot_node, n_requests=cfg.n_requests,
            demand_profile=cfg.demand_profile,
            stochastic_arrivals=cfg.stochastic_arrivals,
            travel_noise=0.0, n_nodes=cfg.n_nodes,
        )
        env = DARPEnv(cfg=eval_cfg)
        obs, _ = env.reset(seed=eval_cfg.seed)
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        env._advance_vehicles_to(cfg.service_end + 500)
        results.append(env.episode_summary())

    def _m(k):
        vals = [r[k] for r in results if r.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "service_rate": _m("service_rate"),
        "mean_wait":    _m("mean_wait"),
        "p95_wait":     _m("p95_wait"),
        "mean_ride":    _m("mean_ride"),
        "rejected":     _m("rejected"),
    }


def _evaluate_rl_plus_ts(model, cfg, n_episodes: int) -> dict:
    """
    RL + TS combo — THIS IS THE OPTUNA OBJECTIVE.

    TS parameters match production config.py defaults exactly:
      tabu_tenure=7, max_neighbours=50, iterations=200,
      patience=30, decision_time_limit=0.3

    Seeds 3000+i match standalone eval for consistency.
    """
    from rl_env import DARPEnv
    from ts import TSPolicy
    from feasibility import evaluate_plan, check_feasibility
    from config import SimulationConfig
    from copy import deepcopy
    import random as _random

    ts = TSPolicy(
        tabu_tenure=7, max_neighbours=50, iterations=200,
        patience=30, decision_time_limit=0.3,
        rng=_random.Random(999),
    )

    results = []
    for i in range(n_episodes):
        eval_cfg = SimulationConfig(
            seed=3000 + i,
            fleet_size=cfg.fleet_size, vehicle_capacity=cfg.vehicle_capacity,
            depot_node=cfg.depot_node, n_requests=cfg.n_requests,
            demand_profile=cfg.demand_profile,
            stochastic_arrivals=cfg.stochastic_arrivals,
            travel_noise=0.0, n_nodes=cfg.n_nodes,
        )
        env = DARPEnv(cfg=eval_cfg)
        obs, _ = env.reset(seed=eval_cfg.seed)
        done = False
        ts_improvements = 0

        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            if not done:
                ts_state = {**env._system_state, "vehicles": {}}
                for vid, v in env._vehicles.items():
                    vs = v.to_state_dict(env._sim_time)
                    n_committed = 1 if v.in_transit_stop is not None else 0
                    ts_state["vehicles"][vid] = {
                        **vs,
                        "plan":        deepcopy(vs["plan_snapshot"]),
                        "n_committed": n_committed,
                    }
                changes = ts.propose(ts_state, check_feasibility, cfg.weights)
                if changes:
                    before = sum(evaluate_plan(
                        ts_state["vehicles"][vid]["plan"],
                        ts_state["vehicles"][vid],
                        env._system_state, cfg.weights,
                    ) for vid in changes)
                    after = sum(evaluate_plan(
                        new_plan,
                        ts_state["vehicles"][vid],
                        env._system_state, cfg.weights,
                    ) for vid, new_plan in changes.items())
                    if after < before:
                        for vid, new_plan in changes.items():
                            v = env._vehicles[vid]
                            n_committed = 1 if v.in_transit_stop is not None else 0
                            v.plan = new_plan[n_committed:]
                        ts_improvements += 1

        env._advance_vehicles_to(cfg.service_end + 500)
        s = env.episode_summary()
        s["ts_improvements"] = ts_improvements
        results.append(s)

    def _m(k):
        vals = [r[k] for r in results if r.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "service_rate":    _m("service_rate"),
        "mean_wait":       _m("mean_wait"),
        "p95_wait":        _m("p95_wait"),
        "mean_ride":       _m("mean_ride"),
        "rejected":        _m("rejected"),
        "ts_improvements": _m("ts_improvements"),
    }


# ---------------------------------------------------------------------------
# Single Optuna trial
# ---------------------------------------------------------------------------

def run_trial(trial: optuna.Trial, timesteps: int, n_envs: int,
              tb_base: str) -> float:
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from config import SimulationConfig

    w_wait      = trial.suggest_float("w_wait",      1.0,  3.0)
    w_rejection = trial.suggest_float("w_rejection", 4.0,  9.0)
    w_imbalance = trial.suggest_float("w_imbalance", 0.1,  0.8)
    lr_start    = trial.suggest_float("lr_start",    1e-4, 4e-4, log=True)
    gamma       = trial.suggest_float("gamma",       0.990, 0.997)
    ent_coef    = trial.suggest_float("ent_coef",    0.01,  0.05, log=True)

    learning_rate = lambda progress: lr_start * progress  # linear

    print(f"\n  --- Trial {trial.number + 1} ---")
    print(f"    Reward: w_wait={w_wait:.2f}  w_rej={w_rejection:.2f}  "
          f"w_imbal={w_imbalance:.2f}")
    print(f"    PPO:    lr={lr_start:.5f}(linear)  "
          f"gamma={gamma:.4f}  ent={ent_coef:.4f}")

    cfg = SimulationConfig(
        seed=42, fleet_size=6, vehicle_capacity=16,
        depot_node=0, n_requests=400, demand_profile="malta",
        stochastic_arrivals=True, travel_noise=0.0, n_nodes=71,
    )

    def make_env(seed: int):
        def _init():
            env_cfg = SimulationConfig(
                seed=seed, fleet_size=cfg.fleet_size,
                vehicle_capacity=cfg.vehicle_capacity,
                depot_node=cfg.depot_node, n_requests=cfg.n_requests,
                demand_profile=cfg.demand_profile,
                stochastic_arrivals=cfg.stochastic_arrivals,
                travel_noise=0.0, n_nodes=cfg.n_nodes,
            )
            return DARPEnvV5.make(env_cfg, w_wait, w_rejection, w_imbalance)
        return _init

    vec_env = SubprocVecEnv([make_env(42 + i) for i in range(n_envs)])
    vec_env = VecNormalize(
        vec_env, norm_obs=False, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=gamma,
    )

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    tb_log_dir = os.path.join(tb_base, f"trial_{trial.number + 1:03d}")

    model = MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef,
        n_steps=1024, batch_size=128, n_epochs=5,
        vf_coef=1.0, gae_lambda=0.95, clip_range=0.2, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0, tensorboard_log=tb_log_dir, device=device,
    )

    t0 = time.time()

    class TimeoutCallback(BaseCallback):
        def __init__(self): super().__init__(verbose=0)
        def _on_step(self) -> bool: return time.time() - t0 < 30 * 60

    model.learn(timesteps, callback=CallbackList([TimeoutCallback()]))
    train_time = time.time() - t0

    # 1. Standalone — logged for graphs, not the objective
    print(f"    Evaluating standalone (5 eps)...")
    standalone = _evaluate_standalone(model, cfg, n_episodes=5)
    print(f"    Standalone: svc={standalone.get('service_rate', 0):.1%}"
          f"  wait={standalone.get('mean_wait', 0):.2f}"
          f"  rej={standalone.get('rejected', 0):.0f}")

    # 2. RL+TS — THE OBJECTIVE (15 eps for low variance)
    print(f"    Evaluating rl+ts (15 eps)...")
    rl_ts = _evaluate_rl_plus_ts(model, cfg, n_episodes=15)
    print(f"    rl+ts: svc={rl_ts.get('service_rate', 0):.1%}"
          f"  wait={rl_ts.get('mean_wait', 0):.2f}"
          f"  rej={rl_ts.get('rejected', 0):.0f}"
          f"  ts_impr={rl_ts.get('ts_improvements', 0):.1f}")

    vec_env.close()

    score, metrics = compute_objective(rl_ts, cfg.n_requests)

    for k, v in metrics.items():
        if v is not None:
            trial.set_user_attr(k, v)
    for k, v in standalone.items():
        if v is not None:
            trial.set_user_attr(f"sa_{k}", round(v, 4) if isinstance(v, float) else v)
    trial.set_user_attr("train_time_s", round(train_time, 1))

    delta  = metrics.get("vs_greedy_ts_score", 0)
    mw_all = metrics.get("rl_ts_mean_wait_all", 0)
    status = "✓ BEATS BASELINE" if score < GREEDY_TS_SCORE else "✗ below baseline"
    print(f"    Score={score:.3f}  wait_all={mw_all:.2f}"
          f"  delta={delta:+.3f}  {status}")

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="rl_tune_v5 — corrected TS-initialiser")
    parser.add_argument("--samples",    type=int,  default=30)
    parser.add_argument("--timesteps",  type=int,  default=300_000)
    parser.add_argument("--n-envs",     type=int,  default=6)
    parser.add_argument("--output-dir", default="rl_outputs/tune_v5")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--study-name", default="darp_ppo_v5")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tb_base    = os.path.join(args.output_dir, "tb")
    study_path = os.path.join(args.output_dir, "study.pkl")
    os.makedirs(tb_base, exist_ok=True)

    if args.resume and os.path.exists(study_path):
        with open(study_path, "rb") as f:
            study = pickle.load(f)
        completed = len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"Resuming: {completed} trials complete")
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            sampler=TPESampler(seed=42, n_startup_trials=8),
        )

        # Seed 1: v3 best config — strongest prior on w_wait=2.42
        study.enqueue_trial({
            "w_wait": 2.42, "w_rejection": 8.46, "w_imbalance": 0.30,
            "lr_start": 1.54e-4, "gamma": 0.9947, "ent_coef": 0.0266,
        })
        # Seed 2: v4 best with w_wait restored to 1.8
        study.enqueue_trial({
            "w_wait": 1.8, "w_rejection": 3.22, "w_imbalance": 0.22,
            "lr_start": 3.0e-4, "gamma": 0.9937, "ent_coef": 0.021,
        })
        # Seed 3: balanced midpoint hedge
        study.enqueue_trial({
            "w_wait": 2.0, "w_rejection": 6.0, "w_imbalance": 0.40,
            "lr_start": 2.0e-4, "gamma": 0.993, "ent_coef": 0.020,
        })
        print(f"New study: {args.study_name} (3 seed trials from v1–v4 history)")

    def trial_callback(study: optuna.Study, trial: optuna.Trial):
        attrs = trial.user_attrs
        delta = attrs.get("vs_greedy_ts_score", 0)
        print(f"\n  Trial {trial.number + 1} result:")
        print(f"    score:          {trial.value:.3f}  "
              f"(greedy+ts: {GREEDY_TS_SCORE:.2f}  delta: {delta:+.3f})")
        print(f"    rl+ts svc:      {attrs.get('rl_ts_service_rate', 0):.1%}")
        print(f"    rl+ts wait:     {attrs.get('rl_ts_mean_wait', 0):.2f} min")
        print(f"    rl+ts wait_all: {attrs.get('rl_ts_mean_wait_all', 0):.2f} min")
        print(f"    rl+ts p95_wait: {attrs.get('rl_ts_p95_wait', 0):.2f} min")
        print(f"    rl+ts rejected: {attrs.get('rl_ts_rejected', 0):.1f}")
        print(f"    rl+ts ts_impr:  {attrs.get('rl_ts_improvements', 0):.1f}")
        print(f"    standalone svc: {attrs.get('sa_service_rate', 0):.1%}")
        print(f"    standalone wait:{attrs.get('sa_mean_wait', 0):.2f}")
        print(f"    train_time:     {attrs.get('train_time_s', 0):.0f}s")

        valid = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value != float("inf")]
        if valid:
            b = min(valid, key=lambda t: t.value)
            b_delta = b.user_attrs.get("vs_greedy_ts_score", 0)
            print(f"  >>> Best: trial {b.number + 1}"
                  f"  score={b.value:.3f}"
                  f"  delta={b_delta:+.3f}"
                  f"  wait_all={b.user_attrs.get('rl_ts_mean_wait_all', 0):.2f}"
                  f"  svc={b.user_attrs.get('rl_ts_service_rate', 0):.1%}")

        with open(study_path, "wb") as f:
            pickle.dump(study, f)

    completed   = [t for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE]
    n_remaining = args.samples - len(completed)

    if n_remaining <= 0:
        print(f"All {args.samples} trials already complete.")
        return

    print("=" * 65)
    print("rl_tune_v5 — Corrected TS-Initialiser (6-dim search)")
    print("=" * 65)
    print(f"  Objective:  mean_wait_all_penalised + 0.05*p95_wait  (rl+ts)")
    print(f"              MAX_WAIT_PENALTY={MAX_WAIT_PENALTY:.0f} min  "
          f"(system MAX_WAIT={MAX_WAIT:.0f} min)")
    print()
    print(f"  Key fixes from v4:")
    print(f"    w_wait    searched 1.0–3.0  (was wrongly fixed at 1.0)")
    print(f"    w_slack   REMOVED  (collinear with w_wait)")
    print(f"    balance   onboard + plan//2  (was onboard only)")
    print(f"    seeds     3000+i  (not 2000+i)")
    print(f"    penalty   MAX_WAIT_PENALTY=60  (not 30)")
    print()
    print(f"  Searched (6 params): w_wait  w_rejection  w_imbalance  "
          f"lr  gamma  ent")
    print(f"  Fixed reward: w_acceptance=1.0  w_ride=0.8  w_cost=0.1")
    print(f"  Fixed PPO:    linear LR  1024 steps  5 epochs  [128,128]  "
          f"norm_reward=True")
    print()
    print(f"  Baseline (baseline_aggregated.csv, 20 seeds):")
    print(f"    greedy+ts:  wait={GREEDY_TS_MEAN_WAIT:.2f}  "
          f"rej={GREEDY_TS_REJECTED:.0f}  score={GREEDY_TS_SCORE:.2f}")
    print(f"    Target: score < {GREEDY_TS_SCORE:.2f}")
    print()
    print(f"  Trials: {args.samples} ({n_remaining} remaining)  |  "
          f"Timesteps: {args.timesteps:,}  |  Envs: {args.n_envs}")
    print(f"  Safe to Ctrl+C — study saved after every trial.")
    print("=" * 65)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    t_start = time.time()
    study.optimize(
        lambda trial: run_trial(trial, args.timesteps, args.n_envs, tb_base),
        n_trials=n_remaining, callbacks=[trial_callback],
        show_progress_bar=False,
    )
    total_time = time.time() - t_start

    valid = [t for t in study.trials
             if t.state == optuna.trial.TrialState.COMPLETE
             and t.value != float("inf")]

    if not valid:
        print("\nNo valid trials completed.")
        return

    best   = min(valid, key=lambda t: t.value)
    best_p = best.params
    best_a = best.user_attrs

    print("\n" + "=" * 65)
    print("SEARCH v5 COMPLETE")
    print("=" * 65)
    print(f"  Time:   {total_time/3600:.1f} hours  |  "
          f"Trials: {len(valid)} valid / {len(study.trials)} total")
    print()
    print(f"  Best trial #{best.number + 1}:")
    print(f"    score:          {best.value:.3f}"
          f"  (greedy+ts: {GREEDY_TS_SCORE:.2f}"
          f"  delta: {best.value - GREEDY_TS_SCORE:+.3f})")
    print(f"    rl+ts svc:      {best_a.get('rl_ts_service_rate', 0):.1%}")
    print(f"    rl+ts wait:     {best_a.get('rl_ts_mean_wait', 0):.2f} min")
    print(f"    rl+ts wait_all: {best_a.get('rl_ts_mean_wait_all', 0):.2f} min")
    print(f"    rl+ts p95_wait: {best_a.get('rl_ts_p95_wait', 0):.2f} min")
    print(f"    rl+ts rejected: {best_a.get('rl_ts_rejected', 0):.1f}")
    print(f"    standalone svc: {best_a.get('sa_service_rate', 0):.1%}")
    print(f"    standalone wait:{best_a.get('sa_mean_wait', 0):.2f}")
    print()
    print(f"    w_wait={best_p['w_wait']:.3f}  "
          f"w_rejection={best_p['w_rejection']:.3f}  "
          f"w_imbalance={best_p['w_imbalance']:.3f}")
    print(f"    lr={best_p['lr_start']:.5f}  "
          f"gamma={best_p['gamma']:.4f}  "
          f"ent={best_p['ent_coef']:.4f}")

    best_config = {
        "source":        "rl_tune_v5.py",
        "objective":     "mean_wait_all_penalised(MAX_WAIT_PENALTY=60) + 0.05*p95_wait",
        "reward_mode":   "composite",
        "w_acceptance":  1.0,
        "w_wait":        best_p["w_wait"],
        "w_ride":        0.8,
        "w_ride_sq":     0.0,
        "w_detour":      0.0,
        "w_cost":        0.1,
        "w_rejection":   best_p["w_rejection"],
        "w_imbalance":   best_p["w_imbalance"],
        "lr_start":      best_p["lr_start"],
        "lr_schedule":   "linear",
        "gamma":         best_p["gamma"],
        "ent_coef":      best_p["ent_coef"],
        "n_steps":       1024,
        "n_epochs":      5,
        "batch_size":    128,
        "vf_coef":       1.0,
        "gae_lambda":    0.95,
        "clip_range":    0.2,
        "max_grad_norm": 0.5,
        "net_arch":      [128, 128],
        "norm_obs":      False,
        "norm_reward":   True,
        "timesteps":     args.timesteps,
        "n_envs":        args.n_envs,
        "best_trial_number":            best.number + 1,
        "n_trials":                     len(study.trials),
        "n_valid_trials":               len(valid),
        "best_score":                   round(best.value, 4),
        "achieved_rl_ts_service_rate":  best_a.get("rl_ts_service_rate"),
        "achieved_rl_ts_mean_wait":     best_a.get("rl_ts_mean_wait"),
        "achieved_rl_ts_mean_wait_all": best_a.get("rl_ts_mean_wait_all"),
        "achieved_rl_ts_p95_wait":      best_a.get("rl_ts_p95_wait"),
        "achieved_rl_ts_rejected":      best_a.get("rl_ts_rejected"),
        "achieved_standalone_svc":      best_a.get("sa_service_rate"),
        "achieved_standalone_wait":     best_a.get("sa_mean_wait"),
        "greedy_ts_mean_wait":          GREEDY_TS_MEAN_WAIT,
        "greedy_ts_score":              round(GREEDY_TS_SCORE, 3),
        "max_wait_penalty":             MAX_WAIT_PENALTY,
    }
    config_path = os.path.join(args.output_dir, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)

    csv_path   = os.path.join(args.output_dir, "all_trials.csv")
    fieldnames = [
        "trial", "score", "vs_greedy_ts_score",
        "rl_ts_service_rate", "rl_ts_mean_wait", "rl_ts_mean_wait_all",
        "rl_ts_p95_wait", "rl_ts_mean_ride", "rl_ts_rejected",
        "rl_ts_improvements", "sa_service_rate", "sa_mean_wait",
        "w_wait", "w_rejection", "w_imbalance",
        "lr_start", "gamma", "ent_coef", "train_time_s",
    ]
    all_complete = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in sorted(all_complete,
                        key=lambda x: x.value if x.value != float("inf") else 999):
            row = {"trial": t.number + 1, "score": round(t.value, 4)}
            row.update(t.user_attrs)
            row.update(t.params)
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    top5 = sorted(valid, key=lambda t: t.value)[:5]
    print(f"\n  TOP 5:")
    print(f"  {'#':<4} {'score':>7} {'delta':>7} {'wait_all':>9} {'wait':>7}"
          f" {'svc':>7} {'rej':>5}")
    for t in top5:
        a = t.user_attrs
        d = a.get("vs_greedy_ts_score", 0)
        print(f"  {t.number+1:<4} {t.value:>7.3f} {d:>+7.3f}"
              f" {a.get('rl_ts_mean_wait_all', 0):>9.2f}"
              f" {a.get('rl_ts_mean_wait', 0):>7.2f}"
              f" {a.get('rl_ts_service_rate', 0):>6.1%}"
              f" {a.get('rl_ts_rejected', 0):>5.0f}")

    print(f"\n  Saved config: {config_path}")
    print(f"  Saved CSV:    {csv_path}")
    print(f"\n  Next: python rl_train_from_tune_v5.py --config {config_path}")


if __name__ == "__main__":
    main()
