#!/usr/bin/env python3
# rl_tune_v4.py
# Optuna hyperparameter search — v4 "TS-initialiser" objective.
#
# =====================================================================
# THE CORE QUESTION CHANGE
# =====================================================================
#
# v1-v3 asked: "How can the RL agent get the lowest wait time?"
#
# v4 asks:     "How can the RL agent give Tabu Search a starting
#               solution that is easy to fix?"
#
# This is a fundamentally different objective.  A good TS initialiser:
#   1. Accepts most requests (TS can fix routing; it cannot un-reject)
#   2. Balances load across vehicles (gives TS room to move requests)
#   3. Avoids routes with tight feasibility margins (TS won't get stuck)
#   4. Keeps insertion cost moderate — not minimal (minimal cost often
#      means maximum tightness; TS prefers some slack to work with)
#
# =====================================================================
# WHY rl_tuned FAILED
# =====================================================================
#
# The v3 objective was:
#   score = mean_wait_all(MAX_WAIT=120) + 0.5*mean_ride + 0.05*p95_wait
#
# mean_wait_all penalises rejections by charging MAX_WAIT=120 per
# rejected request.  But it penalises a rejected request ONCE.
# A high w_rejection + high w_wait created a policy that learned:
#   "Reject borderline requests that would raise wait, accept easy ones"
# This gamed the metric — lowering mean_wait_served at the cost of
# service_rate dropping to 83.2% and 10 violations.
#
# rl_base has w_rejection=5.0 (v3 default) vs rl_tuned's w_rejection≈9+.
# rl_base: 87.7% service, 13.0 min wait.
# rl_tuned: 83.2% service, 12.4 min wait — worse on BOTH after all.
#
# =====================================================================
# V4 REWARD REDESIGN
# =====================================================================
#
# The reward now has THREE structural changes:
#
# 1. FLEET BALANCE BONUS (new)
#    After each insertion, reward the agent for balancing load across
#    vehicles.  Computed as: -w_imbalance * load_std_after_insertion
#    A balanced fleet gives TS more inter-vehicle move options because
#    no vehicle is overloaded and no vehicle is empty.
#    This is the key "easy to fix" signal.
#
# 2. PLAN SLACK BONUS (new)
#    Reward insertions that leave headroom before the new request's
#    pickup deadline.  Slack = (latest_pu - est_pu) / max_wait.
#    A plan with slack=0 is tightly packed; TS can barely reorder it.
#    A plan with slack=0.5 gives TS half the window to work with.
#    This directly incentivises "fixable" starting solutions.
#
# 3. REJECTION PENALTY REDESIGN
#    Old: flat -w_rejection per reject.
#    New: -w_rejection * (1 + n_rejected_so_far / n_requests)
#    The penalty GROWS as more requests are rejected.  This prevents
#    the agent from learning "reject the hard ones early" — rejections
#    become exponentially more costly as the episode progresses.
#    Inspired by the multi-objective DARP penalty shaping in
#    Gschwind & Drexl (2019).
#
# =====================================================================
# V4 TUNING OBJECTIVE (changed from v3)
# =====================================================================
#
# v3 objective (MINIMISE):
#   mean_wait_all + 0.5*mean_ride + 0.05*p95_wait
#
# v4 objective (MINIMISE) — evaluated on rl+ts, not standalone rl:
#   primary:  mean_wait_ts + 0.3*rejected_ts
#   where mean_wait_ts = mean wait of rl+ts combo (not standalone rl)
#         rejected_ts  = rejection count of rl+ts combo
#
# This is the key structural change: Optuna evaluates the COMBINATION
# of rl+ts, not rl alone.  A trial is good if the rl initialisation
# + ts improvement produces low wait and low rejection.  Standalone rl
# quality is irrelevant — we only care how well it sets TS up.
#
# =====================================================================
# SEARCH SPACE (6 parameters — simpler than v3's 7)
# =====================================================================
#
# REWARD WEIGHTS (3 searched):
#   w_rejection  3.0-8.0   LOWER than v3 (8-10) — reject-gaming fix
#   w_imbalance  0.1-1.0   NEW — fleet balance incentive
#   w_slack      0.1-1.0   NEW — plan headroom incentive
#
# FIXED reward weights (prevents the v3 overfitting vector):
#   w_wait       1.0       LOWER than v3 (1.5-2.8) — less wait obsession
#   w_ride       0.5       fixed
#   w_acceptance 1.0       fixed
#   w_ride_sq    0.0       removed — was complicating the trade-off
#   w_detour     0.0       removed — redundant with ride penalty
#   w_cost       0.1       low — we want balance, not cost minimisation
#
# PPO HYPERPARAMETERS (3 searched):
#   lr_start    1e-4–3e-4 (log)
#   gamma       0.990-0.996
#   ent_coef    0.01-0.05 (log)
#
# FIXED PPO (proven from v3):
#   lr_schedule  linear (v3 best trials all used linear)
#   n_steps      1024
#   n_epochs     5
#   batch_size   128
#   vf_coef      1.0
#   gae_lambda   0.95
#   clip_range   0.2
#   net_arch     [128, 128]
#   norm_obs     False
#   norm_reward  True
#
# =====================================================================
# EVALUATION CHANGE: measure rl+ts, not standalone rl
# =====================================================================
#
# Each trial trains an RL model, then evaluates it in two ways:
#   A. Standalone rl (as before) — for comparison only, not the score
#   B. rl+ts combo   — greedy+ts with RL vehicle assignment replacing
#                       greedy vehicle assignment
# The Optuna score is computed from B.
#
# =====================================================================
# Usage:
#   python rl_tune_v4.py                   # 25 trials (~6h with GPU)
#   python rl_tune_v4.py --resume          # safe resume
#   python rl_tune_v4.py --samples 5       # quick smoke test
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
MAX_WAIT         = 30.0    # minutes — actual constraint, not the v3 fake 120
GREEDY_TS_WAIT   = 7.41   # greedy+ts baseline from single-seed run
GREEDY_TS_SVC    = 0.892  # greedy+ts service rate baseline

# Tuning objective weights
OBJ_W_WAIT       = 1.0    # 1 point per minute of mean wait
OBJ_W_REJECTED   = 0.1    # 0.1 points per rejected passenger (flat)
#
# Calibration: 10 rejections costs the same as +1.0 min mean wait.
# This makes gaming unprofitable: rejecting 40 passengers to save
# 1.5 min wait costs +4.0 - 1.5 = +2.5 net (worse score).
# But Optuna can still see wait-time improvements: saving 1 min wait
# is worth accepting up to 10 extra rejections — a sensible trade-off.
#
# Why NOT Gemini's OBJ_W_REJECTED=1.0 (flat, one point per rejection):
# With coefficient 1.0, a single rejection (cost +1.0) outweighs
# cutting mean wait from 15 to 6 minutes (gain +9.0 only covers 9
# rejections). Optuna would ignore wait time entirely and minimise
# rejections above all else — flipping the bias rather than fixing it.


# ---------------------------------------------------------------------------
# Helper: evaluate a policy (standalone rl or rl+ts) over multiple episodes
# ---------------------------------------------------------------------------

def _evaluate_rl_standalone(model, cfg, n_episodes: int) -> dict:
    """Evaluate trained model running standalone (no TS improvement)."""
    from rl_env import DARPEnv
    from config import SimulationConfig

    results = []
    for i in range(n_episodes):
        eval_cfg = SimulationConfig(
            seed=2000 + i,
            fleet_size=cfg.fleet_size,
            vehicle_capacity=cfg.vehicle_capacity,
            depot_node=cfg.depot_node,
            n_requests=cfg.n_requests,
            demand_profile=cfg.demand_profile,
            stochastic_arrivals=cfg.stochastic_arrivals,
            travel_noise=0.0,
            n_nodes=cfg.n_nodes,
        )
        env = DARPEnv(cfg=eval_cfg)
        obs, _ = env.reset(seed=eval_cfg.seed)
        done = False
        total_reward = 0.0

        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        env._advance_vehicles_to(cfg.service_end + 500)
        s = env.episode_summary()
        s["reward"] = total_reward
        results.append(s)

    def _m(k):
        vals = [r[k] for r in results if r.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "service_rate": _m("service_rate"),
        "mean_wait":    _m("mean_wait"),
        "p95_wait":     _m("p95_wait"),
        "mean_ride":    _m("mean_ride"),
        "rejected":     _m("rejected"),
        "reward":       _m("reward"),
    }


def _evaluate_rl_plus_ts(model, cfg, n_episodes: int) -> dict:
    """
    Evaluate the rl+ts combination.

    Uses DARPEnv (the RL environment) for vehicle assignment,
    then applies TSPolicy as an improvement pass after each insertion.
    This mirrors exactly how rl+ts runs in the live simulation.

    The key difference from standalone eval: TS gets to fix the RL's
    routing decisions.  A good v4 model produces plans that TS can
    improve significantly.
    """
    from rl_env import DARPEnv
    from ts import TSPolicy
    from feasibility import check_feasibility, evaluate_plan
    from config import SimulationConfig
    import random as _random

    ts = TSPolicy(
        tabu_tenure         = 7,
        max_neighbours      = 50,
        iterations          = 200,
        patience            = 30,
        decision_time_limit = 0.3,
        rng                 = _random.Random(999),
    )

    results = []
    for i in range(n_episodes):
        eval_cfg = SimulationConfig(
            seed=2000 + i,
            fleet_size=cfg.fleet_size,
            vehicle_capacity=cfg.vehicle_capacity,
            depot_node=cfg.depot_node,
            n_requests=cfg.n_requests,
            demand_profile=cfg.demand_profile,
            stochastic_arrivals=cfg.stochastic_arrivals,
            travel_noise=0.0,
            n_nodes=cfg.n_nodes,
        )
        env = DARPEnv(cfg=eval_cfg)
        obs, _ = env.reset(seed=eval_cfg.seed)
        done = False
        total_reward = 0.0
        ts_improvements = 0

        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

            # TS improvement pass after each insertion (mirrors dispatcher)
            if not done:
                ts_system_state = {
                    **env._system_state,
                    "vehicles": {},
                }
                for vid, v in env._vehicles.items():
                    vs = v.to_state_dict(env._sim_time)
                    n_committed = 1 if v.in_transit_stop is not None else 0
                    from copy import deepcopy
                    ts_system_state["vehicles"][vid] = {
                        **vs,
                        "plan":        deepcopy(vs["plan_snapshot"]),
                        "n_committed": n_committed,
                    }

                changes = ts.propose(ts_system_state, check_feasibility, cfg.weights)

                if changes:
                    # Verify combined improvement before applying
                    total_before = sum(
                        evaluate_plan(
                            ts_system_state["vehicles"][vid]["plan"],
                            ts_system_state["vehicles"][vid],
                            env._system_state, cfg.weights,
                        )
                        for vid in changes
                    )
                    total_after = sum(
                        evaluate_plan(
                            new_plan,
                            ts_system_state["vehicles"][vid],
                            env._system_state, cfg.weights,
                        )
                        for vid, new_plan in changes.items()
                    )
                    if total_after < total_before:
                        for vid, new_plan in changes.items():
                            v = env._vehicles[vid]
                            n_committed = 1 if v.in_transit_stop is not None else 0
                            v.plan = new_plan[n_committed:]
                        ts_improvements += 1

        env._advance_vehicles_to(cfg.service_end + 500)
        s = env.episode_summary()
        s["reward"]          = total_reward
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
        "reward":          _m("reward"),
        "ts_improvements": _m("ts_improvements"),
    }


# ---------------------------------------------------------------------------
# Tuning objective — evaluated on rl+ts, not standalone rl
# ---------------------------------------------------------------------------

def compute_objective(rl_ts_metrics: dict, n_requests: int) -> tuple[float, dict]:
    """
    Score = OBJ_W_WAIT * mean_wait + OBJ_W_REJECTED * n_rejected

    Both terms are in comparable units: minutes of equivalent passenger
    discomfort.  The coefficient OBJ_W_REJECTED = 0.1 sets the exchange
    rate: 10 rejections costs the same as 1 extra minute of mean wait.

    This prevents rejection gaming (penalty too weak in v3/early v4)
    without making Optuna ignore wait time (Gemini's OBJ_W_REJECTED=1.0
    would make 1 rejection cost more than 9 minutes of wait reduction).
    """
    svc      = rl_ts_metrics.get("service_rate") or 0.0
    mw       = rl_ts_metrics.get("mean_wait")    or MAX_WAIT
    rej      = rl_ts_metrics.get("rejected")     or n_requests
    p95w     = rl_ts_metrics.get("p95_wait")     or MAX_WAIT
    mr       = rl_ts_metrics.get("mean_ride")    or 0.0
    ts_impr  = rl_ts_metrics.get("ts_improvements") or 0.0

    # Flat penalty per rejected passenger — scale-matched to wait time
    rej_penalty = OBJ_W_REJECTED * rej
    score       = OBJ_W_WAIT * mw + rej_penalty

    rej_rate = rej / max(n_requests, 1)

    return score, {
        "rl_ts_service_rate":    round(svc,     4),
        "rl_ts_mean_wait":       round(mw,      2),
        "rl_ts_p95_wait":        round(p95w,    2),
        "rl_ts_mean_ride":       round(mr,       2),
        "rl_ts_rejected":        round(rej,      1),
        "rl_ts_rejection_rate":  round(rej_rate, 4),
        "rl_ts_improvements":    round(ts_impr,  1),
        "rej_penalty":           round(rej_penalty, 3),
        "score":                 round(score,    3),
        "vs_greedy_ts_wait":     round(mw - GREEDY_TS_WAIT, 2),
        "vs_greedy_ts_svc":      round(svc - GREEDY_TS_SVC, 4),
    }


# ---------------------------------------------------------------------------
# Modified DARPEnv with v4 reward
# ---------------------------------------------------------------------------

class DARPEnvV4:
    """
    Thin wrapper around DARPEnv that replaces the reward function
    with the v4 TS-initialiser reward.

    Instead of subclassing (which would require duplicating __init__),
    we monkey-patch the reward method after construction.
    """

    @staticmethod
    def make(cfg, w_rejection, w_imbalance, w_slack):
        """
        Return a DARPEnv instance with the v4 reward patched in.

        v4 reward per accepted insertion:
          +1.0  acceptance
          -1.0  wait_penalty (normalised, fixed weight)
          -0.5  ride_penalty (normalised, fixed weight)
          -0.1  cost_penalty (low weight — we want balance, not min-cost)
          +w_imbalance * fleet_balance_bonus  (NEW: rewards even load)
          +w_slack     * plan_slack_bonus     (NEW: rewards headroom)

        Rejection penalty:
          -w_rejection * (1 + n_rejected / n_requests)  (GROWING penalty)
        """
        from rl_env import DARPEnv
        env = DARPEnv(
            cfg          = cfg,
            reward_mode  = "composite",
            w_acceptance = 1.0,
            w_wait       = 1.0,       # fixed low — less wait obsession
            w_ride       = 0.5,       # fixed
            w_ride_sq    = 0.0,       # removed
            w_detour     = 0.0,       # removed
            w_cost       = 0.1,       # low — balance > cost
            w_rejection  = w_rejection,
        )

        # Store v4-specific weights on the env
        env._v4_w_imbalance = w_imbalance
        env._v4_w_slack      = w_slack
        env._v4_n_requests   = cfg.n_requests

        # Patch the reward methods
        import types

        def _v4_compute_reward(self, req, old_cost, new_cost, est_wait, est_ride):
            direct   = max(req.direct_time or 1.0, 1.0)
            max_ride = direct * self.cfg.ride_factor
            norm     = direct * 5

            wait_pen  = -self.w_wait  * (est_wait / self.cfg.max_wait)
            ride_pen  = -self.w_ride  * (est_ride / max(max_ride, 1.0))
            cost_pen  = -self.w_cost  * ((new_cost - old_cost) / max(norm, 1.0))

            # Fleet balance bonus: reward lower load standard deviation
            # after this insertion.  Computed as negative std — lower std
            # means more balanced, so bonus = -std (higher when balanced).
            loads = [len(v.onboard) for v in self._vehicles.values()]
            load_std = float(np.std(loads)) if loads else 0.0
            balance_bonus = self._v4_w_imbalance * (1.0 - min(load_std / max(self.cfg.vehicle_capacity, 1), 1.0))

            # Plan slack bonus: reward insertions that leave the new
            # passenger's pickup window with headroom.
            # slack = (latest_pu - est_pu) / max_wait, clamped to [0,1].
            latest_pu = req.request_time + self.cfg.max_wait
            slack = max(0.0, latest_pu - (req.request_time + est_wait))
            slack_norm = min(slack / max(self.cfg.max_wait, 1.0), 1.0)
            slack_bonus = self._v4_w_slack * slack_norm

            return (self.w_acceptance
                    + wait_pen
                    + ride_pen
                    + cost_pen
                    + balance_bonus
                    + slack_bonus)

        def _v4_rejection_penalty(self, req):
            # Growing rejection penalty — becomes costlier as rejections accumulate.
            # This prevents "reject the hard ones early" gaming.
            n_rej_so_far = self._n_rejections  # already incremented before call
            growth = 1.0 + (n_rej_so_far / max(self._v4_n_requests, 1))
            return -self.w_rejection * growth

        env._compute_reward    = types.MethodType(_v4_compute_reward,    env)
        env._rejection_penalty = types.MethodType(_v4_rejection_penalty, env)

        return env


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

    # ---- Searched parameters ----
    w_rejection = trial.suggest_float("w_rejection", 3.0, 8.0)
    w_imbalance = trial.suggest_float("w_imbalance", 0.1, 1.0)
    w_slack     = trial.suggest_float("w_slack",     0.1, 1.0)
    lr_start    = trial.suggest_float("lr_start",    1e-4, 3e-4, log=True)
    gamma       = trial.suggest_float("gamma",       0.990, 0.996)
    ent_coef    = trial.suggest_float("ent_coef",    0.01,  0.05, log=True)

    # ---- Fixed parameters ----
    lr_schedule = "linear"
    n_steps     = 1024
    n_epochs    = 5
    batch_size  = 128

    learning_rate = lambda progress: lr_start * progress

    print(f"\n  --- Trial {trial.number + 1} ---")
    print(f"    Reward: w_rej={w_rejection:.2f}  "
          f"w_imbal={w_imbalance:.2f}  w_slack={w_slack:.2f}")
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
                depot_node=cfg.depot_node,
                n_requests=cfg.n_requests,
                demand_profile=cfg.demand_profile,
                stochastic_arrivals=cfg.stochastic_arrivals,
                travel_noise=0.0,
                n_nodes=cfg.n_nodes,
            )
            return DARPEnvV4.make(env_cfg, w_rejection, w_imbalance, w_slack)
        return _init

    vec_env = SubprocVecEnv([make_env(42 + i) for i in range(n_envs)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
    )

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    tb_log_dir = os.path.join(tb_base, f"trial_{trial.number + 1:03d}")

    model = MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate = learning_rate,
        gamma         = gamma,
        ent_coef      = ent_coef,
        n_steps       = n_steps,
        batch_size    = batch_size,
        n_epochs      = n_epochs,
        vf_coef       = 1.0,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        max_grad_norm = 0.5,
        policy_kwargs = dict(net_arch=[128, 128]),
        verbose       = 0,
        tensorboard_log = tb_log_dir,
        device        = device,
    )

    # Timeout: 25 min per trial (slightly more than v3 — eval is heavier)
    t0 = time.time()
    TRIAL_TIMEOUT = 25 * 60

    class TimeoutCallback(BaseCallback):
        def __init__(self): super().__init__(verbose=0)
        def _on_step(self) -> bool:
            return time.time() - t0 < TRIAL_TIMEOUT

    model.learn(
        total_timesteps = timesteps,
        callback        = CallbackList([TimeoutCallback()]),
    )
    train_time = time.time() - t0

    # ---- Evaluation ----
    # 1. Standalone rl — logged but NOT the objective
    print(f"    Evaluating standalone rl (5 episodes)...")
    standalone = _evaluate_rl_standalone(model, cfg, n_episodes=5)
    print(f"    Standalone: svc={standalone.get('service_rate', 0):.1%}"
          f"  wait={standalone.get('mean_wait', 0):.2f}"
          f"  rej={standalone.get('rejected', 0):.0f}")

    # 2. rl+ts — THIS IS THE OBJECTIVE
    print(f"    Evaluating rl+ts combo (10 episodes)...")
    rl_ts = _evaluate_rl_plus_ts(model, cfg, n_episodes=10)
    print(f"    rl+ts:      svc={rl_ts.get('service_rate', 0):.1%}"
          f"  wait={rl_ts.get('mean_wait', 0):.2f}"
          f"  rej={rl_ts.get('rejected', 0):.0f}"
          f"  ts_impr={rl_ts.get('ts_improvements', 0):.1f}")

    vec_env.close()

    score, metrics = compute_objective(rl_ts, cfg.n_requests)

    # Log everything for analysis
    for k, v in metrics.items():
        if v is not None:
            trial.set_user_attr(k, v)
    for k, v in standalone.items():
        if v is not None:
            trial.set_user_attr(f"standalone_{k}", round(v, 4) if isinstance(v, float) else v)
    trial.set_user_attr("train_time_s",   round(train_time, 1))
    trial.set_user_attr("vs_greedy_ts",   round(metrics.get("vs_greedy_ts_wait", 0), 2))

    print(f"    Score: {score:.3f}  "
          f"(greedy+ts baseline: {GREEDY_TS_WAIT:.2f}  "
          f"delta: {metrics.get('vs_greedy_ts_wait', 0):+.2f})")

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="rl_tune_v4 — TS-initialiser objective"
    )
    parser.add_argument("--samples",    type=int, default=25,
                        help="Number of Optuna trials (default: 25)")
    parser.add_argument("--timesteps",  type=int, default=300_000,
                        help="PPO timesteps per trial (default: 300k)")
    parser.add_argument("--n-envs",     type=int, default=6,
                        help="Parallel training envs (default: 6)")
    parser.add_argument("--output-dir", default="rl_outputs/tune_v4")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume existing study")
    parser.add_argument("--study-name", default="darp_ppo_v4_ts_init")
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
            study_name = args.study_name,
            direction  = "minimize",
            sampler    = TPESampler(seed=789, n_startup_trials=8),
        )
        # Seed with sensible starting points
        # Trial 1: moderate rejection penalty, balanced incentives
        study.enqueue_trial({
            "w_rejection": 5.0, "w_imbalance": 0.5, "w_slack": 0.3,
            "lr_start": 2e-4, "gamma": 0.993, "ent_coef": 0.02,
        })
        # Trial 2: low rejection penalty (anti-gaming), high balance focus
        study.enqueue_trial({
            "w_rejection": 4.0, "w_imbalance": 0.8, "w_slack": 0.5,
            "lr_start": 2.5e-4, "gamma": 0.992, "ent_coef": 0.025,
        })
        # Trial 3: v3-like rejection penalty but with new balance/slack terms
        study.enqueue_trial({
            "w_rejection": 7.0, "w_imbalance": 0.3, "w_slack": 0.2,
            "lr_start": 1.5e-4, "gamma": 0.995, "ent_coef": 0.015,
        })
        print(f"New study: {args.study_name} (3 seed trials)")

    def trial_callback(study, trial):
        attrs = trial.user_attrs
        print(f"\n  Trial {trial.number + 1} result:")
        print(f"    score:            {trial.value:.3f}")
        print(f"    rl+ts svc:        {attrs.get('rl_ts_service_rate', 0):.1%}")
        print(f"    rl+ts mean_wait:  {attrs.get('rl_ts_mean_wait', 0):.2f} min")
        print(f"    rl+ts rejected:   {attrs.get('rl_ts_rejected', 0):.1f}")
        print(f"    rl+ts ts_improv:  {attrs.get('rl_ts_improvements', 0):.1f}")
        print(f"    standalone svc:   {attrs.get('standalone_service_rate', 0):.1%}")
        print(f"    standalone wait:  {attrs.get('standalone_mean_wait', 0):.2f} min")
        print(f"    vs greedy+ts:     {attrs.get('vs_greedy_ts_wait', 0):+.2f} min")

        valid = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value != float("inf")]
        if valid:
            b = min(valid, key=lambda t: t.value)
            print(f"  >>> Best: trial {b.number + 1}"
                  f"  score={b.value:.3f}"
                  f"  rl+ts_wait={b.user_attrs.get('rl_ts_mean_wait', 0):.2f}"
                  f"  svc={b.user_attrs.get('rl_ts_service_rate', 0):.1%}")

        with open(study_path, "wb") as f:
            pickle.dump(study, f)

    completed   = [t for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE]
    n_remaining = args.samples - len(completed)

    if n_remaining <= 0:
        print(f"All {args.samples} trials already complete.")
        return

    print("=" * 60)
    print("rl_tune_v4 — TS-Initialiser Objective")
    print("=" * 60)
    print(f"  Question: 'How can RL give TS a better start?'")
    print(f"  Objective: minimise rl+ts mean_wait (not standalone rl)")
    print(f"  Trials:    {args.samples} ({n_remaining} remaining)")
    print(f"  Timesteps: {args.timesteps:,} per trial")
    print(f"  Envs:      {args.n_envs}")
    print()
    print(f"  Searched (6 params):")
    print(f"    w_rejection(3.0-8.0)  [lower than v3 to prevent gaming]")
    print(f"    w_imbalance(0.1-1.0)  [NEW: fleet balance bonus]")
    print(f"    w_slack(0.1-1.0)      [NEW: plan headroom bonus]")
    print(f"    lr(1e-4..3e-4) gamma(0.990-0.996) ent(0.01-0.05)")
    print(f"  Fixed reward:")
    print(f"    w_wait=1.0 w_ride=0.5 w_cost=0.1")
    print(f"    w_ride_sq=0 w_detour=0 w_acceptance=1.0")
    print(f"  Fixed PPO:")
    print(f"    lr=linear n_steps=1024 epochs=5 batch=128")
    print(f"    vf=1.0 lam=0.95 clip=0.2 arch=[128,128]")
    print(f"    norm_obs=False norm_reward=True")
    print()
    print(f"  Baselines:")
    print(f"    greedy+ts: wait={GREEDY_TS_WAIT:.2f}  svc={GREEDY_TS_SVC:.1%}")
    print(f"  Target: score < {GREEDY_TS_WAIT:.2f}")
    print(f"  Safe to Ctrl+C — study saved after every trial.")
    print("=" * 60)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    t_start = time.time()
    study.optimize(
        lambda trial: run_trial(trial, args.timesteps, args.n_envs, tb_base),
        n_trials          = n_remaining,
        callbacks         = [trial_callback],
        show_progress_bar = False,
    )
    total_time = time.time() - t_start

    # ---- Results ----
    valid = [t for t in study.trials
             if t.state == optuna.trial.TrialState.COMPLETE
             and t.value != float("inf")]

    if not valid:
        print("\nNo valid trials completed.")
        return

    best        = min(valid, key=lambda t: t.value)
    best_params = best.params
    best_attrs  = best.user_attrs

    print("\n" + "=" * 60)
    print("SEARCH v4 COMPLETE")
    print("=" * 60)
    print(f"  Time:   {total_time/3600:.1f} hours")
    print(f"  Trials: {len(valid)} valid / {len(study.trials)} total")
    print()
    print(f"  Best trial #{best.number + 1}:")
    print(f"    score:           {best.value:.3f}")
    print(f"    rl+ts svc:       {best_attrs.get('rl_ts_service_rate', 0):.1%}")
    print(f"    rl+ts mean_wait: {best_attrs.get('rl_ts_mean_wait', 0):.2f} min")
    print(f"    rl+ts rejected:  {best_attrs.get('rl_ts_rejected', 0):.1f}")
    print(f"    vs greedy+ts:    {best_attrs.get('vs_greedy_ts_wait', 0):+.2f} min")
    print(f"    standalone svc:  {best_attrs.get('standalone_service_rate', 0):.1%}")
    print(f"    standalone wait: {best_attrs.get('standalone_mean_wait', 0):.2f} min")
    print()
    print(f"    w_rejection={best_params.get('w_rejection', 0):.2f}  "
          f"w_imbalance={best_params.get('w_imbalance', 0):.2f}  "
          f"w_slack={best_params.get('w_slack', 0):.2f}")
    print(f"    lr={best_params.get('lr_start', 0):.5f}(linear)  "
          f"gamma={best_params.get('gamma', 0):.4f}  "
          f"ent={best_params.get('ent_coef', 0):.4f}")

    # Save best_config.json — compatible with rl_train_from_tune.py
    best_config = {
        "source":       "rl_tune_v4.py",
        "objective":    "rl+ts combo wait time (TS-initialiser)",
        "reward_mode":  "composite",
        # v4 reward weights
        "w_acceptance": 1.0,
        "w_wait":       1.0,
        "w_ride":       0.5,
        "w_ride_sq":    0.0,
        "w_detour":     0.0,
        "w_cost":       0.1,
        "w_rejection":  best_params.get("w_rejection"),
        "w_imbalance":  best_params.get("w_imbalance"),
        "w_slack":      best_params.get("w_slack"),
        # PPO
        "lr_start":     best_params.get("lr_start"),
        "lr_schedule":  "linear",
        "gamma":        best_params.get("gamma"),
        "ent_coef":     best_params.get("ent_coef"),
        "n_steps":      1024,
        "n_epochs":     5,
        "batch_size":   128,
        "vf_coef":      1.0,
        "gae_lambda":   0.95,
        "clip_range":   0.2,
        "max_grad_norm": 0.5,
        "net_arch":     [128, 128],
        "norm_obs":     False,
        "norm_reward":  True,
        "timesteps":    args.timesteps,
        "n_envs":       args.n_envs,
        "best_trial_number": best.number + 1,
        "n_trials":          len(study.trials),
        "n_valid_trials":    len(valid),
        "best_score":        round(best.value, 4),
        "achieved_rl_ts_service_rate": best_attrs.get("rl_ts_service_rate"),
        "achieved_rl_ts_mean_wait":    best_attrs.get("rl_ts_mean_wait"),
        "achieved_rl_ts_rejected":     best_attrs.get("rl_ts_rejected"),
        "achieved_standalone_svc":     best_attrs.get("standalone_service_rate"),
        "achieved_standalone_wait":    best_attrs.get("standalone_mean_wait"),
        "greedy_ts_baseline_wait":     GREEDY_TS_WAIT,
        "greedy_ts_baseline_svc":      GREEDY_TS_SVC,
    }
    config_path = os.path.join(args.output_dir, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n  Saved config: {config_path}")

    # Save all_trials.csv
    csv_path   = os.path.join(args.output_dir, "all_trials.csv")
    fieldnames = [
        "trial", "score",
        "rl_ts_service_rate", "rl_ts_mean_wait", "rl_ts_p95_wait",
        "rl_ts_mean_ride", "rl_ts_rejected", "rl_ts_improvements",
        "vs_greedy_ts_wait", "vs_greedy_ts_svc",
        "standalone_service_rate", "standalone_mean_wait",
        "w_rejection", "w_imbalance", "w_slack",
        "lr_start", "gamma", "ent_coef", "train_time_s",
    ]
    all_complete = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in sorted(all_complete, key=lambda x: x.value
                        if x.value != float("inf") else 999):
            row = {"trial": t.number + 1, "score": round(t.value, 4)}
            row.update(t.user_attrs)
            row.update(t.params)
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"  Saved CSV:    {csv_path}")

    # Top 5
    top5 = sorted(valid, key=lambda t: t.value)[:5]
    print(f"\n  TOP 5 (by rl+ts score):")
    print(f"  {'#':<4} {'score':>7} {'rl+ts_wait':>11} {'rl+ts_svc':>10}"
          f" {'rl_svc':>7} {'rl_wait':>8} {'rej':>4}")
    for t in top5:
        a = t.user_attrs
        print(f"  {t.number+1:<4} {t.value:>7.3f}"
              f" {a.get('rl_ts_mean_wait', 0):>10.2f}"
              f" {a.get('rl_ts_service_rate', 0):>9.1%}"
              f" {a.get('standalone_service_rate', 0):>6.1%}"
              f" {a.get('standalone_mean_wait', 0):>8.2f}"
              f" {a.get('rl_ts_rejected', 0):>4.0f}")

    print(f"\n  Next step: python rl_train_from_tune.py --config {config_path}")
    print(f"  Then run:  python main.py --policy rl+ts --model <trained_model>")


if __name__ == "__main__":
    main()
