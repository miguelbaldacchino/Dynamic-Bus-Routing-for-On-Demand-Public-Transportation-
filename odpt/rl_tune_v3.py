#!/usr/bin/env python3
# rl_tune_v3.py
# Optuna hyperparameter search — v3 (lean, informed by v1+v2 findings).
#
# ===================================================================
# PROGRESSION: v1 → v2 → v3
# ===================================================================
#
# v1: 10 search dims, 27 trials. Discovered rejection gaming, objective
#     instability (4 patches), norm_obs train/eval mismatch.
#     Best: trial 5, score=25.73, wait_svd=10.10, svc=88.8%.
#
# v2: 12 search dims, 38 trials. Fixed norm_obs, added 4 anticipatory
#     obs features (78 dims), expanded LR schedules and architectures.
#     TPE couldn't converge — too many dimensions. Couldn't isolate
#     which changes helped. Best: trial 27, score=29.07, wait_svd=10.38.
#
# v3: 7 search dims, 30 trials. Keeps the proven norm_obs fix.
#     Reverts to 74-dim obs (anticipatory features behind flag for v4).
#     Tight search ranges informed by v1+v2 data. Goal: find best
#     PPO config with clean methodology before adding complexity.
#
# ===================================================================
# CHANGES FROM v2
# ===================================================================
#
# 1. Obs dims 78 → 74 (USE_ANTICIPATORY_FEATURES=False in rl_env.py)
# 2. Search dims 12 → 7 (fixed: arch, n_steps, n_epochs, w_detour)
# 3. Tighter ranges informed by v1+v2 best trials
# 4. 30 trials (was 50) — sufficient for 7 dims
# 5. Eval callback every 100k (was 50k) — 3 episodes (was 5)
#
# ===================================================================
# KEPT FROM v2
# ===================================================================
#
# - norm_obs=False (the actual bug fix — halved approx_kl)
# - norm_reward=True (run005 fix — enables critic learning)
# - Objective: mean_wait_all(MAX_WAIT=120) + 0.5*mean_ride + 0.05*p95_wait
# - All thesis TensorBoard metrics (peak/offpeak, load_std)
# - n_envs=6, eval seeds 2000+, in-progress flush
#
# ===================================================================
# SEARCH SPACE (7 parameters)
# ===================================================================
#
# REWARD WEIGHTS (3 searched, 4 fixed):
#   w_wait      1.5-2.8   v2 best 2.36, v1 best 1.84
#   w_ride      0.6-1.4   v2 best 0.87, v1 best 1.31
#   w_rejection 8.0-10.0  every good trial in v2 had w_rej > 8
#   w_detour    1.0       fixed — inconsistent across v1/v2
#   w_ride_sq   0.3       fixed
#   w_cost      0.2       fixed
#   w_acceptance 1.0      fixed
#
# PPO HYPERPARAMETERS (4 searched, 6 fixed):
#   lr_start    1e-4–4e-4 (log)   trials below 1e-4 were disasters
#   lr_schedule constant/linear   both proven
#   gamma       0.990-0.996       v2 best 0.9925
#   ent_coef    0.01-0.05 (log)   TPE pushed toward high entropy
#   n_steps     1024               fixed — all good trials
#   n_epochs    5                  fixed — all good trials
#   batch_size  128                fixed
#   vf_coef     1.0                fixed
#   gae_lambda  0.95               fixed
#   clip_range  0.2                fixed
#   net_arch    [128, 128]         fixed — proven at 74 dims
#
# ===================================================================
# Usage:
#   python rl_tune_v3.py                   # 30 trials (~5h)
#   python rl_tune_v3.py --resume          # continue safely
#   python rl_tune_v3.py --samples 10      # quick test
# ===================================================================

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
# Objective constants
# ---------------------------------------------------------------------------
OBJ_WEIGHT_RIDE = 0.5
OBJ_WEIGHT_P95  = 0.05
MAX_WAIT         = 120.0

GREEDY_WAIT    = 8.3       # approximate — update after multi-seed greedy run
GREEDY_SERVICE = 0.885


# ---------------------------------------------------------------------------
# Peak hour definitions (sim time, t=0 is 05:30)
# ---------------------------------------------------------------------------
PEAK_WINDOWS = [(90, 210), (570, 750)]


def _is_peak(t: float) -> bool:
    return any(lo <= t < hi for lo, hi in PEAK_WINDOWS)


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def compute_objective(results: list[dict], n_requests: int) -> tuple[float, dict]:
    service_rate     = float(np.mean([r["service_rate"] for r in results]))
    rejected         = float(np.mean([r["rejected"]     for r in results]))
    mean_ride        = float(np.mean([r["mean_ride"]    for r in results if r.get("mean_ride")]))
    mean_wait_served = float(np.mean([r["mean_wait"]    for r in results if r.get("mean_wait")]))
    p95_wait         = float(np.mean([r["p95_wait"]     for r in results if r.get("p95_wait")]))
    p95_ride         = float(np.mean([r["p95_ride"]     for r in results if r.get("p95_ride")]))
    mean_reward      = float(np.mean([r["reward"]       for r in results]))
    mean_detour      = float(np.mean([r["mean_detour"]  for r in results if r.get("mean_detour")]))

    n_served   = service_rate * n_requests
    n_rejected = rejected
    if n_served + n_rejected > 0:
        mean_wait_all = ((n_served * mean_wait_served) + (n_rejected * MAX_WAIT)) \
                        / (n_served + n_rejected)
    else:
        mean_wait_all = MAX_WAIT

    score = (
        mean_wait_all
        + OBJ_WEIGHT_RIDE * mean_ride
        + OBJ_WEIGHT_P95  * p95_wait
    )

    return score, {
        "service_rate":      round(service_rate, 4),
        "mean_wait_all":     round(mean_wait_all, 2),
        "mean_wait_served":  round(mean_wait_served, 2),
        "mean_ride":         round(mean_ride, 2),
        "p95_wait":          round(p95_wait, 2),
        "p95_ride":          round(p95_ride, 2),
        "mean_detour":       round(mean_detour, 3),
        "rejected":          round(rejected, 1),
        "rejection_rate":    round(rejected / max(n_requests, 1), 4),
        "mean_reward":       round(mean_reward, 2),
        "score":             round(score, 3),
    }


# ---------------------------------------------------------------------------
# Evaluation — with thesis metrics
# ---------------------------------------------------------------------------

def evaluate_policy(model, cfg, n_episodes: int = 10) -> tuple[list[dict], dict]:
    from rl_env import DARPEnv
    from config import SimulationConfig

    results = []
    for i in range(n_episodes):
        eval_cfg = SimulationConfig(
            seed=1000 + i,
            fleet_size=cfg.fleet_size,
            vehicle_capacity=cfg.vehicle_capacity,
            depot_node=cfg.depot_node,
            n_requests=cfg.n_requests,
            demand_profile=cfg.demand_profile,
            stochastic_arrivals=cfg.stochastic_arrivals,
            travel_noise=0.0,
            n_nodes=cfg.n_nodes,
        )
        env  = DARPEnv(cfg=eval_cfg)
        obs, _ = env.reset(seed=eval_cfg.seed)
        done   = False
        total_reward = 0.0
        load_samples = []

        for _ in range(cfg.n_requests * 2):
            if done:
                break
            loads = [len(v.onboard) for v in env._vehicles.values()]
            if any(l > 0 for l in loads):
                load_samples.append(float(np.std(loads)))

            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        # Flush in-progress
        env._advance_vehicles_to(cfg.service_end + 500)

        summary = env.episode_summary()
        summary["reward"] = total_reward

        # Detour
        reqs = env._requests
        detours = []
        for r in reqs.values():
            if (r.pickup_time is not None and r.dropoff_time is not None
                    and r.direct_time and r.direct_time > 0):
                detours.append((r.dropoff_time - r.pickup_time) / r.direct_time)
        summary["mean_detour"] = float(np.mean(detours)) if detours else None

        # Peak vs off-peak
        peak_waits, offpeak_waits = [], []
        peak_total = peak_rejected = offpeak_total = offpeak_rejected = 0

        for r in reqs.values():
            if _is_peak(r.request_time):
                peak_total += 1
                if r.status == "REJECTED":
                    peak_rejected += 1
                elif r.pickup_time is not None:
                    peak_waits.append(r.pickup_time - r.request_time)
            else:
                offpeak_total += 1
                if r.status == "REJECTED":
                    offpeak_rejected += 1
                elif r.pickup_time is not None:
                    offpeak_waits.append(r.pickup_time - r.request_time)

        summary["peak_mean_wait"] = float(np.mean(peak_waits)) if peak_waits else None
        summary["peak_rejection_rate"] = peak_rejected / max(peak_total, 1)
        summary["offpeak_mean_wait"] = float(np.mean(offpeak_waits)) if offpeak_waits else None
        summary["offpeak_rejection_rate"] = offpeak_rejected / max(offpeak_total, 1)
        summary["load_std"] = float(np.mean(load_samples)) if load_samples else 0.0

        results.append(summary)

    def _mean(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return results, {
        "service_rate":           _mean("service_rate"),
        "mean_wait":              _mean("mean_wait"),
        "p95_wait":               _mean("p95_wait"),
        "mean_ride":              _mean("mean_ride"),
        "p95_ride":               _mean("p95_ride"),
        "mean_detour":            _mean("mean_detour"),
        "rejected":               _mean("rejected"),
        "reward":                 _mean("reward"),
        "peak_mean_wait":         _mean("peak_mean_wait"),
        "peak_rejection_rate":    _mean("peak_rejection_rate"),
        "offpeak_mean_wait":      _mean("offpeak_mean_wait"),
        "offpeak_rejection_rate": _mean("offpeak_rejection_rate"),
        "load_std":               _mean("load_std"),
    }


# ---------------------------------------------------------------------------
# TensorBoard eval callback
# ---------------------------------------------------------------------------

def make_eval_callback(cfg, eval_freq: int):
    from stable_baselines3.common.callbacks import BaseCallback

    class DARPEvalCallback(BaseCallback):
        def __init__(self, cfg, eval_freq):
            super().__init__(verbose=0)
            self.cfg       = cfg
            self.eval_freq = eval_freq

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                _, m = evaluate_policy(self.model, self.cfg, n_episodes=3)

                sr  = m["service_rate"] or 0
                rej = m["rejected"] or 0
                mw  = m["mean_wait"] or 0
                n   = self.cfg.n_requests
                mw_all = ((sr * n * mw) + (rej * MAX_WAIT)) / max(sr * n + rej, 1)

                self.logger.record("eval/service_rate",            sr)
                self.logger.record("eval/mean_wait",               mw)
                self.logger.record("eval/mean_wait_all",           mw_all)
                self.logger.record("eval/p95_wait",                m["p95_wait"]              or 0)
                self.logger.record("eval/mean_ride",               m["mean_ride"]             or 0)
                self.logger.record("eval/p95_ride",                m["p95_ride"]              or 0)
                self.logger.record("eval/mean_detour",             m["mean_detour"]           or 0)
                self.logger.record("eval/rejected",                rej)
                self.logger.record("eval/rejection_rate",          rej / max(n, 1))
                self.logger.record("eval/peak_mean_wait",          m["peak_mean_wait"]        or 0)
                self.logger.record("eval/peak_rejection_rate",     m["peak_rejection_rate"]   or 0)
                self.logger.record("eval/offpeak_mean_wait",       m["offpeak_mean_wait"]     or 0)
                self.logger.record("eval/offpeak_rejection_rate",  m["offpeak_rejection_rate"]or 0)
                self.logger.record("eval/load_std",                m["load_std"]              or 0)
                self.logger.dump(self.num_timesteps)

            return True

    return DARPEvalCallback(cfg, eval_freq)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(trial: optuna.Trial, timesteps: int, n_envs: int,
              tb_base: str) -> float:
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from config import SimulationConfig
    from rl_env import DARPEnv

    # ==================================================================
    # Searched reward weights (3)
    # ==================================================================
    w_wait      = trial.suggest_float("w_wait",      1.5, 2.8)
    w_ride      = trial.suggest_float("w_ride",      0.6, 1.4)
    w_rejection = trial.suggest_float("w_rejection", 8.0, 10.0)

    # Fixed reward weights
    w_detour     = 1.0
    w_ride_sq    = 0.3
    w_cost       = 0.2
    w_acceptance = 1.0

    # ==================================================================
    # Searched PPO hyperparameters (4)
    # ==================================================================
    lr_start    = trial.suggest_float("lr_start", 1e-4, 4e-4, log=True)
    lr_schedule = trial.suggest_categorical("lr_schedule", ["constant", "linear"])
    gamma       = trial.suggest_float("gamma",    0.990, 0.996)
    ent_coef    = trial.suggest_float("ent_coef", 0.01,  0.05, log=True)

    # Fixed PPO
    n_steps    = 1024
    n_epochs   = 5
    batch_size = 128
    vf_coef    = 1.0
    gae_lambda = 0.95
    clip_range = 0.2
    net_arch   = [128, 128]

    # LR schedule
    if lr_schedule == "linear":
        learning_rate = lambda progress: lr_start * progress
    else:
        learning_rate = lr_start

    print(f"\n  --- Trial {trial.number + 1} ---")
    print(f"    Reward:  w_wait={w_wait:.2f}  w_ride={w_ride:.2f}"
          f"  w_rej={w_rejection:.2f}")
    print(f"    PPO:     lr={lr_start:.5f}({lr_schedule})"
          f"  gamma={gamma:.4f}  ent={ent_coef:.4f}")

    # ==================================================================
    # Simulation config
    # ==================================================================
    cfg = SimulationConfig(
        seed=42, fleet_size=6, vehicle_capacity=16,
        depot_node=0, n_requests=400, demand_profile="malta",
        stochastic_arrivals=True, travel_noise=0.0, n_nodes=71,
    )

    # ==================================================================
    # Parallel environments
    # ==================================================================
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
            return DARPEnv(
                cfg          = env_cfg,
                reward_mode  = "composite",
                w_acceptance = w_acceptance,
                w_wait       = w_wait,
                w_ride       = w_ride,
                w_ride_sq    = w_ride_sq,
                w_detour     = w_detour,
                w_cost       = w_cost,
                w_rejection  = w_rejection,
            )
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

    # ==================================================================
    # Model
    # ==================================================================
    model = MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate   = learning_rate,
        gamma           = gamma,
        ent_coef        = ent_coef,
        n_steps         = n_steps,
        batch_size      = batch_size,
        n_epochs        = n_epochs,
        vf_coef         = vf_coef,
        gae_lambda      = gae_lambda,
        clip_range      = clip_range,
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(net_arch=net_arch),
        verbose         = 0,
        tensorboard_log = tb_log_dir,
        device          = device,
    )

    # Eval callback: every 100k steps, 3 episodes (lightweight)
    eval_cb = make_eval_callback(cfg, eval_freq=100_000 // n_envs)

    # Timeout: 20 min max per trial
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    t0 = time.time()
    TRIAL_TIMEOUT = 20 * 60

    class TimeoutCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
        def _on_step(self) -> bool:
            if time.time() - t0 > TRIAL_TIMEOUT:
                print(f"\n  [TIMEOUT] Trial {trial.number + 1}")
                return False
            return True

    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList([eval_cb, TimeoutCallback()]),
    )
    train_time = time.time() - t0

    # ==================================================================
    # Final evaluation — 10 episodes
    # ==================================================================
    results, _ = evaluate_policy(model, cfg, n_episodes=10)
    vec_env.close()

    score, metrics = compute_objective(results, cfg.n_requests)

    for k, v in metrics.items():
        if v is not None:
            trial.set_user_attr(k, v)
    trial.set_user_attr("train_time_s", round(train_time, 1))

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optuna HPO v3 — lean search, norm_obs fix, 74-dim obs"
    )
    parser.add_argument("--samples",    type=int, default=30)
    parser.add_argument("--timesteps",  type=int, default=320_000)
    parser.add_argument("--n-envs",     type=int, default=6)
    parser.add_argument("--output-dir", default="rl_outputs/tune_v3")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--study-name", default="darp_ppo_v3")
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
        print(f"Resuming: {completed} trials already complete")
    else:
        study = optuna.create_study(
            study_name = args.study_name,
            direction  = "minimize",
            sampler    = TPESampler(seed=456, n_startup_trials=10),
        )
        # Seed TPE with best configs from v1+v2 (evaluated fresh on v3 seeds).
        # These run first, giving TPE informed starting points instead of
        # pure random exploration. Values are clipped to v3 search ranges.
        study.enqueue_trial({
            # v1 trial 5 — best overall in v1 (score=25.73 on v1 seeds)
            "w_wait": 1.84, "w_ride": 1.31, "w_rejection": 8.62,
            "lr_start": 1.1e-4, "lr_schedule": "constant",
            "gamma": 0.993, "ent_coef": 0.010,
        })
        study.enqueue_trial({
            # v2 trial 27 — best in v2 (score=29.07 on v2 seeds)
            "w_wait": 2.36, "w_ride": 0.87, "w_rejection": 9.59,
            "lr_start": 3.4e-4, "lr_schedule": "linear",
            "gamma": 0.993, "ent_coef": 0.042,
        })
        study.enqueue_trial({
            # v2 trial 4 — good balance of wait + service rate
            "w_wait": 2.34, "w_ride": 1.09, "w_rejection": 8.37,
            "lr_start": 3.5e-4, "lr_schedule": "linear",
            "gamma": 0.995, "ent_coef": 0.01,  # v2 used 0.006, clipped to v3 floor
        })
        print(f"New study: {args.study_name} (3 seed trials from v1+v2)")

    def trial_callback(study: optuna.Study, trial: optuna.Trial):
        attrs = trial.user_attrs
        print(f"\n  Trial {trial.number + 1} result:")
        print(f"    score:          {trial.value:.3f}")
        print(f"    service_rate:   {attrs.get('service_rate', 0):.1%}")
        print(f"    mean_wait_all:  {attrs.get('mean_wait_all', 0):.2f} min")
        print(f"    mean_wait_svd:  {attrs.get('mean_wait_served', 0):.2f} min")
        print(f"    mean_ride:      {attrs.get('mean_ride', 0):.2f} min")
        print(f"    p95_wait:       {attrs.get('p95_wait', 0):.2f} min")
        print(f"    rejected:       {attrs.get('rejected', 0):.1f}")
        print(f"    train_time:     {attrs.get('train_time_s', 0):.0f}s")

        valid = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value != float("inf")]
        if valid:
            b = min(valid, key=lambda t: t.value)
            print(f"  >>> Best: trial {b.number + 1}"
                  f"  score={b.value:.3f}"
                  f"  wait_all={b.user_attrs.get('mean_wait_all', 0):.2f}"
                  f"  svc={b.user_attrs.get('service_rate', 0):.1%}")

        with open(study_path, "wb") as f:
            pickle.dump(study, f)

    completed   = [t for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE]
    n_remaining = args.samples - len(completed)

    if n_remaining <= 0:
        print(f"All {args.samples} trials already complete.")
        return

    print("=" * 60)
    print("Optuna v3 — Lean Search, Proven Fixes Only")
    print("=" * 60)
    print(f"  Trials:    {args.samples} ({n_remaining} remaining)")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Envs:      {args.n_envs}")
    print(f"  Obs dims:  74 (anticipatory features disabled)")
    print(f"  Startup:   10 random, then Bayesian")
    print()
    print(f"  Searched (7 params):")
    print(f"    w_wait(1.5-2.8) w_ride(0.6-1.4) w_rej(8-10)")
    print(f"    lr(1e-4..4e-4) schedule(const/linear)")
    print(f"    gamma(0.990-0.996) ent(0.01-0.05)")
    print(f"  Fixed:")
    print(f"    w_detour=1.0 w_ride_sq=0.3 w_cost=0.2")
    print(f"    arch=[128,128] steps=1024 epochs=5")
    print(f"    batch=128 vf=1.0 lam=0.95 clip=0.2")
    print(f"    norm_obs=False norm_reward=True")
    print("=" * 60)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Safe to Ctrl+C — study saved after every trial.\n")

    t_start = time.time()
    study.optimize(
        lambda trial: run_trial(trial, args.timesteps, args.n_envs, tb_base),
        n_trials          = n_remaining,
        callbacks         = [trial_callback],
        show_progress_bar = False,
    )
    total_time = time.time() - t_start

    # ==================================================================
    # Results
    # ==================================================================
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
    print("SEARCH v3 COMPLETE")
    print("=" * 60)
    print(f"  Time: {total_time/3600:.1f} hours")
    print(f"  Trials: {len(valid)} valid / {len(study.trials)} total")
    print()
    print(f"  {'Metric':<20} {'Best':>10}")
    print(f"  {'-'*32}")
    print(f"  {'score':<20} {best.value:>10.3f}")
    print(f"  {'service_rate':<20} {best_attrs.get('service_rate',0):>9.1%}")
    print(f"  {'mean_wait_all':<20} {best_attrs.get('mean_wait_all',0):>9.2f}")
    print(f"  {'mean_wait_served':<20} {best_attrs.get('mean_wait_served',0):>9.2f}")
    print(f"  {'mean_ride':<20} {best_attrs.get('mean_ride',0):>9.2f}")
    print(f"  {'rejected':<20} {best_attrs.get('rejected',0):>9.1f}")
    print()
    print(f"  Config (trial #{best.number + 1}):")
    print(f"    w_wait={best_params.get('w_wait',0):.3f}"
          f"  w_ride={best_params.get('w_ride',0):.3f}"
          f"  w_rej={best_params.get('w_rejection',0):.3f}")
    print(f"    lr={best_params.get('lr_start',0):.6f}"
          f"({best_params.get('lr_schedule','')})"
          f"  gamma={best_params.get('gamma',0):.4f}"
          f"  ent={best_params.get('ent_coef',0):.5f}")

    # Save best_config.json
    best_config = {
        "w_acceptance": 1.0,
        "w_wait":       best_params.get("w_wait"),
        "w_ride":       best_params.get("w_ride"),
        "w_detour":     1.0,
        "w_rejection":  best_params.get("w_rejection"),
        "w_ride_sq":    0.3,
        "w_cost":       0.2,
        "lr_start":     best_params.get("lr_start"),
        "lr_schedule":  best_params.get("lr_schedule"),
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
        "reward_mode":  "composite",
        "timesteps":    args.timesteps,
        "n_envs":       args.n_envs,
        "norm_obs":     False,
        "norm_reward":  True,
        "source":       "rl_tune_v3.py",
        "best_trial":   best.number + 1,
        "best_score":   round(best.value, 4),
    }
    config_path = os.path.join(args.output_dir, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n  Saved: {config_path}")

    # Save all_trials.csv
    csv_path   = os.path.join(args.output_dir, "all_trials.csv")
    fieldnames = [
        "trial", "score", "service_rate",
        "mean_wait_all", "mean_wait_served", "p95_wait",
        "mean_ride", "p95_ride", "mean_detour",
        "rejected", "rejection_rate", "train_time_s",
        "w_wait", "w_ride", "w_rejection",
        "lr_start", "lr_schedule", "gamma", "ent_coef",
    ]
    all_complete = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in sorted(all_complete, key=lambda x: x.value
                        if x.value != float("inf") else 999):
            row = {"trial": t.number + 1, "score": t.value}
            row.update(t.user_attrs)
            row.update(t.params)
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"  Saved: {csv_path}")

    # Top 5
    top5 = sorted(valid, key=lambda t: t.value)[:5]
    print(f"\n  TOP 5:")
    print(f"  {'#':<4} {'score':>7} {'w_all':>7} {'svc':>6}"
          f" {'wait':>6} {'rej':>4}")
    for t in top5:
        a = t.user_attrs
        print(f"  {t.number+1:<4} {t.value:>7.3f}"
              f" {a.get('mean_wait_all',0):>6.2f}"
              f" {a.get('service_rate',0):>5.1%}"
              f" {a.get('mean_wait_served',0):>6.2f}"
              f" {a.get('rejected',0):>4.0f}")

    print(f"\n  Next: python rl_train_from_tune.py --config {config_path}")


if __name__ == "__main__":
    main()