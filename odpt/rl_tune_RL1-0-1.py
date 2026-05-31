# rl_tune_RL1-0-1.py
# Optuna hyperparameter search for v6 reward (wait_all + noise training).
# Canonical final tuner; writes best config to rl_tune_RL1-0-1_best.json.
#
# python rl_tune_RL1-0-1.py --trials 30 --jobs 4

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
MAX_WAIT         = 120.0   # rejected passengers counted at this wait time

GREEDY_WAIT    = 8.3       # greedy baseline (DARPEnv, deterministic)
GREEDY_SERVICE = 0.885


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def compute_objective(results: list[dict], n_requests: int) -> tuple[float, dict]:
    """
    Compute Optuna minimisation score from evaluation episodes.

    score = mean_wait_all(MAX_WAIT=120) + 0.5*mean_ride + 0.05*p95_wait

    No explicit rejection penalty — handled by mean_wait_all.
    """
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
# Peak hour definitions (sim time, t=0 is 05:30)
# From config.py Malta demand profile:
#   Morning peak: 07:00-09:00 = t=90-210  (interval / 2.0)
#   Evening peak: 15:00-18:00 = t=570-750 (interval / 1.8)
# ---------------------------------------------------------------------------
PEAK_WINDOWS = [(90, 210), (570, 750)]


def _is_peak(t: float) -> bool:
    """True if sim time t falls within a demand peak window."""
    return any(lo <= t < hi for lo, hi in PEAK_WINDOWS)


# ---------------------------------------------------------------------------
# Evaluation — with in-progress flush + thesis metrics
# ---------------------------------------------------------------------------

def evaluate_policy(model, cfg, n_episodes: int = 10) -> tuple[list[dict], dict]:
    """
    Evaluate a trained model over held-out episodes.

    Returns (raw_results_list, aggregated_metrics_dict).

    Standard metrics: service_rate, mean_wait, p95_wait, mean_ride,
        p95_ride, mean_detour, rejected, reward.

    Thesis-specific additions:
      peak_mean_wait        — mean wait for requests arriving during peaks
      peak_rejection_rate   — rejection rate during peaks only
      offpeak_mean_wait     — mean wait outside peaks
      offpeak_rejection_rate— rejection rate outside peaks
      load_std              — std dev of vehicle onboard counts across
                              the fleet, sampled at each decision step.
                              Low = balanced assignment. High = lopsided.
    """
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
        env  = DARPEnv(cfg=eval_cfg)
        obs, _ = env.reset(seed=eval_cfg.seed)
        done   = False
        total_reward = 0.0

        # Track fleet load balance: sample vehicle loads at each step
        load_samples = []

        for _ in range(cfg.n_requests * 2):
            if done:
                break

            # Sample fleet load distribution before this decision
            loads = [len(v.onboard) for v in env._vehicles.values()]
            if any(l > 0 for l in loads):  # skip all-zero (idle fleet)
                load_samples.append(float(np.std(loads)))

            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        # Flush in-progress passengers
        flush_time = cfg.service_end + 500
        env._advance_vehicles_to(flush_time)

        # --- Standard summary ---
        summary = env.episode_summary()
        summary["reward"] = total_reward

        # --- Detour ratio ---
        reqs = env._requests
        detours = []
        for r in reqs.values():
            if (r.pickup_time is not None and r.dropoff_time is not None
                    and r.direct_time and r.direct_time > 0):
                detours.append((r.dropoff_time - r.pickup_time) / r.direct_time)
        summary["mean_detour"] = float(np.mean(detours)) if detours else None

        # --- Peak vs off-peak breakdown ---
        peak_waits    = []
        offpeak_waits = []
        peak_total    = 0
        peak_rejected = 0
        offpeak_total    = 0
        offpeak_rejected = 0

        for r in reqs.values():
            is_pk = _is_peak(r.request_time)
            if is_pk:
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

        summary["peak_mean_wait"] = (
            float(np.mean(peak_waits)) if peak_waits else None
        )
        summary["peak_rejection_rate"] = (
            peak_rejected / max(peak_total, 1)
        )
        summary["offpeak_mean_wait"] = (
            float(np.mean(offpeak_waits)) if offpeak_waits else None
        )
        summary["offpeak_rejection_rate"] = (
            offpeak_rejected / max(offpeak_total, 1)
        )

        # --- Load balance ---
        summary["load_std"] = (
            float(np.mean(load_samples)) if load_samples else 0.0
        )

        results.append(summary)

    # --- Aggregate across episodes ---
    def _mean(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return results, {
        # Standard
        "service_rate": _mean("service_rate"),
        "mean_wait":    _mean("mean_wait"),
        "p95_wait":     _mean("p95_wait"),
        "mean_ride":    _mean("mean_ride"),
        "p95_ride":     _mean("p95_ride"),
        "mean_detour":  _mean("mean_detour"),
        "rejected":     _mean("rejected"),
        "reward":       _mean("reward"),
        # Thesis additions
        "peak_mean_wait":         _mean("peak_mean_wait"),
        "peak_rejection_rate":    _mean("peak_rejection_rate"),
        "offpeak_mean_wait":      _mean("offpeak_mean_wait"),
        "offpeak_rejection_rate": _mean("offpeak_rejection_rate"),
        "load_std":               _mean("load_std"),
    }


# ---------------------------------------------------------------------------
# TensorBoard eval callback — logs all thesis metrics
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
                _, metrics = evaluate_policy(self.model, self.cfg, n_episodes=5)

                sr  = metrics["service_rate"] or 0
                rej = metrics["rejected"] or 0
                mw  = metrics["mean_wait"] or 0
                n   = self.cfg.n_requests
                mw_all = ((sr * n * mw) + (rej * MAX_WAIT)) / max(sr * n + rej, 1)
                rr  = rej / max(n, 1)

                # --- Core eval metrics ---
                self.logger.record("eval/service_rate",   sr)
                self.logger.record("eval/mean_wait",      mw)
                self.logger.record("eval/mean_wait_all",  mw_all)
                self.logger.record("eval/p95_wait",       metrics["p95_wait"]     or 0)
                self.logger.record("eval/mean_ride",      metrics["mean_ride"]    or 0)
                self.logger.record("eval/p95_ride",       metrics["p95_ride"]     or 0)
                self.logger.record("eval/mean_detour",    metrics["mean_detour"]  or 0)
                self.logger.record("eval/rejected",       rej)
                self.logger.record("eval/rejection_rate", rr)

                # --- Peak vs off-peak (anticipatory behaviour evidence) ---
                self.logger.record("eval/peak_mean_wait",
                                   metrics["peak_mean_wait"] or 0)
                self.logger.record("eval/peak_rejection_rate",
                                   metrics["peak_rejection_rate"] or 0)
                self.logger.record("eval/offpeak_mean_wait",
                                   metrics["offpeak_mean_wait"] or 0)
                self.logger.record("eval/offpeak_rejection_rate",
                                   metrics["offpeak_rejection_rate"] or 0)

                # --- Fleet utilisation balance ---
                self.logger.record("eval/load_std",
                                   metrics["load_std"] or 0)

                # --- Gap vs greedy baseline ---
                self.logger.record("eval/greedy_gap_wait", mw - GREEDY_WAIT)

                self.logger.dump(self.num_timesteps)

            return True

    return DARPEvalCallback(cfg, eval_freq)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(trial: optuna.Trial, timesteps: int, n_envs: int,
              tb_base: str) -> float:
    """
    Train MaskablePPO with Optuna-suggested hyperparameters.
    Returns composite passenger-satisfaction score (minimised).
    """
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from config import SimulationConfig
    from rl_env import DARPEnv

    # ==================================================================
    # Group 1: Reward weights (4 searched, 3 fixed)
    # ==================================================================
    w_wait      = trial.suggest_float("w_wait",      1.0, 3.0)
    w_ride      = trial.suggest_float("w_ride",      0.5, 1.5)
    w_detour    = trial.suggest_float("w_detour",    0.0, 1.5)
    w_rejection = trial.suggest_float("w_rejection", 5.0, 10.0)

    # Fixed reward weights
    w_ride_sq    = 0.3
    w_cost       = 0.2
    w_acceptance = 1.0

    # ==================================================================
    # Group 2: PPO hyperparameters (6 searched, 4 fixed)
    # ==================================================================
    lr_start    = trial.suggest_float("lr_start", 5e-5, 5e-4, log=True)
    lr_schedule = trial.suggest_categorical(
        "lr_schedule", ["constant", "linear", "cosine", "warmup_cosine"]
    )
    gamma       = trial.suggest_float("gamma",    0.990, 0.998)
    ent_coef    = trial.suggest_float("ent_coef", 0.005, 0.05, log=True)
    n_steps     = trial.suggest_categorical("n_steps",  [1024, 2048])
    n_epochs    = trial.suggest_categorical("n_epochs", [5, 8, 10, 15])

    # Fixed PPO
    batch_size = 128
    vf_coef    = 1.0
    gae_lambda = 0.95
    clip_range = 0.2

    # ==================================================================
    # Group 3: Architecture (1 searched)
    # ==================================================================
    net_arch_choice = trial.suggest_categorical("net_arch", ["64_64", "128_128", "256_128"])
    net_arch_map = {
        "64_64":   [64, 64],
        "128_128": [128, 128],
        "256_128": [256, 128],
    }
    net_arch = net_arch_map[net_arch_choice]

    # LR schedule factory
    # SB3 progress goes from 1.0 (start) → 0.0 (end of training)
    if lr_schedule == "linear":
        learning_rate = lambda progress: lr_start * progress
    elif lr_schedule == "cosine":
        import math as _math
        # Cosine annealing: slower mid-training decay than linear
        learning_rate = lambda progress: (
            lr_start * 0.5 * (1 + _math.cos(_math.pi * (1 - progress)))
        )
    elif lr_schedule == "warmup_cosine":
        import math as _math
        # Warmup first 10% of training, then cosine decay
        _warmup_frac = 0.1
        def learning_rate(progress):
            elapsed = 1.0 - progress  # 0→1 as training progresses
            if elapsed < _warmup_frac:
                return lr_start * (elapsed / _warmup_frac)
            cos_progress = (elapsed - _warmup_frac) / (1.0 - _warmup_frac)
            return lr_start * 0.5 * (1 + _math.cos(_math.pi * cos_progress))
    else:  # constant
        learning_rate = lr_start

    print(f"\n  --- Trial {trial.number + 1} ---")
    print(f"    Reward:  w_wait={w_wait:.2f}  w_ride={w_ride:.2f}"
          f"  w_detour={w_detour:.2f}  w_rej={w_rejection:.2f}")
    print(f"    PPO:     lr={lr_start:.5f}({lr_schedule})"
          f"  gamma={gamma:.4f}  ent={ent_coef:.4f}"
          f"  steps={n_steps}  epochs={n_epochs}")
    print(f"    Arch:    {net_arch}")

    # ==================================================================
    # Simulation config — thesis constants
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

    # KEY FIX: norm_obs=False
    # Observations are already normalised to [-1, 1] by _encode_state().
    # VecNormalize's obs normalisation is redundant and creates a
    # train/eval mismatch (v1 bug: eval used raw obs, model expected
    # VecNormalize-standardised obs).
    # norm_reward=True remains — this is the critical fix from run005
    # that enabled critic learning.
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,      # FIX: obs already normalised manually
        norm_reward=True,    # KEEP: rescales rewards for critic learning
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
    )

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    tb_log_dir = os.path.join(tb_base, f"trial_{trial.number + 1:03d}")

    # Resolve batch_size compatibility
    buffer = n_steps * n_envs
    actual_batch = batch_size
    if buffer % actual_batch != 0:
        valid = [b for b in [64, 128, 256] if buffer % b == 0 and b <= actual_batch]
        actual_batch = max(valid) if valid else 64

    # ==================================================================
    # Model
    # ==================================================================
    model = MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate   = learning_rate,
        gamma           = gamma,
        ent_coef        = ent_coef,
        n_steps         = n_steps,
        batch_size      = actual_batch,
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

    # Eval callback: log task metrics every 50k steps
    eval_cb = make_eval_callback(cfg, eval_freq=50_000 // n_envs)

    # Timeout callback: 25 min max per trial
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    t0 = time.time()
    TRIAL_TIMEOUT = 25 * 60

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
    # Evaluation — 10 episodes (up from 8 in v1)
    # ==================================================================
    results, _ = evaluate_policy(model, cfg, n_episodes=10)
    vec_env.close()

    score, metrics = compute_objective(results, cfg.n_requests)

    # Store all metrics as user_attrs for later analysis
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
        description="Optuna HPO v2 — DARP MaskablePPO (clean start)"
    )
    parser.add_argument("--samples",    type=int, default=50)
    parser.add_argument("--timesteps",  type=int, default=400_000)
    parser.add_argument("--n-envs",     type=int, default=6)
    parser.add_argument("--output-dir", default="rl_outputs/tune_v2")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--study-name", default="darp_ppo_tune_v2_clean")
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
            sampler    = TPESampler(
                seed=123,             # different seed from v1
                n_startup_trials=15,  # up from 10 — 11 search params
            ),
        )
        print(f"New study: {args.study_name}")

    def trial_callback(study: optuna.Study, trial: optuna.Trial):
        attrs = trial.user_attrs
        print(f"\n  Trial {trial.number + 1} result:")
        print(f"    score:          {trial.value:.3f}")
        print(f"    service_rate:   {attrs.get('service_rate', 0):.1%}"
              f"  (greedy: {GREEDY_SERVICE:.1%})")
        print(f"    mean_wait_all:  {attrs.get('mean_wait_all', 0):.2f} min")
        print(f"    mean_wait_svd:  {attrs.get('mean_wait_served', 0):.2f} min"
              f"  (greedy: {GREEDY_WAIT})")
        print(f"    mean_ride:      {attrs.get('mean_ride', 0):.2f} min")
        print(f"    p95_wait:       {attrs.get('p95_wait', 0):.2f} min")
        print(f"    mean_detour:    {attrs.get('mean_detour', 0):.3f}x")
        print(f"    rejected:       {attrs.get('rejected', 0):.1f}")
        print(f"    train_time:     {attrs.get('train_time_s', 0):.0f}s")

        valid = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value != float("inf")]
        if valid:
            b = min(valid, key=lambda t: t.value)
            print(f"  >>> Best so far: trial {b.number + 1}"
                  f"  score={b.value:.3f}"
                  f"  wait_all={b.user_attrs.get('mean_wait_all', 0):.2f}"
                  f"  svc={b.user_attrs.get('service_rate', 0):.1%}")

        # Save after every trial — resume-safe
        with open(study_path, "wb") as f:
            pickle.dump(study, f)

    completed   = [t for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE]
    n_remaining = args.samples - len(completed)

    if n_remaining <= 0:
        print(f"All {args.samples} trials already complete.")
        return

    print("=" * 65)
    print("Optuna Search v2 — DARP MaskablePPO — Clean Start")
    print("=" * 65)
    print(f"  Trials:    {args.samples} total ({n_remaining} remaining)")
    print(f"  Timesteps: {args.timesteps:,} per trial")
    print(f"  Envs:      {args.n_envs} parallel (matches CPU cores)")
    print(f"  Startup:   15 random trials before Bayesian")
    print(f"  Timeout:   25 min per trial")
    print(f"  Eval:      10 episodes per trial")
    print(f"  Obs dims:  78 (v2: 4 anticipatory features added)")
    print(f"  Est. time: ~{n_remaining * 12 // 60}h {n_remaining * 12 % 60}m"
          f" (at ~12 min/trial with {args.n_envs} envs)")
    print()
    print(f"  KEY FIXES:")
    print(f"    1. norm_obs=False (eliminates train/eval obs mismatch)")
    print(f"    2. +4 anticipatory obs (demand forecast, spare capacity)")
    print(f"    3. Fresh study (v1 TPE model corrupted by 4 patches)")
    print()
    print(f"  Objective (minimise):")
    print(f"    mean_wait_all(MAX_WAIT={MAX_WAIT:.0f})"
          f" + {OBJ_WEIGHT_RIDE}*mean_ride"
          f" + {OBJ_WEIGHT_P95}*p95_wait")
    print()
    print(f"  Searched (12 parameters):")
    print(f"    Reward: w_wait(1-3) w_ride(0.5-1.5)"
          f" w_detour(0-1.5) w_rej(5-10)")
    print(f"    PPO:    lr(5e-5..5e-4)"
          f" schedule(const/linear/cosine/warmup_cosine)")
    print(f"            gamma(0.990-0.998) ent(0.005-0.05)")
    print(f"            n_steps(1024/2048) n_epochs(5/8/10/15)")
    print(f"    Arch:   net_arch(64x64 / 128x128 / 256x128)")
    print(f"  Fixed: w_ride_sq=0.3 w_cost=0.2 batch=128"
          f" vf=1.0 lam=0.95 clip=0.2")
    print()
    print(f"  Greedy baseline: wait={GREEDY_WAIT} svc={GREEDY_SERVICE:.1%}")
    print(f"  Resume: python rl_tune_v2.py --resume")
    print("=" * 65)
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

    print("\n" + "=" * 65)
    print("SEARCH v2 COMPLETE")
    print("=" * 65)
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f} hours)")
    print(f"  Trials:     {len(valid)} valid / {len(study.trials)} total")
    print()
    print(f"  {'Metric':<20} {'Best':>10} {'v1 Best':>10} {'Greedy':>10}")
    print(f"  {'-'*52}")
    print(f"  {'score':<20} {best.value:>10.3f} {'25.73':>10}")
    print(f"  {'service_rate':<20} {best_attrs.get('service_rate',0):>9.1%}"
          f" {'88.8%':>10} {'88.5%':>10}")
    print(f"  {'mean_wait_all':<20} {best_attrs.get('mean_wait_all',0):>9.2f}"
          f" {'20.05':>10}")
    print(f"  {'mean_wait_served':<20} {best_attrs.get('mean_wait_served',0):>9.2f}"
          f" {'10.10':>10} {'8.3':>10}")
    print(f"  {'mean_ride':<20} {best_attrs.get('mean_ride',0):>9.2f}"
          f" {'8.51':>10} {'8.1':>10}")
    print(f"  {'p95_wait':<20} {best_attrs.get('p95_wait',0):>9.2f}"
          f" {'28.29':>10}")
    print(f"  {'mean_detour':<20} {best_attrs.get('mean_detour',0):>9.3f}x")
    print(f"  {'rejected':<20} {best_attrs.get('rejected',0):>9.1f}"
          f" {'35':>10} {'~40':>10}")
    print()
    print(f"  Best config (trial #{best.number + 1}):")
    print(f"    Reward:  w_wait={best_params.get('w_wait',0):.3f}"
          f"  w_ride={best_params.get('w_ride',0):.3f}"
          f"  w_detour={best_params.get('w_detour',0):.3f}"
          f"  w_rej={best_params.get('w_rejection',0):.3f}")
    print(f"    PPO:     lr={best_params.get('lr_start',0):.6f}"
          f"({best_params.get('lr_schedule','')})"
          f"  gamma={best_params.get('gamma',0):.4f}"
          f"  ent={best_params.get('ent_coef',0):.5f}")
    print(f"             steps={best_params.get('n_steps',0)}"
          f"  epochs={best_params.get('n_epochs',0)}"
          f"  arch={best_params.get('net_arch','')}")

    # ------------------------------------------------------------------
    # Save best_config.json
    # ------------------------------------------------------------------
    best_config = {
        # Searched reward weights
        "w_acceptance": 1.0,
        "w_wait":       best_params.get("w_wait"),
        "w_ride":       best_params.get("w_ride"),
        "w_detour":     best_params.get("w_detour"),
        "w_rejection":  best_params.get("w_rejection"),
        # Fixed reward weights
        "w_ride_sq":    0.3,
        "w_cost":       0.2,
        # Searched PPO params
        "lr_start":     best_params.get("lr_start"),
        "lr_schedule":  best_params.get("lr_schedule"),
        "gamma":        best_params.get("gamma"),
        "ent_coef":     best_params.get("ent_coef"),
        "n_steps":      best_params.get("n_steps"),
        "n_epochs":     best_params.get("n_epochs"),
        # Fixed PPO params
        "batch_size":   128,
        "vf_coef":      1.0,
        "gae_lambda":   0.95,
        "clip_range":   0.2,
        "max_grad_norm": 0.5,
        # Architecture
        "net_arch":     net_arch_map[best_params.get("net_arch", "128_128")],
        "reward_mode":  "composite",
        "timesteps":    args.timesteps,
        "n_envs":       args.n_envs,
        # Context
        "source":                    "rl_tune_v2.py — clean search",
        "n_trials":                  len(study.trials),
        "n_valid_trials":            len(valid),
        "best_trial_number":         best.number + 1,
        "best_score":                round(best.value, 4),
        "norm_obs":                  False,   # KEY: documents the fix
        "norm_reward":               True,
        "max_wait_penalty":          MAX_WAIT,
        "baseline_greedy_wait":      GREEDY_WAIT,
        "baseline_greedy_service":   GREEDY_SERVICE,
        "achieved_mean_wait_all":    best_attrs.get("mean_wait_all"),
        "achieved_mean_wait_served": best_attrs.get("mean_wait_served"),
        "achieved_service_rate":     best_attrs.get("service_rate"),
        "achieved_mean_ride":        best_attrs.get("mean_ride"),
        "achieved_p95_wait":         best_attrs.get("p95_wait"),
        "achieved_p95_ride":         best_attrs.get("p95_ride"),
        "achieved_mean_detour":      best_attrs.get("mean_detour"),
    }
    config_path = os.path.join(args.output_dir, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n  Saved: {config_path}")

    # ------------------------------------------------------------------
    # Save all_trials.csv
    # ------------------------------------------------------------------
    csv_path   = os.path.join(args.output_dir, "all_trials.csv")
    fieldnames = [
        "trial", "score", "service_rate",
        "mean_wait_all", "mean_wait_served", "p95_wait",
        "mean_ride", "p95_ride", "mean_detour",
        "rejected", "rejection_rate", "mean_reward", "train_time_s",
        "w_wait", "w_ride", "w_detour", "w_rejection",
        "lr_start", "lr_schedule", "gamma", "ent_coef",
        "n_steps", "n_epochs", "net_arch",
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
    print("\n" + "=" * 65)
    print("TOP 5 TRIALS:")
    print("=" * 65)
    print(f"  {'#':<4} {'score':>7} {'w_all':>7} {'svc':>6}"
          f" {'ride':>5} {'rej':>4} {'arch':>8}"
          f" {'w_w':>5} {'w_r':>5} {'w_rej':>5}")
    print(f"  {'-'*70}")
    for t in top5:
        a = t.user_attrs
        p = t.params
        print(f"  {t.number+1:<4} {t.value:>7.3f}"
              f" {a.get('mean_wait_all',0):>6.2f}"
              f" {a.get('service_rate',0):>5.1%}"
              f" {a.get('mean_ride',0):>5.2f}"
              f" {a.get('rejected',0):>4.0f}"
              f" {p.get('net_arch',''):>8}"
              f" {p.get('w_wait',0):>5.2f}"
              f" {p.get('w_ride',0):>5.2f}"
              f" {p.get('w_rejection',0):>5.2f}")

    print(f"\n  Next: python rl_train_from_tune.py --config {config_path}")
    print(f"        tensorboard --logdir {tb_base}")


if __name__ == "__main__":
    main()