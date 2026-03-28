#!/usr/bin/env python3
# rl_tune.py
# Optuna hyperparameter search for DARP MaskablePPO.
# No Ray dependency — works on Windows with venv.
#
# Two distinct layers — both searched, neither conflated:
#
#   1. REWARD WEIGHTS (DARPEnv — shape agent behaviour during training)
#      w_wait, w_ride, w_cost, w_rejection
#      Determine what signal the agent optimises at every step.
#
#   2. PPO HYPERPARAMETERS (shape learning stability and speed)
#      lr_start, lr_schedule, gamma, ent_coef, n_steps, batch_size,
#      n_epochs, vf_coef, gae_lambda, clip_range
#
#   3. OPTUNA OBJECTIVE (evaluates final policy — independent of reward)
#      score = mean_wait_all            <- ALL passengers incl. rejected
#            + 0.5  * mean_ride
#            + 0.3  * p95_wait          <- tail performance
#            + 10.0 * rejection_rate    <- explicit rejection penalty
#      Rejected passengers counted as MAX_WAIT (30 min) in mean_wait_all.
#      No hard service_rate floor — 79-88% is the normal operating range.
#
# TensorBoard output per trial:
#   SB3 defaults: approx_kl, clip_frac, entropy_loss, explained_variance,
#                 learning_rate, loss, policy_gradient_loss, val_loss
#   Added (thesis-relevant):
#     eval/service_rate, eval/mean_wait, eval/mean_ride, eval/rejected,
#     eval/mean_wait_all, eval/rejection_rate, eval/p95_wait, eval/p95_ride
#
# Usage:
#   pip install optuna
#   python rl_tune.py                   # 25 trials (~6h at ~14min/trial)
#   python rl_tune.py --samples 10      # quick test
#   python rl_tune.py --resume          # continue after interrupt
#
# Outputs:
#   rl_outputs/tune_results/
#     study.pkl          <- saved after every trial (resume-safe)
#     best_config.json   <- winning hyperparameters + reward weights
#     all_trials.csv     <- all results for thesis reporting
#     tb/trial_NNN/      <- TensorBoard logs per trial

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
# Optuna objective constants — evaluation only, not training signal
# ---------------------------------------------------------------------------
OBJ_WEIGHT_RIDE      = 0.5
OBJ_WEIGHT_P95       = 0.05
OBJ_WEIGHT_REJECTION = 0.0
MAX_WAIT             = 120.0   # minutes — penalty applied to rejected passengers

GREEDY_WAIT    = 8.3
GREEDY_SERVICE = 0.885
RUN006_WAIT    = 10.3


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def compute_objective(results: list[dict], n_requests: int) -> tuple[float, dict]:
    """
    Compute Optuna minimisation score from evaluation episode results.

    Loopholes closed:
      1. mean_wait_all — rejected passengers treated as waiting MAX_WAIT.
         Cannot game wait by rejecting hard requests.
      2. rejection_rate penalty (weight 10.0) — double-penalises rejection
         on top of mean_wait_all. Redundant but makes gaming even harder.
      3. p95_wait — cannot hide bad tail performance behind a good mean.
      4. mean_ride — cannot accept fast but route badly.
      5. No hard service_rate floor — 79-88% is normal across all
         algorithms in this thesis. Composite penalty handles rejection.
    """
    service_rate     = float(np.mean([r["service_rate"] for r in results]))
    rejected         = float(np.mean([r["rejected"]     for r in results]))
    mean_ride        = float(np.mean([r["mean_ride"]    for r in results if r.get("mean_ride")]))
    p95_ride         = float(np.mean([r["p95_ride"]     for r in results if r.get("p95_ride")]))
    mean_reward      = float(np.mean([r["reward"]       for r in results]))
    mean_wait_served = float(np.mean([r["mean_wait"]    for r in results if r.get("mean_wait")]))
    p95_wait         = float(np.mean([r["p95_wait"]     for r in results if r.get("p95_wait")]))
    mean_detour      = float(np.mean([r["mean_detour"]  for r in results if r.get("mean_detour")]))

    # mean_wait_all: rejected passengers counted at MAX_WAIT
    n_served   = service_rate * n_requests
    n_rejected = rejected
    if n_served + n_rejected > 0:
        mean_wait_all = ((n_served * mean_wait_served) + (n_rejected * MAX_WAIT)) \
                        / (n_served + n_rejected)
    else:
        mean_wait_all = MAX_WAIT

    rejection_rate = n_rejected / max(n_requests, 1)

    score = (
        mean_wait_all
        + OBJ_WEIGHT_RIDE      * mean_ride
        + OBJ_WEIGHT_P95       * p95_wait
    )

    return score, {
        "service_rate":      service_rate,
        "mean_wait_all":     mean_wait_all,
        "mean_wait_served":  mean_wait_served,
        "mean_ride":         mean_ride,
        "p95_wait":          p95_wait,
        "p95_ride":          p95_ride,
        "mean_detour":       mean_detour,
        "rejected":          rejected,
        "rejection_rate":    rejection_rate,
        "mean_reward":       mean_reward,
        "score":             score,
    }


# ---------------------------------------------------------------------------
# Evaluation helper — mirrors main.py metrics as closely as possible
# ---------------------------------------------------------------------------

def evaluate_policy(model, cfg, n_episodes: int = 8) -> tuple[list[dict], dict]:
    """
    Evaluate a trained model over n_episodes held-out episodes.
    Returns raw results list and aggregated metrics dict.

    Tracks: service_rate, mean_wait, p95_wait, mean_ride, p95_ride,
            mean_detour, rejected — matching main.py SimPy output.
    """
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
        max_steps    = cfg.n_requests * 2   # hard limit — prevents hangs

        for _ in range(max_steps):
            if done:
                break
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        summary = env.episode_summary()
        summary["reward"] = total_reward

        # Compute detour ratio from episode data
        reqs = env._requests
        detours = [
            r.detour_ratio for r in reqs.values()
            if r.detour_ratio is not None
        ]
        summary["mean_detour"] = float(np.mean(detours)) if detours else None
        results.append(summary)

    # Aggregate
    def _mean(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return results, {
        "service_rate": _mean("service_rate"),
        "mean_wait":    _mean("mean_wait"),
        "p95_wait":     _mean("p95_wait"),
        "mean_ride":    _mean("mean_ride"),
        "p95_ride":     _mean("p95_ride"),
        "mean_detour":  _mean("mean_detour"),
        "rejected":     _mean("rejected"),
        "reward":       _mean("reward"),
    }


# ---------------------------------------------------------------------------
# Custom TensorBoard callback — logs task metrics alongside SB3 defaults
# ---------------------------------------------------------------------------

def make_eval_callback(model_ref_holder: list, cfg, eval_freq: int, tb_writer):
    """
    Returns an SB3 BaseCallback that logs evaluation metrics to TensorBoard
    every eval_freq steps. Logs both SB3 training diagnostics and
    task-level metrics (service_rate, mean_wait, rejected, etc.)
    for thesis-quality TensorBoard graphs.

    tb_writer: a simple dict holder for the writer reference
    (SB3 doesn't expose the TB writer easily, so we use the model's logger)
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class DARPEvalCallback(BaseCallback):
        def __init__(self, cfg, eval_freq):
            super().__init__(verbose=0)
            self.cfg       = cfg
            self.eval_freq = eval_freq

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                _, metrics = evaluate_policy(self.model, self.cfg, n_episodes=5)

                # Compute mean_wait_all for logging
                sr  = metrics["service_rate"] or 0
                rej = metrics["rejected"] or 0
                mw  = metrics["mean_wait"] or 0
                n   = self.cfg.n_requests
                mw_all = ((sr * n * mw) + (rej * MAX_WAIT)) / max(sr * n + rej, 1)
                rr  = rej / max(n, 1)

                # Log to TensorBoard via SB3 logger
                # These appear alongside SB3 defaults in the same TB run
                self.logger.record("eval/service_rate",   metrics["service_rate"] or 0)
                self.logger.record("eval/mean_wait",      metrics["mean_wait"]    or 0)
                self.logger.record("eval/mean_wait_all",  mw_all)
                self.logger.record("eval/p95_wait",       metrics["p95_wait"]     or 0)
                self.logger.record("eval/mean_ride",      metrics["mean_ride"]    or 0)
                self.logger.record("eval/p95_ride",       metrics["p95_ride"]     or 0)
                self.logger.record("eval/mean_detour",    metrics["mean_detour"]  or 0)
                self.logger.record("eval/rejected",       metrics["rejected"]     or 0)
                self.logger.record("eval/rejection_rate", rr)
                self.logger.dump(self.num_timesteps)

            return True

    return DARPEvalCallback(cfg, eval_freq)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(trial: optuna.Trial, timesteps: int, n_envs: int,
              tb_base: str) -> float:
    """
    Train MaskablePPO with Optuna-suggested hyperparameters + reward weights.
    Returns composite passenger-satisfaction score (minimised by Optuna).
    """
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from config import SimulationConfig
    from rl_env import DARPEnv

    # ------------------------------------------------------------------
    # Group 1: Reward weights — searched (4 parameters)
    # ------------------------------------------------------------------
    w_wait = trial.suggest_float("w_wait", 1.0, 4.0)
    # Primary passenger satisfaction signal.
    # Originally hardcoded 2.0 — never searched.

    w_ride = trial.suggest_float("w_ride", 0.5, 2.0)
    # Absolute linear ride time penalty — catches long rides regardless
    # of whether they are a detour. Originally hardcoded 1.0.

    w_detour = trial.suggest_float("w_detour", 0.0, 4.0)
    # Excess detour penalty — only fires above 1.0x direct time.
    # Maps directly to your measured detour=1.29x metric.
    # More semantically precise than ride penalty alone.

    w_rejection = trial.suggest_float("w_rejection", 3.0, 12.0)
    # Rejection penalty. Originally hardcoded 5.0.
    # Critical — too low causes rejection gaming during training.

    # Fixed reward weights — present but not worth searching
    w_ride_sq    = 0.3   # tail suppression proxy for p95_ride;
                         # fixed to avoid correlation with w_ride search
    w_cost       = 0.2   # operational signal; low — not passenger satisfaction
    w_acceptance = 1.0   # fixed baseline acceptance bonus

    # ------------------------------------------------------------------
    # Group 2: PPO hyperparameters — searched
    # ------------------------------------------------------------------
    lr_start    = trial.suggest_float("lr_start", 1e-4, 5e-4, log=True)
    lr_schedule = trial.suggest_categorical("lr_schedule", ["constant", "linear"])
    learning_rate = (
        (lambda progress: lr_start * progress) if lr_schedule == "linear"
        else lr_start
    )

    gamma    = trial.suggest_float("gamma",    0.992, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 0.01,  0.05, log=True)
    n_steps  = trial.suggest_categorical("n_steps",  [1024, 2048, 4096])
    n_epochs = trial.suggest_categorical("n_epochs", [5, 8, 10, 15])

    # PPO hyperparameters — fixed (evidence-based)
    batch_size = 128    # known good from run005; always divides n_steps*4
    vf_coef    = 1.0    # run006 value_loss=0.005 — critic fully stable
    gae_lambda = 0.95   # standard; VecNormalize handles reward noise
    clip_range = 0.2    # lr_start search already controls update size

    print(f"\n  --- Trial {trial.number + 1} config ---")
    print(f"    Reward:  w_wait={w_wait:.2f}  w_ride={w_ride:.2f}"
          f"  w_detour={w_detour:.2f}  w_rej={w_rejection:.2f}")
    print(f"    Fixed:   w_ride_sq={w_ride_sq}  w_cost={w_cost}"
          f"  batch={batch_size}  vf={vf_coef}"
          f"  lam={gae_lambda}  clip={clip_range}")
    print(f"    PPO:     lr={lr_start:.5f}({lr_schedule})"
          f"  gamma={gamma:.4f}  ent={ent_coef:.4f}"
          f"  steps={n_steps}  epochs={n_epochs}")

    # ------------------------------------------------------------------
    # Simulation config — fixed thesis constants
    # ------------------------------------------------------------------
    cfg = SimulationConfig(
        seed=42, fleet_size=6, vehicle_capacity=16,
        depot_node=0, n_requests=400, demand_profile="malta",
        stochastic_arrivals=True, travel_noise=0.0, n_nodes=71,
    )

    # ------------------------------------------------------------------
    # Parallel environments with searched reward weights
    # ------------------------------------------------------------------
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
        vec_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=gamma,
    )

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    tb_log_dir = os.path.join(tb_base, f"trial_{trial.number + 1:03d}")

    # ------------------------------------------------------------------
    # MaskablePPO with eval callback for TB logging
    # ------------------------------------------------------------------
    model = MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate   = learning_rate,
        gamma           = gamma,
        ent_coef        = ent_coef,
        n_steps         = n_steps,
        batch_size      = batch_size,   # fixed: 128
        n_epochs        = n_epochs,
        vf_coef         = vf_coef,      # fixed: 1.0
        gae_lambda      = gae_lambda,   # fixed: 0.95
        clip_range      = clip_range,   # fixed: 0.2
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(net_arch=[128, 128]),
        verbose         = 0,
        tensorboard_log = tb_log_dir,
        device          = device,
    )

    # Eval callback logs task metrics to TensorBoard every 50k steps
    eval_cb = make_eval_callback(
        model_ref_holder = [],
        cfg              = cfg,
        eval_freq        = 50_000 // n_envs,
        tb_writer        = {},
    )

    t0 = time.time()

    # Per-trial timeout via SB3 callback.
    # If training takes more than TRIAL_TIMEOUT_SECONDS, stop early
    # and let evaluation run on the partially trained model.
    # This prevents rare hangs (like trial 12 which took 3+ hours)
    # from stalling the entire search.
    TRIAL_TIMEOUT_SECONDS = 25 * 60  # 25 minutes max per trial

    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback

    class TimeoutCallback(_BaseCallback):
        def __init__(self, t0: float, timeout: float):
            super().__init__(verbose=0)
            self.t0      = t0
            self.timeout = timeout

        def _on_step(self) -> bool:
            if time.time() - self.t0 > self.timeout:
                print(f"\n  [TIMEOUT] Trial {trial.number + 1} exceeded"
                      f" {self.timeout/60:.0f} min — stopping training early.")
                return False  # stops model.learn()
            return True

    from stable_baselines3.common.callbacks import CallbackList
    combined_cb = CallbackList([eval_cb, TimeoutCallback(t0, TRIAL_TIMEOUT_SECONDS)])

    model.learn(total_timesteps=timesteps, callback=combined_cb)
    train_time = time.time() - t0

    # ------------------------------------------------------------------
    # Final evaluation — 8 episodes
    # ------------------------------------------------------------------
    results, _ = evaluate_policy(model, cfg, n_episodes=8)
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
        description="Optuna hyperparameter search — DARP MaskablePPO"
    )
    parser.add_argument("--samples",    type=int, default=35)
    parser.add_argument("--timesteps",  type=int, default=400_000)
    parser.add_argument("--n-envs",     type=int, default=4)
    parser.add_argument("--output-dir", default="rl_outputs/tune_results")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--study-name", default="darp_ppo_tune_v2")
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
            sampler    = TPESampler(seed=42, n_startup_trials=10),
        )
        print(f"New study: {args.study_name}")

    def trial_callback(study: optuna.Study, trial: optuna.Trial):
        attrs = trial.user_attrs
        print(f"\n  Trial {trial.number + 1} result:")
        print(f"    score:          {trial.value:.3f}")
        print(f"    service_rate:   {attrs.get('service_rate', 0):.1%}"
              f"  (greedy: {GREEDY_SERVICE:.1%})")
        print(f"    mean_wait_all:  {attrs.get('mean_wait_all', 0):.2f} min"
              f"  (rejected @ {MAX_WAIT:.0f} min)")
        print(f"    mean_wait_svd:  {attrs.get('mean_wait_served', 0):.2f} min"
              f"  (greedy: {GREEDY_WAIT})")
        print(f"    mean_ride:      {attrs.get('mean_ride', 0):.2f} min"
              f"  (greedy: 8.1)")
        print(f"    p95_wait:       {attrs.get('p95_wait', 0):.2f} min")
        print(f"    p95_ride:       {attrs.get('p95_ride', 0):.2f} min")
        print(f"    mean_detour:    {attrs.get('mean_detour', 0):.2f}x")
        print(f"    rejected:       {attrs.get('rejected', 0):.1f}"
              f"  ({attrs.get('rejection_rate', 0):.1%})")
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

    print("=" * 60)
    print("Optuna Search — DARP MaskablePPO — Passenger Satisfaction")
    print("=" * 60)
    print(f"  Trials:    {args.samples} ({n_remaining} remaining)")
    print(f"  Timesteps: {args.timesteps:,} per trial")
    print(f"  Timeout:   25 min per trial (prevents hangs)")
    print(f"  Est. time: ~{n_remaining * 12 // 60}h {n_remaining * 12 % 60}m"
          f" (at ~12 min/trial)")
    print()
    print(f"  Optuna objective (minimise — evaluation, not training):")
    print(f"    mean_wait_all + {OBJ_WEIGHT_RIDE}*mean_ride"
        f" + {OBJ_WEIGHT_P95}*p95_wait")
    print(f"    Rejection handled by mean_wait_all (rejected = {MAX_WAIT:.0f} min wait)")
    print(f"    Rejected = {MAX_WAIT:.0f} min wait. No hard service floor.")
    print()
    print(f"  Searched (10 parameters):")
    print(f"    Reward weights: w_wait(1.0-4.0), w_ride(0.5-2.0),")
    print(f"                    w_detour(0.0-1.5), w_rejection(3.0-10.0)")
    print(f"    PPO:            lr_start, lr_schedule, gamma, ent_coef,")
    print(f"                    n_steps, n_epochs")
    print(f"  Fixed (6 parameters):")
    print(f"    w_ride_sq=0.3, w_cost=0.2")
    print(f"    batch_size=128, vf_coef=1.0, gae_lambda=0.95, clip_range=0.2")
    print()
    print(f"  TensorBoard (task metrics added):")
    print(f"    eval/service_rate, eval/mean_wait, eval/mean_wait_all,")
    print(f"    eval/p95_wait, eval/mean_ride, eval/p95_ride,")
    print(f"    eval/mean_detour, eval/rejected, eval/rejection_rate")
    print(f"    tensorboard --logdir {tb_base}")
    print()
    print(f"  Greedy: wait={GREEDY_WAIT} | service={GREEDY_SERVICE:.1%}")
    print(f"  Run006: wait={RUN006_WAIT} | service=88.3%")
    print(f"  Resume: python rl_tune.py --resume")
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
    print("SEARCH COMPLETE")
    print("=" * 60)
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f} hours)")
    print(f"  Trials:     {len(valid)} valid / {len(study.trials)} total")
    print()
    print(f"  {'Metric':<20} {'Best':>10} {'Run006':>10} {'Greedy':>10}")
    print(f"  {'-'*52}")
    print(f"  {'score':<20} {best.value:>10.3f}")
    print(f"  {'service_rate':<20} {best_attrs.get('service_rate',0):>9.1%}"
          f" {'88.3%':>10} {'88.5%':>10}")
    print(f"  {'mean_wait_all':<20} {best_attrs.get('mean_wait_all',0):>9.2f}")
    print(f"  {'mean_wait_served':<20} {best_attrs.get('mean_wait_served',0):>9.2f}"
          f" {'10.3':>10} {'8.3':>10}")
    print(f"  {'mean_ride':<20} {best_attrs.get('mean_ride',0):>9.2f}"
          f" {'8.6':>10} {'8.1':>10}")
    print(f"  {'p95_wait':<20} {best_attrs.get('p95_wait',0):>9.2f}")
    print(f"  {'p95_ride':<20} {best_attrs.get('p95_ride',0):>9.2f}")
    print(f"  {'mean_detour':<20} {best_attrs.get('mean_detour',0):>9.2f}x")
    print(f"  {'rejected':<20} {best_attrs.get('rejected',0):>9.1f}"
          f" {'n/a':>10} {'~40':>10}")
    print()
    print(f"  Best config (trial #{best.number + 1}):")
    print(f"    Reward:  w_wait={best_params.get('w_wait',0):.3f}"
          f"  w_ride={best_params.get('w_ride',0):.3f}"
          f"  w_detour={best_params.get('w_detour',0):.3f}"
          f"  w_rej={best_params.get('w_rejection',0):.3f}")
    print(f"    Fixed:   w_ride_sq=0.3  w_cost=0.2"
          f"  batch=128  vf=1.0  lam=0.95  clip=0.2")
    print(f"    PPO:     lr={best_params.get('lr_start',0):.6f}"
          f"({best_params.get('lr_schedule','')})"
          f"  gamma={best_params.get('gamma',0):.4f}"
          f"  ent={best_params.get('ent_coef',0):.5f}")
    print(f"             steps={best_params.get('n_steps',0)}"
          f"  epochs={best_params.get('n_epochs',0)}")

    # Save best_config.json
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
        "net_arch":      [128, 128],
        "reward_mode":   "composite",
        "timesteps":     args.timesteps,
        "n_envs":        args.n_envs,
        # Context
        "source":                    "rl_tune.py Optuna TPE v2",
        "n_trials":                  len(study.trials),
        "n_valid_trials":            len(valid),
        "best_trial_number":         best.number + 1,
        "best_score":                round(best.value, 4),
        "baseline_greedy_wait":      GREEDY_WAIT,
        "baseline_greedy_service":   GREEDY_SERVICE,
        "baseline_run006_wait":      RUN006_WAIT,
        "baseline_run006_service":   0.883,
        "achieved_mean_wait_all":    round(best_attrs.get("mean_wait_all", 0), 2),
        "achieved_mean_wait_served": round(best_attrs.get("mean_wait_served", 0), 2),
        "achieved_service_rate":     round(best_attrs.get("service_rate", 0), 4),
        "achieved_mean_ride":        round(best_attrs.get("mean_ride", 0), 2),
        "achieved_p95_wait":         round(best_attrs.get("p95_wait", 0), 2),
        "achieved_p95_ride":         round(best_attrs.get("p95_ride", 0), 2),
        "achieved_mean_detour":      round(best_attrs.get("mean_detour", 0), 3),
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
        "rejected", "rejection_rate", "mean_reward", "train_time_s",
        # searched reward weights
        "w_wait", "w_ride", "w_detour", "w_rejection",
        # searched PPO params
        "lr_start", "lr_schedule", "gamma", "ent_coef",
        "n_steps", "n_epochs",
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
    print("\n" + "=" * 60)
    print("TOP 5 TRIALS:")
    print("=" * 60)
    print(f"  {'#':<4} {'score':>7} {'wait_all':>9} {'svc':>7}"
          f" {'ride':>6} {'det':>5} {'rej':>5} {'w_w':>5} {'w_r':>5}")
    print(f"  {'-'*65}")
    for t in top5:
        a = t.user_attrs
        p = t.params
        print(f"  {t.number+1:<4} {t.value:>7.3f}"
              f" {a.get('mean_wait_all',0):>8.2f}"
              f" {a.get('service_rate',0):>6.1%}"
              f" {a.get('mean_ride',0):>6.2f}"
              f" {a.get('mean_detour',0):>5.2f}"
              f" {a.get('rejected',0):>5.0f}"
              f" {p.get('w_wait',0):>5.2f}"
              f" {p.get('w_rejection',0):>5.2f}")

    print("\n  Next: python rl_train_from_tune.py")
    print(f"        tensorboard --logdir {tb_base}")


if __name__ == "__main__":
    main()