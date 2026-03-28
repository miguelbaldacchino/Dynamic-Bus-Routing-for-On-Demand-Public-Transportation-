#!/usr/bin/env python3
# rl_train_from_tune.py
# Retrain MaskablePPO using the best config found by rl_tune.py.
# Passes reward weights into DARPEnv and logs full metrics to TensorBoard.
#
# Usage:
#   python rl_train_from_tune.py
#   python rl_train_from_tune.py --timesteps 1000000

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Retrain with Optuna best config")
    parser.add_argument("--config", default="rl_outputs/tune_results/best_config.json")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs",    type=int, default=4)
    parser.add_argument("--device",    default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: {args.config} not found. Run rl_tune.py first.")
        sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    timesteps = args.timesteps or cfg.get("timesteps", 500_000)
    n_envs    = args.n_envs

    print("=" * 60)
    print("Retraining with Optuna best config")
    print("=" * 60)
    print(f"  Source:  {cfg.get('source', 'unknown')}")
    print(f"  Trial:   #{cfg.get('best_trial_number','?')}"
          f" of {cfg.get('n_trials','?')}"
          f" ({cfg.get('n_valid_trials','?')} valid)")
    print(f"  Score:   {cfg.get('best_score','?')}")
    print()
    print(f"  Tune results:")
    print(f"    mean_wait_all:    {cfg.get('achieved_mean_wait_all','?')}")
    print(f"    mean_wait_served: {cfg.get('achieved_mean_wait_served','?')}")
    print(f"    service_rate:     {cfg.get('achieved_service_rate',0):.1%}")
    print(f"    mean_ride:        {cfg.get('achieved_mean_ride','?')}")
    print(f"    p95_wait:         {cfg.get('achieved_p95_wait','?')}")
    print(f"    p95_ride:         {cfg.get('achieved_p95_ride','?')}")
    print(f"    mean_detour:      {cfg.get('achieved_mean_detour','?')}x")
    print()
    print(f"  Reward weights:")
    print(f"    w_acceptance = {cfg.get('w_acceptance',1.0):.3f}")
    print(f"    w_wait       = {cfg.get('w_wait',2.0):.3f}  (orig: 2.0)")
    print(f"    w_ride       = {cfg.get('w_ride',1.0):.3f}  (orig: 1.0)")
    print(f"    w_ride_sq    = {cfg.get('w_ride_sq',0.5):.3f}  (orig: 0.0 — new)")
    print(f"    w_detour     = {cfg.get('w_detour',0.5):.3f}  (orig: 0.0 — new)")
    print(f"    w_cost       = {cfg.get('w_cost',0.5):.3f}  (orig: 0.5)")
    print(f"    w_rejection  = {cfg.get('w_rejection',5.0):.3f}  (orig: 5.0)")
    print()
    print(f"  PPO hyperparameters:")
    print(f"    lr_start    = {cfg['lr_start']:.6f}")
    print(f"    lr_schedule = {cfg['lr_schedule']}")
    print(f"    gamma       = {cfg['gamma']:.4f}")
    print(f"    ent_coef    = {cfg['ent_coef']:.6f}")
    print(f"    n_steps     = {cfg['n_steps']}")
    print(f"    batch_size  = {cfg['batch_size']}")
    print(f"    n_epochs    = {cfg['n_epochs']}")
    print(f"    vf_coef     = {cfg['vf_coef']:.4f}")
    print(f"    gae_lambda  = {cfg['gae_lambda']:.4f}")
    print(f"    clip_range  = {cfg['clip_range']}")
    print(f"    timesteps   = {timesteps:,}")
    print("=" * 60)

    import torch
    import numpy as np
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback

    from config import SimulationConfig
    from rl_env import DARPEnv
    from rl_train import make_run_dir
    from rl_tune import evaluate_policy, MAX_WAIT

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\n  Device: {device}\n")

    sim_cfg = SimulationConfig(
        seed=42, fleet_size=6, vehicle_capacity=16,
        depot_node=0, n_requests=400, demand_profile="malta",
        stochastic_arrivals=True, travel_noise=0.0, n_nodes=71,
    )

    def make_env(seed: int):
        def _init():
            env_cfg = SimulationConfig(
                seed=seed, fleet_size=sim_cfg.fleet_size,
                vehicle_capacity=sim_cfg.vehicle_capacity,
                depot_node=sim_cfg.depot_node,
                n_requests=sim_cfg.n_requests,
                demand_profile=sim_cfg.demand_profile,
                stochastic_arrivals=sim_cfg.stochastic_arrivals,
                travel_noise=0.0, n_nodes=sim_cfg.n_nodes,
            )
            return DARPEnv(
                cfg          = env_cfg,
                reward_mode  = cfg["reward_mode"],
                w_acceptance = cfg.get("w_acceptance", 1.0),
                w_wait       = cfg.get("w_wait",       2.0),
                w_ride       = cfg.get("w_ride",       1.0),
                w_ride_sq    = cfg.get("w_ride_sq",    0.5),
                w_detour     = cfg.get("w_detour",     0.5),
                w_cost       = cfg.get("w_cost",       0.5),
                w_rejection  = cfg.get("w_rejection",  5.0),
            )
        return _init

    vec_env = SubprocVecEnv([make_env(42 + i) for i in range(n_envs)])
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=cfg["gamma"],
    )

    # Resolve batch_size
    n_steps    = cfg["n_steps"]
    batch_size = cfg["batch_size"]
    buffer     = n_steps * n_envs
    if buffer % batch_size != 0:
        valid = [b for b in [64, 128, 256] if buffer % b == 0 and b <= batch_size]
        batch_size = max(valid) if valid else 64
        print(f"  batch_size snapped: {cfg['batch_size']} -> {batch_size}")

    # LR schedule
    lr_start    = cfg["lr_start"]
    lr_schedule = cfg["lr_schedule"]
    if lr_schedule == "linear":
        learning_rate = lambda progress: lr_start * progress
        print(f"  LR: linear {lr_start:.6f} -> 0.0")
    else:
        learning_rate = lr_start
        print(f"  LR: constant {lr_start:.6f}")

    run_dir = make_run_dir("rl_outputs")

    # Eval callback — logs task metrics to TensorBoard every 50k steps
    class FullEvalCallback(BaseCallback):
        def __init__(self, sim_cfg, eval_freq):
            super().__init__(verbose=0)
            self.sim_cfg   = sim_cfg
            self.eval_freq = eval_freq

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                _, metrics = evaluate_policy(
                    self.model, self.sim_cfg, n_episodes=5
                )
                sr  = metrics["service_rate"] or 0
                rej = metrics["rejected"] or 0
                mw  = metrics["mean_wait"] or 0
                n   = self.sim_cfg.n_requests
                mw_all = ((sr * n * mw) + (rej * MAX_WAIT)) / max(sr * n + rej, 1)
                rr     = rej / max(n, 1)

                self.logger.record("eval/service_rate",   sr)
                self.logger.record("eval/mean_wait",      mw)
                self.logger.record("eval/mean_wait_all",  mw_all)
                self.logger.record("eval/p95_wait",       metrics["p95_wait"]    or 0)
                self.logger.record("eval/mean_ride",      metrics["mean_ride"]   or 0)
                self.logger.record("eval/p95_ride",       metrics["p95_ride"]    or 0)
                self.logger.record("eval/mean_detour",    metrics["mean_detour"] or 0)
                self.logger.record("eval/rejected",       rej)
                self.logger.record("eval/rejection_rate", rr)
                self.logger.dump(self.num_timesteps)

                print(f"\n  [Eval @ {self.num_timesteps:,}]"
                      f"  svc={sr:.1%}"
                      f"  wait={mw:.2f}"
                      f"  wait_all={mw_all:.2f}"
                      f"  ride={metrics['mean_ride'] or 0:.2f}"
                      f"  rej={rej:.0f}"
                      f"  | greedy: wait=8.3 svc=88.5%")
            return True

    model = MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate   = learning_rate,
        gamma           = cfg["gamma"],
        ent_coef        = cfg["ent_coef"],
        n_steps         = n_steps,
        batch_size      = batch_size,
        n_epochs        = cfg["n_epochs"],
        vf_coef         = cfg["vf_coef"],
        gae_lambda      = cfg["gae_lambda"],
        clip_range      = cfg["clip_range"],
        max_grad_norm   = cfg.get("max_grad_norm", 0.5),
        policy_kwargs   = dict(net_arch=cfg.get("net_arch", [128, 128])),
        verbose         = 1,
        device          = device,
        tensorboard_log = os.path.join(run_dir, "tb_logs"),
    )

    print(f"\nTraining for {timesteps:,} steps...")
    t0 = time.time()
    model.learn(
        total_timesteps = timesteps,
        callback        = FullEvalCallback(sim_cfg, 50_000 // n_envs),
        progress_bar    = True,
    )
    train_time = time.time() - t0

    print("\nFinal evaluation (10 episodes)...")
    results, final = evaluate_policy(model, sim_cfg, n_episodes=10)

    # Compute mean_wait_all for final report
    sr    = final["service_rate"] or 0
    rej   = final["rejected"] or 0
    mw    = final["mean_wait"] or 0
    n_req = sim_cfg.n_requests
    mw_all = ((sr * n_req * mw) + (rej * MAX_WAIT)) / max(sr * n_req + rej, 1)

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20} {'This run':>10} {'Run006':>10} {'Greedy':>10}")
    print(f"  {'-'*52}")
    print(f"  {'service_rate':<20} {sr:>9.1%} {'88.3%':>10} {'88.5%':>10}")
    print(f"  {'mean_wait_all':<20} {mw_all:>9.2f}")
    print(f"  {'mean_wait_served':<20} {mw:>9.2f} {'10.3':>10} {'8.3':>10}")
    print(f"  {'mean_ride':<20} {final['mean_ride'] or 0:>9.2f} {'8.6':>10} {'8.1':>10}")
    print(f"  {'p95_wait':<20} {final['p95_wait'] or 0:>9.2f}")
    print(f"  {'p95_ride':<20} {final['p95_ride'] or 0:>9.2f}")
    print(f"  {'mean_detour':<20} {final['mean_detour'] or 0:>9.2f}x")
    print(f"  {'rejected':<20} {rej:>9.0f} {'n/a':>10} {'~40':>10}")
    print(f"\n  Training time: {train_time:.0f}s ({train_time/60:.1f} min)")

    model.save(os.path.join(run_dir, "model"))
    vec_env.save(os.path.join(run_dir, "vec_normalize.pkl"))

    summary = {
        "source": "rl_train_from_tune.py",
        "tune_config": cfg,
        "final_results": {
            "service_rate":      round(sr, 4),
            "mean_wait_all":     round(mw_all, 2),
            "mean_wait_served":  round(mw, 2),
            "mean_ride":         round(final["mean_ride"] or 0, 2),
            "p95_wait":          round(final["p95_wait"]  or 0, 2),
            "p95_ride":          round(final["p95_ride"]  or 0, 2),
            "mean_detour":       round(final["mean_detour"] or 0, 3),
            "rejected":          round(rej, 1),
        },
        "training_time_seconds": round(train_time, 1),
        "wait_gap_vs_greedy":    round(mw - 8.3, 2),
        "wait_gap_vs_run006":    round(mw - 10.3, 2),
    }
    summary_path = os.path.join(run_dir, "tune_retrain_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Model:   {run_dir}/model.zip")
    print(f"  Summary: {summary_path}")
    print(f"  TB:      tensorboard --logdir {run_dir}/tb_logs")
    vec_env.close()


if __name__ == "__main__":
    main()