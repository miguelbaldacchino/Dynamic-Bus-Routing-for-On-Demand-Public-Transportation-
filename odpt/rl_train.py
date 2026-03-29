#!/usr/bin/env python3
# rl_train.py
# Train MaskablePPO for DARP vehicle assignment dispatch.
#
# Usage:
#   python rl_train.py                              # defaults
#   python rl_train.py --timesteps 500000           # longer training
#   python rl_train.py --n-envs 8                   # more parallel envs
#   python rl_train.py --reward-mode wait            # wait-time focused
#   python rl_train.py --device cuda                 # force GPU
#
# Outputs saved to rl_outputs/run_001/, run_002/, etc.
# Each run contains: model.zip, training_log.json, config.json

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from config import SimulationConfig
from rl_env import DARPEnv


def make_run_dir(base: str = "rl_outputs") -> str:
    """Create auto-numbered run directory: run_001, run_002, etc."""
    os.makedirs(base, exist_ok=True)
    existing = [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")
    ]
    nums = []
    for d in existing:
        try:
            nums.append(int(d.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(nums) + 1 if nums else 1
    run_dir = os.path.join(base, f"run_{next_num:03d}")
    os.makedirs(run_dir)
    return run_dir


def make_env(cfg: SimulationConfig, reward_mode: str, seed: int):
    """Factory for creating env instances (used by SubprocVecEnv)."""
    def _init():
        c = SimulationConfig(
            seed=seed,
            fleet_size=cfg.fleet_size,
            vehicle_capacity=cfg.vehicle_capacity,
            depot_node=cfg.depot_node,
            n_requests=cfg.n_requests,
            demand_profile=cfg.demand_profile,
            stochastic_arrivals=cfg.stochastic_arrivals,
            n_nodes=cfg.n_nodes,
            ride_factor=cfg.ride_factor,
            max_wait=cfg.max_wait,
            ride_time_margin=cfg.ride_time_margin,
            travel_noise=0.0,  # deterministic for training
            weights=cfg.weights,
        )
        env = DARPEnv(cfg=c, reward_mode=reward_mode)
        return env
    return _init


def run_greedy_episode(cfg: SimulationConfig) -> dict:
    """
    Run one episode where we always pick the vehicle with lowest
    insertion cost (greedy vehicle assignment).  This is the baseline.
    """
    env = DARPEnv(cfg=cfg, reward_mode="composite")
    obs, info = env.reset(seed=cfg.seed)
    done = False
    total_reward = 0.0

    while not done:
        mask = env.action_masks()
        # Pick lowest-cost feasible vehicle
        best_action = 0
        best_cost = float("inf")

        for act in range(1, env.n_actions):
            if mask[act] == 0:
                continue
            vid = env._vehicle_ids[act - 1]
            if vid in env._vehicle_insertions:
                _, _, cost = env._vehicle_insertions[vid]
                if cost < best_cost:
                    best_cost = cost
                    best_action = act

        # If all vehicles infeasible, reject
        if best_action == 0 and mask.sum() > 1:
            for act in range(1, env.n_actions):
                if mask[act] == 1:
                    best_action = act
                    break

        obs, reward, terminated, truncated, info = env.step(best_action)
        total_reward += reward
        done = terminated or truncated

    summary = env.episode_summary()
    summary["reward"] = total_reward
    return summary


def evaluate_model(model, cfg: SimulationConfig, n_episodes: int = 5) -> dict:
    """Evaluate trained model deterministically over multiple episodes."""
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
            n_nodes=cfg.n_nodes,
            ride_factor=cfg.ride_factor,
            max_wait=cfg.max_wait,
            ride_time_margin=cfg.ride_time_margin,
            travel_noise=0.0,
            weights=cfg.weights,
        )
        env = DARPEnv(cfg=eval_cfg)
        obs, info = env.reset(seed=eval_cfg.seed)
        done = False
        total_reward = 0.0

        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        summary = env.episode_summary()
        summary["reward"] = total_reward
        results.append(summary)

    # Aggregate
    return {
        "service_rate": np.mean([r["service_rate"] for r in results]),
        "mean_wait": np.mean([r["mean_wait"] for r in results if r["mean_wait"]]),
        "mean_ride": np.mean([r["mean_ride"] for r in results if r["mean_ride"]]),
        "p95_wait": np.mean([r["p95_wait"] for r in results if r["p95_wait"]]),
        "p95_ride": np.mean([r["p95_ride"] for r in results if r["p95_ride"]]),
        "reward": np.mean([r["reward"] for r in results]),
        "rejected": np.mean([r["rejected"] for r in results]),
    }


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO for DARP")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Total training timesteps (default: 200k)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel training environments (default: 4)")
    parser.add_argument("--fleet", type=int, default=6)
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--depot", type=int, default=0)
    parser.add_argument("--reward-mode", default="composite",
                        choices=["composite", "cost", "wait"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--eval-freq", type=int, default=50_000,
                        help="Evaluate every N timesteps")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", default="rl_outputs")
    args = parser.parse_args()

    # --- Imports (heavy, so deferred) ---
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    # --- Resolve device ---
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # --- Config ---
    cfg = SimulationConfig(
        seed=42,
        fleet_size=args.fleet,
        vehicle_capacity=16,
        depot_node=args.depot,
        n_requests=args.requests,
        demand_profile="malta",
        stochastic_arrivals=True,
        travel_noise=0.0,
        n_nodes=71,
    )

    # --- Run directory ---
    run_dir = make_run_dir(args.output_dir)

    # --- Save config ---
    run_config = {
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "fleet_size": args.fleet,
        "n_requests": args.requests,
        "depot_node": args.depot,
        "reward_mode": args.reward_mode,
        "lr": args.lr,
        "ent_coef": args.ent_coef,
        "gamma": args.gamma,
        "device": device,
        "eval_freq": args.eval_freq,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # --- Print banner ---
    print("=" * 60)
    print("MaskablePPO Training - DARP Vehicle Assignment")
    print("=" * 60)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Fleet: {args.fleet} buses, depot={args.depot}")
    print(f"  Requests: {args.requests}")
    print(f"  Reward: {args.reward_mode}")
    print(f"  LR={args.lr}, gamma={args.gamma}, ent={args.ent_coef}")
    print(f"  Output: {run_dir}/")
    print("=" * 60)

    # --- Greedy baseline ---
    print("\nRunning greedy baseline...")
    greedy_results = []
    for i in range(args.eval_episodes):
        gcfg = SimulationConfig(
            seed=1000 + i, fleet_size=args.fleet,
            vehicle_capacity=16, depot_node=args.depot,
            n_requests=args.requests, demand_profile="malta",
            stochastic_arrivals=True, travel_noise=0.0, n_nodes=71,
        )
        greedy_results.append(run_greedy_episode(gcfg))

    greedy_mean = {
        "service_rate": np.mean([r["service_rate"] for r in greedy_results]),
        "mean_wait": np.mean([r["mean_wait"] for r in greedy_results if r["mean_wait"]]),
        "mean_ride": np.mean([r["mean_ride"] for r in greedy_results if r["mean_ride"]]),
        "reward": np.mean([r["reward"] for r in greedy_results]),
    }
    print(f"  Greedy: service={greedy_mean['service_rate']:.1%}  "
          f"wait={greedy_mean['mean_wait']:.1f}  "
          f"ride={greedy_mean['mean_ride']:.1f}")

    # --- Create parallel envs ---
    env_fns = [
        make_env(cfg, args.reward_mode, seed=42 + i)
        for i in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)

    # Normalise rewards — this is critical for PPO when reward magnitudes
    # vary widely.  The critic can't learn when returns range from -700 to 0.
    # VecNormalize rescales rewards to ~zero mean, unit variance, making
    # the value function learnable.
    #
    # norm_obs=False: observations are already normalised to [-1, 1] by
    # _encode_state(). Double-normalising creates a train/eval mismatch
    # because evaluate_policy() uses raw DARPEnv without VecNormalize.
    from stable_baselines3.common.vec_env import VecNormalize
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,      # obs already [-1,1] from _encode_state
        norm_reward=True,    # normalise rewards (the key fix)
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=args.gamma,
    )

    # --- Evaluation callback ---
    class EvalCallback(BaseCallback):
        def __init__(self, eval_freq, eval_episodes, cfg, log, verbose=1):
            super().__init__(verbose)
            self.eval_freq = eval_freq
            self.eval_episodes = eval_episodes
            self.cfg = cfg
            self.log = log

        def _on_step(self):
            if self.n_calls % self.eval_freq == 0:
                results = evaluate_model(
                    self.model, self.cfg, self.eval_episodes
                )
                self.log["evaluations"].append({
                    "timestep": self.num_timesteps,
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in results.items()},
                })
                print(f"\n  [Eval @ {self.num_timesteps:,} steps]  "
                      f"service={results['service_rate']:.1%}  "
                      f"wait={results['mean_wait']:.1f}  "
                      f"ride={results['mean_ride']:.1f}  "
                      f"reward={results['reward']:.1f}")
            return True

    training_log = {
        "greedy_baseline": greedy_mean,
        "evaluations": [],
    }

    eval_cb = EvalCallback(
        eval_freq=args.eval_freq // args.n_envs,  # per-env steps
        eval_episodes=args.eval_episodes,
        cfg=cfg,
        log=training_log,
    )

    # --- Create model ---
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        n_steps=2048,         # longer rollouts -> better return estimates
        batch_size=128,       # smaller batches -> more gradient steps per rollout
        n_epochs=10,          # more passes over each rollout
        clip_range=0.2,
        vf_coef=1.0,          # up from 0.5 -> train critic harder
        max_grad_norm=0.5,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        device=device,
        tensorboard_log=os.path.join(run_dir, "tb_logs"),
    )

    # --- Train ---
    print(f"\nTraining for {args.timesteps:,} timesteps...")
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_cb,
        progress_bar=True,
    )
    train_time = time.time() - t0

    # --- Final evaluation ---
    print("\n\nFinal evaluation...")
    final_results = evaluate_model(model, cfg, n_episodes=10)
    training_log["final_evaluation"] = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in final_results.items()
    }

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  RL:     service={final_results['service_rate']:.1%}  "
          f"wait={final_results['mean_wait']:.1f}  "
          f"ride={final_results['mean_ride']:.1f}  "
          f"rej={final_results['rejected']:.0f}")
    print(f"  Greedy: service={greedy_mean['service_rate']:.1%}  "
          f"wait={greedy_mean['mean_wait']:.1f}  "
          f"ride={greedy_mean['mean_ride']:.1f}")
    print(f"  Training time: {train_time:.0f}s ({train_time/60:.1f} min)")

    # --- Save ---
    model_path = os.path.join(run_dir, "model")
    model.save(model_path)
    print(f"\n  Model saved: {model_path}.zip")

    # Save VecNormalize statistics (needed for evaluation with normalised obs)
    vec_norm_path = os.path.join(run_dir, "vec_normalize.pkl")
    vec_env.save(vec_norm_path)
    print(f"  VecNormalize saved: {vec_norm_path}")

    training_log["training_time_seconds"] = round(train_time, 1)
    log_path = os.path.join(run_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2, default=str)
    print(f"  Log saved: {log_path}")
    print(f"  Run directory: {run_dir}")

    vec_env.close()
    return model, training_log


if __name__ == "__main__":
    main()