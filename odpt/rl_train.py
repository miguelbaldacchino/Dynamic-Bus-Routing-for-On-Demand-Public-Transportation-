#!/usr/bin/env python3
# rl_train.py
# Training loop for the PPO DARP dispatch agent.
#
# Training protocol:
# -----------------
# 1. Collect rollouts: Run K episodes, accumulating transitions in the
#    PPO buffer.  Each transition is one dispatch decision (request arrival).
#
# 2. Update: Run PPO update with the collected transitions.
#
# 3. Evaluate: Every N updates, run deterministic evaluation episodes
#    and log metrics.
#
# 4. Curriculum: Optionally start with fewer requests per episode and
#    ramp up, so the agent first learns good insertions on easy instances
#    before facing the full 400-request day.
#
# Usage:
#   python rl_train.py                          # default config
#   python rl_train.py --episodes 500 --eval-every 20
#   python rl_train.py --curriculum              # enable curriculum

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from config import SimulationConfig
from rl_env import DARPEnv, OBS_SIZE, MAX_ACTIONS


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Training
    total_episodes:    int   = 1000
    episodes_per_update: int = 4       # rollouts before PPO update
    eval_every:        int   = 25      # evaluate every N updates
    eval_episodes:     int   = 5       # episodes per evaluation

    # Curriculum learning
    curriculum:        bool  = False
    curriculum_start:  int   = 50      # start with this many requests
    curriculum_end:    int   = 400     # ramp to full demand
    curriculum_ramp:   int   = 300     # ramp over this many episodes

    # PPO hyperparameters (passed to agent)
    lr:                float = 3e-4
    gamma:             float = 0.995   # high: future insertions matter
    gae_lambda:        float = 0.95
    clip_eps:          float = 0.2
    entropy_coef:      float = 0.02
    value_coef:        float = 0.5
    n_epochs:          int   = 4
    batch_size:        int   = 128
    hidden:            int   = 256

    # Simulation
    fleet_size:        int   = 10
    vehicle_capacity:  int   = 16
    n_requests:        int   = 400
    demand_profile:    str   = "malta"

    # Output
    save_dir:          str   = "rl_outputs"
    save_every:        int   = 100     # save model every N episodes


def make_sim_cfg(train_cfg: TrainConfig, n_requests: int = None) -> SimulationConfig:
    """Create a SimulationConfig for RL training."""
    return SimulationConfig(
        seed=42,
        fleet_size=train_cfg.fleet_size,
        vehicle_capacity=train_cfg.vehicle_capacity,
        n_requests=n_requests or train_cfg.n_requests,
        demand_profile=train_cfg.demand_profile,
        stochastic_arrivals=True,
        travel_noise=0.0,  # deterministic during training
    )


def curriculum_requests(episode: int, cfg: TrainConfig) -> int:
    """Compute number of requests for this episode under curriculum."""
    if not cfg.curriculum:
        return cfg.n_requests

    progress = min(episode / max(cfg.curriculum_ramp, 1), 1.0)
    return int(cfg.curriculum_start + progress * (cfg.curriculum_end - cfg.curriculum_start))


# ---------------------------------------------------------------------------
# Greedy baseline (for comparison)
# ---------------------------------------------------------------------------

def run_greedy_episode(env: DARPEnv) -> dict:
    """
    Run one episode using the greedy heuristic (always pick the
    lowest-cost feasible insertion).  Used as evaluation baseline.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        mask = info.get("action_mask", env.get_action_mask())

        # Greedy: evaluate all feasible actions, pick lowest cost
        best_action = 0  # default: reject
        best_cost = float("inf")

        for act_idx in range(len(mask)):
            if mask[act_idx] == 0:
                continue
            if act_idx == 0:
                # Reject: assign high cost
                continue
            if act_idx in env._action_map:
                vid, candidate, n_committed = env._action_map[act_idx]
                v = env._vehicles[vid]
                v_state = v.to_state_dict(env._sim_time)
                from feasibility import evaluate_plan
                cost = evaluate_plan(
                    candidate, v_state,
                    env._system_state, env.cfg.weights,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_action = act_idx

        if best_action == 0 and mask.sum() > 1:
            # There are feasible insertions but we picked reject;
            # pick any feasible insertion instead
            for act_idx in range(1, len(mask)):
                if mask[act_idx] == 1:
                    best_action = act_idx
                    break

        obs, reward, terminated, truncated, info = env.step(best_action)
        total_reward += reward
        done = terminated or truncated

    return {
        "reward": total_reward,
        **env.episode_summary(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(train_cfg: TrainConfig):
    """Main training loop."""
    from rl_agent import PPOAgent

    os.makedirs(train_cfg.save_dir, exist_ok=True)

    # Create agent
    agent = PPOAgent(
        obs_dim=OBS_SIZE,
        act_dim=MAX_ACTIONS,
        lr=train_cfg.lr,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        clip_eps=train_cfg.clip_eps,
        entropy_coef=train_cfg.entropy_coef,
        value_coef=train_cfg.value_coef,
        n_epochs=train_cfg.n_epochs,
        batch_size=train_cfg.batch_size,
        hidden=train_cfg.hidden,
    )

    # Training log
    log = {
        "config": {
            "total_episodes": train_cfg.total_episodes,
            "lr": train_cfg.lr,
            "gamma": train_cfg.gamma,
            "curriculum": train_cfg.curriculum,
            "fleet_size": train_cfg.fleet_size,
            "n_requests": train_cfg.n_requests,
        },
        "training": [],
        "evaluation": [],
    }

    episode = 0
    update_count = 0
    t_start = time.time()

    print("=" * 60)
    print("PPO Training — DARP Dispatch")
    print("=" * 60)
    print(f"  Episodes: {train_cfg.total_episodes}")
    print(f"  Fleet: {train_cfg.fleet_size} vehicles")
    print(f"  Requests: {train_cfg.n_requests}")
    print(f"  Curriculum: {train_cfg.curriculum}")
    print(f"  Output: {train_cfg.save_dir}/")
    print("=" * 60)

    while episode < train_cfg.total_episodes:
        # --- Collect rollouts ---
        batch_rewards = []
        batch_service_rates = []

        for _ in range(train_cfg.episodes_per_update):
            if episode >= train_cfg.total_episodes:
                break

            n_req = curriculum_requests(episode, train_cfg)
            sim_cfg = make_sim_cfg(train_cfg, n_requests=n_req)
            # Vary seed per episode for diverse training data
            sim_cfg.seed = 42 + episode

            env = DARPEnv(cfg=sim_cfg)
            obs, info = env.reset(seed=sim_cfg.seed)
            done = False
            ep_reward = 0.0

            while not done:
                mask = info.get("action_mask", env.get_action_mask())

                action, log_prob, value = agent.select_action(obs, mask)
                next_obs, reward, terminated, truncated, info = env.step(action)

                agent.buffer.store(
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    mask=mask,
                    done=terminated or truncated,
                )

                obs = next_obs
                ep_reward += reward
                done = terminated or truncated

            summary = env.episode_summary()
            batch_rewards.append(ep_reward)
            batch_service_rates.append(summary["service_rate"])
            episode += 1

        # --- PPO update ---
        update_metrics = agent.update()
        update_count += 1

        # Log training batch
        log_entry = {
            "episode": episode,
            "update": update_count,
            "mean_reward": float(np.mean(batch_rewards)),
            "mean_service_rate": float(np.mean(batch_service_rates)),
            **update_metrics,
        }
        log["training"].append(log_entry)

        elapsed = time.time() - t_start
        print(
            f"  Ep {episode:4d} | "
            f"reward={np.mean(batch_rewards):7.2f} | "
            f"service={np.mean(batch_service_rates):.1%} | "
            f"π_loss={update_metrics.get('policy_loss', 0):.4f} | "
            f"v_loss={update_metrics.get('value_loss', 0):.4f} | "
            f"H={update_metrics.get('entropy', 0):.3f} | "
            f"{elapsed:.0f}s"
        )

        # --- Evaluation ---
        if update_count % train_cfg.eval_every == 0:
            print(f"\n  --- Evaluation at episode {episode} ---")
            eval_cfg = make_sim_cfg(train_cfg)
            eval_results = {"rl": [], "greedy": []}

            for e in range(train_cfg.eval_episodes):
                eval_cfg.seed = 1000 + e  # fixed eval seeds
                env = DARPEnv(cfg=eval_cfg)

                # RL (deterministic)
                obs, info = env.reset(seed=eval_cfg.seed)
                done = False
                rl_reward = 0.0
                while not done:
                    mask = info.get("action_mask", env.get_action_mask())
                    action, _, _ = agent.select_action(
                        obs, mask, deterministic=True
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    rl_reward += reward
                    done = terminated or truncated
                rl_summary = env.episode_summary()
                rl_summary["reward"] = rl_reward
                eval_results["rl"].append(rl_summary)

                # Greedy baseline
                env2 = DARPEnv(cfg=eval_cfg)
                greedy_summary = run_greedy_episode(env2)
                eval_results["greedy"].append(greedy_summary)

            # Aggregate
            rl_mean = {
                "service_rate": np.mean([r["service_rate"] for r in eval_results["rl"]]),
                "mean_wait": np.mean([r["mean_wait"] for r in eval_results["rl"] if r["mean_wait"] is not None]),
                "reward": np.mean([r["reward"] for r in eval_results["rl"]]),
            }
            gr_mean = {
                "service_rate": np.mean([r["service_rate"] for r in eval_results["greedy"]]),
                "mean_wait": np.mean([r["mean_wait"] for r in eval_results["greedy"] if r["mean_wait"] is not None]),
                "reward": np.mean([r["reward"] for r in eval_results["greedy"]]),
            }

            print(f"    RL:     service={rl_mean['service_rate']:.1%}  "
                  f"wait={rl_mean['mean_wait']:.1f}  "
                  f"reward={rl_mean['reward']:.1f}")
            print(f"    Greedy: service={gr_mean['service_rate']:.1%}  "
                  f"wait={gr_mean['mean_wait']:.1f}  "
                  f"reward={gr_mean['reward']:.1f}")
            print()

            log["evaluation"].append({
                "episode": episode,
                "rl": rl_mean,
                "greedy": gr_mean,
            })

        # --- Save checkpoint ---
        if episode % train_cfg.save_every == 0:
            ckpt_path = os.path.join(
                train_cfg.save_dir, f"ppo_ep{episode:04d}.pt"
            )
            agent.save(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # --- Final save ---
    final_path = os.path.join(train_cfg.save_dir, "ppo_final.pt")
    agent.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    log_path = os.path.join(train_cfg.save_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"Training log saved: {log_path}")

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time:.0f}s "
          f"({total_time/60:.1f} min)")

    return agent, log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PPO for DARP dispatch")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--fleet", type=int, default=10)
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", default="rl_outputs")
    args = parser.parse_args()

    cfg = TrainConfig(
        total_episodes=args.episodes,
        eval_every=args.eval_every,
        curriculum=args.curriculum,
        fleet_size=args.fleet,
        n_requests=args.requests,
        lr=args.lr,
        save_dir=args.save_dir,
    )

    train(cfg)


if __name__ == "__main__":
    main()
