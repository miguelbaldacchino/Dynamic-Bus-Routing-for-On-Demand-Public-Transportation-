#!/usr/bin/env python3
# rl_train_from_tune_v6.py
# Retrain MaskablePPO using the best config from rl_tune_v6.py.
#
# Key differences from v5 train script:
#   - Uses DARPEnvV6.make() — v6 reward (w_wait_all, balance, growing rejection)
#   - Training noise from config (train_noise=0.10)
#   - USE_V6_FEATURES=True — 12 per-vehicle obs features
#   - Eval uses DARPEnvV6 with tuned weights (not hardcoded defaults)
#   - Eval noise=0.0 (consistent with tuning, comparable to v1-v5)
#   - Baseline updated to seeds 100-104 greedy+ts values
#
# TensorBoard groups (identical to v5 for cross-version comparison):
#   eval/rl_ts_mean_wait        — primary metric
#   eval/standalone_mean_wait   — synergy gap partner
#   eval/rl_ts_ts_improvements  — TS contribution
#   eval/synergy_gap_wait       — TS contribution in minutes
#   eval/rl_ts_service_rate
#   eval/rl_ts_rejected
#   eval/rl_ts_gap_vs_baseline
#   eval/rl_ts_score
#   eval/load_std
#   eval/rl_ts_p95_wait
#   rollout/ep_rew_mean
#   train/entropy_loss
#   train/approx_kl
#
# Usage:
#   python rl_train_from_tune_v6.py
#   python rl_train_from_tune_v6.py --config rl_outputs/tune_v6/best_config.json
#   python rl_train_from_tune_v6.py --timesteps 1000000
#   python rl_train_from_tune_v6.py --eval-freq 50000 --eval-eps 5

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Retrain with v6 Optuna best config")
    parser.add_argument("--config",          default="rl_outputs/tune_v6/best_config.json")
    parser.add_argument("--timesteps",       type=int, default=None)
    parser.add_argument("--n-envs",          type=int, default=None)
    parser.add_argument("--eval-freq",       type=int, default=50_000)
    parser.add_argument("--eval-eps",        type=int, default=5,
                        help="Episodes per eval call (default 5)")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--out",             default="rl_outputs",
                        help="Base output directory (default: rl_outputs)")
    parser.add_argument("--device",          default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: {args.config} not found. Run rl_tune_v6.py first.")
        sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    timesteps = args.timesteps or cfg.get("timesteps", 400_000)
    n_envs    = args.n_envs    or cfg.get("n_envs",    6)

    print("=" * 65)
    print("Retraining — v6 (richer obs + noise training)")
    print("=" * 65)
    print(f"  Source:    {cfg.get('source', 'unknown')}")
    print(f"  Trial:     #{cfg.get('best_trial_number', '?')}"
          f" of {cfg.get('n_trials', '?')}"
          f" ({cfg.get('n_valid_trials', '?')} valid)")
    print(f"  Score:     {cfg.get('best_score', '?')}"
          f"  (greedy+ts: {cfg.get('greedy_ts_score', '?')})")
    print()
    print(f"  Tune results (RL+TS):")
    print(f"    mean_wait:    {cfg.get('achieved_rl_ts_mean_wait', '?')}")
    print(f"    wait_all:     {cfg.get('achieved_rl_ts_mean_wait_all', '?')}")
    print(f"    p95_wait:     {cfg.get('achieved_rl_ts_p95_wait', '?')}")
    print(f"    service_rate: {cfg.get('achieved_rl_ts_service_rate', 0):.1%}")
    print(f"    rejected:     {cfg.get('achieved_rl_ts_rejected', '?')}")
    print(f"  Standalone RL:")
    print(f"    service_rate: {cfg.get('achieved_standalone_svc', 0):.1%}")
    print(f"    mean_wait:    {cfg.get('achieved_standalone_wait', '?')}")
    print(f"  Greedy+TS baseline (seeds 100-104):")
    print(f"    mean_wait:    {cfg.get('greedy_ts_mean_wait', 13.9585)}")
    print()
    print(f"  Reward weights (v6):")
    print(f"    w_acceptance = 1.000  (fixed)")
    print(f"    w_wait       = {cfg.get('w_wait', '?'):.4f}  (tuned)")
    print(f"    w_wait_all   = {cfg.get('w_wait_all', '?'):.4f}  (tuned — mean_wait_all penalty)")
    print(f"    w_ride       = 0.800  (fixed)")
    print(f"    w_cost       = 0.100  (fixed)")
    print(f"    w_rejection  = {cfg.get('w_rejection', '?'):.4f}  (tuned, growing penalty)")
    print(f"    w_imbalance  = {cfg.get('w_imbalance', '?'):.4f}  (tuned — workload balance)")
    print()
    print(f"  Environment:")
    print(f"    train_noise  = {cfg.get('train_noise', 0.10)}")
    print(f"    eval_noise   = {cfg.get('eval_noise',  0.0)}")
    print(f"    use_v6_features = {cfg.get('use_v6_features', True)}")
    print()
    print(f"  PPO hyperparameters:")
    print(f"    lr_start    = {cfg['lr_start']:.6f}  (linear schedule)")
    print(f"    gamma       = {cfg['gamma']:.4f}")
    print(f"    ent_coef    = {cfg['ent_coef']:.6f}")
    print(f"    n_steps     = {cfg['n_steps']}")
    print(f"    batch_size  = {cfg['batch_size']}")
    print(f"    n_epochs    = {cfg['n_epochs']}")
    print(f"    vf_coef     = {cfg['vf_coef']:.4f}")
    print(f"    gae_lambda  = {cfg['gae_lambda']:.4f}")
    print(f"    clip_range  = {cfg['clip_range']}")
    print(f"    net_arch    = {cfg.get('net_arch', [256, 128])}")
    print(f"    norm_obs    = {cfg.get('norm_obs', False)}")
    print(f"    norm_reward = {cfg.get('norm_reward', True)}")
    print(f"    timesteps   = {timesteps:,}  |  n_envs = {n_envs}")
    print("=" * 65)

    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback

    from config import SimulationConfig
    import rl_env_v6 as _rl_env_v6
    from rl_tune_v6 import (
        DARPEnvV6,
        _eval_rl_ts,
        _eval_standalone,
        GREEDY_TS_MEAN_WAIT,
        GREEDY_TS_SCORE,
        compute_objective,
    )
    from rl_train import make_run_dir

    # Enable v6 features globally
    _rl_env_v6.USE_V6_FEATURES           = True
    _rl_env_v6.USE_ANTICIPATORY_FEATURES = False

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"\n  Device: {device}\n")

    train_noise = cfg.get("train_noise", 0.10)
    eval_noise  = cfg.get("eval_noise",  0.0)

    w_wait      = cfg.get("w_wait",      2.876)
    w_wait_all  = cfg.get("w_wait_all",  1.0)
    w_rejection = cfg.get("w_rejection", 7.223)
    w_imbalance = cfg.get("w_imbalance", 0.675)

    sim_cfg = SimulationConfig(
        seed=42, fleet_size=6, vehicle_capacity=16,
        depot_node=0, n_requests=400, demand_profile="malta",
        stochastic_arrivals=True, travel_noise=train_noise, n_nodes=71,
    )

    def make_env(seed: int):
        def _init():
            _rl_env_v6.USE_V6_FEATURES = True
            env_cfg = SimulationConfig(
                seed=seed, fleet_size=sim_cfg.fleet_size,
                vehicle_capacity=sim_cfg.vehicle_capacity,
                depot_node=sim_cfg.depot_node, n_requests=sim_cfg.n_requests,
                demand_profile=sim_cfg.demand_profile,
                stochastic_arrivals=sim_cfg.stochastic_arrivals,
                travel_noise=train_noise, n_nodes=sim_cfg.n_nodes,
            )
            return DARPEnvV6.make(env_cfg, w_wait, w_wait_all, w_rejection, w_imbalance)
        return _init

    vec_env = SubprocVecEnv([make_env(42 + i) for i in range(n_envs)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = cfg.get("norm_obs",    False),
        norm_reward = cfg.get("norm_reward", True),
        clip_obs    = 10.0, clip_reward = 10.0,
        gamma       = cfg["gamma"],
    )

    n_steps    = cfg["n_steps"]
    batch_size = cfg["batch_size"]
    buffer     = n_steps * n_envs
    if buffer % batch_size != 0:
        valid_bs = [b for b in [64, 128, 256] if buffer % b == 0 and b <= batch_size]
        batch_size = max(valid_bs) if valid_bs else 64
        print(f"  batch_size snapped: {cfg['batch_size']} -> {batch_size}")

    lr_start      = cfg["lr_start"]
    learning_rate = lambda progress: lr_start * progress
    print(f"  LR: linear {lr_start:.6f} -> 0.0")

    run_dir  = make_run_dir(args.out)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    eval_freq_steps = args.eval_freq       // n_envs
    ckpt_freq_steps = args.checkpoint_freq // n_envs

    # Eval cfg — noise=0.0 for comparability with v1-v5
    eval_cfg = SimulationConfig(
        seed=42, fleet_size=6, vehicle_capacity=16,
        depot_node=0, n_requests=400, demand_profile="malta",
        stochastic_arrivals=True, travel_noise=eval_noise, n_nodes=71,
    )

    class EvalAndCheckpointCallback(BaseCallback):
        def __init__(self, eval_cfg, eval_freq, eval_eps,
                     ckpt_dir, ckpt_freq, vec_env_ref):
            super().__init__(verbose=0)
            self.eval_cfg    = eval_cfg
            self.eval_freq   = eval_freq
            self.eval_eps    = eval_eps
            self.ckpt_dir    = ckpt_dir
            self.ckpt_freq   = ckpt_freq
            self.vec_env_ref = vec_env_ref
            self.best_score  = float("inf")
            self._periodic   = []

        def _save_checkpoint(self, label):
            path = os.path.join(self.ckpt_dir, label)
            os.makedirs(path, exist_ok=True)
            self.model.save(os.path.join(path, "model"))
            self.vec_env_ref.save(os.path.join(path, "vec_normalize.pkl"))

        def _prune_periodic(self, keep=3):
            while len(self._periodic) > keep:
                old = self._periodic.pop(0)
                for fname in ("model.zip", "vec_normalize.pkl"):
                    p = os.path.join(self.ckpt_dir, old, fname)
                    if os.path.exists(p): os.remove(p)

        def _on_step(self) -> bool:
            if self.n_calls % self.ckpt_freq == 0:
                label = f"step_{self.num_timesteps:08d}"
                self._save_checkpoint(label)
                self._periodic.append(label)
                self._prune_periodic(keep=3)
                print(f"\n  [Checkpoint] periodic -> {label}")

            if self.n_calls % self.eval_freq != 0:
                return True

            rl_ts = _eval_rl_ts(
                self.model, self.eval_cfg, self.eval_eps,
                w_wait, w_wait_all, w_rejection, w_imbalance,
            )
            svc  = rl_ts.get("service_rate")    or 0.0
            mw   = rl_ts.get("mean_wait")       or 0.0
            rej  = rl_ts.get("rejected")        or 0.0
            p95w = rl_ts.get("p95_wait")        or 0.0
            mr   = rl_ts.get("mean_ride")       or 0.0
            tsim = rl_ts.get("ts_improvements") or 0.0

            try:
                env0  = self.training_env.envs[0]
                loads = [len(v.onboard) + len(v.plan)//2
                         for v in env0._vehicles.values()]
                lstd  = float(np.std(loads)) if loads else 0.0
            except Exception:
                lstd = 0.0

            _, obj_metrics = compute_objective(rl_ts, self.eval_cfg.n_requests)
            score = obj_metrics["score"]

            self.logger.record("eval/rl_ts_mean_wait",       mw)
            self.logger.record("eval/rl_ts_ts_improvements", tsim)
            self.logger.record("eval/rl_ts_service_rate",    svc)
            self.logger.record("eval/rl_ts_rejected",        rej)
            self.logger.record("eval/rl_ts_gap_vs_baseline", mw - GREEDY_TS_MEAN_WAIT)
            self.logger.record("eval/rl_ts_score",           score)
            self.logger.record("eval/load_std",              lstd)
            self.logger.record("eval/rl_ts_p95_wait",        p95w)
            self.logger.record("eval/rl_ts_mean_ride",       mr)

            sa    = _eval_standalone(
                self.model, self.eval_cfg, self.eval_eps,
                w_wait, w_wait_all, w_rejection, w_imbalance,
            )
            sa_sr = sa.get("service_rate") or 0.0
            sa_mw = sa.get("mean_wait")    or 0.0
            sa_rj = sa.get("rejected")     or 0.0

            self.logger.record("eval/standalone_mean_wait",    sa_mw)
            self.logger.record("eval/standalone_service_rate", sa_sr)
            self.logger.record("eval/standalone_rejected",     sa_rj)
            self.logger.record("eval/synergy_gap_wait",        sa_mw - mw)
            self.logger.dump(self.num_timesteps)

            print(f"\n  [Eval @ {self.num_timesteps:,}]"
                  f"  rl+ts: svc={svc:.1%}  wait={mw:.2f}  p95={p95w:.1f}"
                  f"  rej={rej:.0f}  score={score:.3f}"
                  f"  delta={score - GREEDY_TS_SCORE:+.3f}"
                  f"  ts_impr={tsim:.0f}  load_std={lstd:.2f}"
                  f"  |  sa: svc={sa_sr:.1%}  wait={sa_mw:.2f}"
                  f"  synergy={sa_mw - mw:+.2f}")

            if score < self.best_score:
                self.best_score = score
                self._save_checkpoint("best")
                print(f"  [Checkpoint] NEW BEST score={score:.3f}"
                      f"  wait={mw:.2f}  svc={svc:.1%}  rej={rej:.0f}"
                      f"  delta={score - GREEDY_TS_SCORE:+.3f}"
                      f"  -> checkpoints/best/")

            return True

    model = MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate = learning_rate,
        gamma         = cfg["gamma"],
        ent_coef      = cfg["ent_coef"],
        n_steps       = n_steps,
        batch_size    = batch_size,
        n_epochs      = cfg["n_epochs"],
        vf_coef       = cfg["vf_coef"],
        gae_lambda    = cfg["gae_lambda"],
        clip_range    = cfg["clip_range"],
        max_grad_norm = cfg.get("max_grad_norm", 0.5),
        policy_kwargs = dict(net_arch=cfg.get("net_arch", [256, 128])),
        verbose       = 1,
        device        = device,
        tensorboard_log = os.path.join(run_dir, "tb_logs"),
    )

    callback = EvalAndCheckpointCallback(
        eval_cfg    = eval_cfg,
        eval_freq   = eval_freq_steps,
        eval_eps    = args.eval_eps,
        ckpt_dir    = ckpt_dir,
        ckpt_freq   = ckpt_freq_steps,
        vec_env_ref = vec_env,
    )

    print(f"\nTraining for {timesteps:,} steps  ({n_envs} envs)...")
    print(f"  Eval every {args.eval_freq:,} steps  ({args.eval_eps} eps, RL+TS + standalone)")
    print(f"  Periodic checkpoint every {args.checkpoint_freq:,} steps")
    print(f"  Best-score checkpoint -> {ckpt_dir}/best/")
    print()

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    train_time = time.time() - t0

    best_ckpt = os.path.join(ckpt_dir, "best", "model.zip")
    if os.path.exists(best_ckpt):
        print(f"\nLoading best checkpoint for final eval...")
        eval_model = MaskablePPO.load(best_ckpt, device=device)
        print(f"  (best score during training: {callback.best_score:.3f})")
    else:
        print(f"\nNo best checkpoint — using end-of-training model.")
        eval_model = model

    print("Final evaluation (10 episodes, RL+TS)...")
    rl_ts_final = _eval_rl_ts(eval_model, eval_cfg, 10,
                               w_wait, w_wait_all, w_rejection, w_imbalance)
    print("Final evaluation (10 episodes, standalone)...")
    sa_final    = _eval_standalone(eval_model, eval_cfg, 10,
                                   w_wait, w_wait_all, w_rejection, w_imbalance)

    svc  = rl_ts_final.get("service_rate")    or 0.0
    mw   = rl_ts_final.get("mean_wait")       or 0.0
    rej  = rl_ts_final.get("rejected")        or 0.0
    p95w = rl_ts_final.get("p95_wait")        or 0.0
    mr   = rl_ts_final.get("mean_ride")       or 0.0
    tsim = rl_ts_final.get("ts_improvements") or 0.0

    _, obj      = compute_objective(rl_ts_final, eval_cfg.n_requests)
    score_final = obj["score"]
    mw_all      = obj["rl_ts_mean_wait_all"]
    synergy     = (sa_final.get("mean_wait") or 0) - mw

    print(f"\n{'=' * 65}")
    print("FINAL RESULTS  (from best checkpoint)")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<28} {'This run':>10} {'Greedy+TS':>10} {'Gap':>8}")
    print(f"  {'-' * 58}")
    print(f"  {'service_rate':<28} {svc:>9.1%} {0.784:>9.1%} {svc-0.784:>+8.1%}")
    print(f"  {'mean_wait (RL+TS)':<28} {mw:>10.2f} {GREEDY_TS_MEAN_WAIT:>10.2f} {mw-GREEDY_TS_MEAN_WAIT:>+8.2f}")
    print(f"  {'mean_wait_all':<28} {mw_all:>10.2f}")
    print(f"  {'p95_wait':<28} {p95w:>10.2f}")
    print(f"  {'mean_ride':<28} {mr:>10.2f}")
    print(f"  {'rejected':<28} {rej:>10.0f}")
    print(f"  {'ts_improvements':<28} {tsim:>10.0f}")
    print(f"  {'v6_score':<28} {score_final:>10.3f} {GREEDY_TS_SCORE:>10.3f} {score_final-GREEDY_TS_SCORE:>+8.3f}")
    print(f"\n  Standalone RL:")
    print(f"    service_rate: {sa_final.get('service_rate', 0):.1%}")
    print(f"    mean_wait:    {sa_final.get('mean_wait', 0):.2f}")
    print(f"    rejected:     {sa_final.get('rejected', 0):.0f}")
    print(f"    synergy_gap:  {synergy:+.2f} min  (TS contribution)")
    print(f"\n  Training time: {train_time:.0f}s ({train_time/60:.1f} min)")

    model.save(os.path.join(run_dir, "model_final"))
    vec_env.save(os.path.join(run_dir, "vec_normalize_final.pkl"))

    summary = {
        "source":      "rl_train_from_tune_v6.py",
        "tune_config": cfg,
        "final_rl_ts": {
            "service_rate":         round(svc,        4),
            "mean_wait":            round(mw,         2),
            "mean_wait_all":        round(mw_all,      2),
            "p95_wait":             round(p95w,        2),
            "mean_ride":            round(mr,          2),
            "rejected":             round(rej,         1),
            "ts_improvements":      round(tsim,        1),
            "v6_score":             round(score_final, 3),
            "gap_vs_greedy_wait":   round(mw   - GREEDY_TS_MEAN_WAIT, 2),
            "gap_vs_greedy_score":  round(score_final - GREEDY_TS_SCORE, 3),
        },
        "final_standalone": {
            "service_rate": round(sa_final.get("service_rate") or 0, 4),
            "mean_wait":    round(sa_final.get("mean_wait")    or 0, 2),
            "rejected":     round(sa_final.get("rejected")     or 0, 1),
            "synergy_gap":  round(synergy, 2),
        },
        "best_score_during_training": round(callback.best_score, 3),
        "training_time_seconds":      round(train_time, 1),
    }
    with open(os.path.join(run_dir, "tune_retrain_summary.json"), "w") as f:
        import json as _json
        _json.dump(summary, f, indent=2)

    print(f"\n  Best model:  {ckpt_dir}/best/model.zip")
    print(f"  Final model: {run_dir}/model_final.zip")
    print(f"  Summary:     {run_dir}/tune_retrain_summary.json")
    print(f"  TB:          tensorboard --logdir {run_dir}/tb_logs")
    vec_env.close()


if __name__ == "__main__":
    main()
