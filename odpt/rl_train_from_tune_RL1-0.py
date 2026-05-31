# rl_train_from_RL1-0.py
# Full retraining of MaskablePPO using best config from rl_tune_RL1-0.py.
# Canonical final trainer. Saves model_final.zip + TensorBoard logs.
#
# python rl_train_from_tune_RL1-0.py --config rl_tune_RL1-0_best.json

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Retrain with v4 Optuna best config")
    parser.add_argument("--config",     default="rl_outputs/tune_v4/best_config.json")
    parser.add_argument("--timesteps",  type=int,   default=None,
                        help="Override timesteps from config")
    parser.add_argument("--n-envs",     type=int,   default=None,
                        help="Override n_envs (defaults to config value)")
    parser.add_argument("--eval-freq",  type=int,   default=50_000,
                        help="Eval callback frequency in global steps (default 50k)")
    parser.add_argument("--eval-eps",   type=int,   default=3,
                        help="Episodes per eval-callback call (default 3; RL+TS is slow)")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                        help="Save a periodic checkpoint every N global steps "
                             "(default 100k). Best-score checkpoints are always "
                             "saved regardless of this setting.")
    parser.add_argument("--device",     default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: {args.config} not found. Run rl_tune_v4.py first.")
        sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    # ------------------------------------------------------------------ #
    # Resolve run parameters — CLI overrides cfg, cfg overrides defaults  #
    # ------------------------------------------------------------------ #
    timesteps = args.timesteps or cfg.get("timesteps", 300_000)
    n_envs    = args.n_envs    or cfg.get("n_envs",    6)

    print("=" * 60)
    print("Retraining — v4 TS-Initialiser architecture")
    print("=" * 60)
    print(f"  Source:  {cfg.get('source', 'unknown')}")
    print(f"  Trial:   #{cfg.get('best_trial_number', '?')}"
          f" of {cfg.get('n_trials', '?')}"
          f" ({cfg.get('n_valid_trials', '?')} valid)")
    print(f"  Score:   {cfg.get('best_score', '?')}")
    print()
    print(f"  Tune results (RL+TS):")
    print(f"    mean_wait:    {cfg.get('achieved_rl_ts_mean_wait', '?')}")
    print(f"    service_rate: {cfg.get('achieved_rl_ts_service_rate', 0):.1%}")
    print(f"    rejected:     {cfg.get('achieved_rl_ts_rejected', '?')}")
    print(f"  Standalone RL (no TS):")
    print(f"    service_rate: {cfg.get('achieved_standalone_svc', 0):.1%}")
    print(f"    mean_wait:    {cfg.get('achieved_standalone_wait', '?')}")
    print(f"  Greedy+TS baseline:")
    print(f"    mean_wait:    {cfg.get('greedy_ts_baseline_wait', 7.41)}")
    print(f"    service_rate: {cfg.get('greedy_ts_baseline_svc', 0.892):.1%}")
    print()
    print(f"  Reward weights (v4):")
    print(f"    w_acceptance = 1.000  (fixed)")
    print(f"    w_wait       = 1.000  (fixed)")
    print(f"    w_ride       = 0.500  (fixed)")
    print(f"    w_cost       = 0.100  (fixed, low — balance > cost)")
    print(f"    w_rejection  = {cfg.get('w_rejection',  3.22):.4f}  (tuned)")
    print(f"    w_imbalance  = {cfg.get('w_imbalance', 0.222):.4f}  (tuned — fleet balance)")
    print(f"    w_slack      = {cfg.get('w_slack',     0.427):.4f}  (tuned — plan headroom)")
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
    print(f"    net_arch    = {cfg.get('net_arch', [128, 128])}")
    print(f"    norm_obs    = {cfg.get('norm_obs', False)}")
    print(f"    norm_reward = {cfg.get('norm_reward', True)}")
    print(f"    timesteps   = {timesteps:,}")
    print(f"    n_envs      = {n_envs}")
    print("=" * 60)

    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback

    from config import SimulationConfig
    from rl_tune_v4 import (          # v4 is authoritative for this architecture
        DARPEnvV4,
        _evaluate_rl_plus_ts,
        _evaluate_rl_standalone,
        GREEDY_TS_WAIT,
        GREEDY_TS_SVC,
        MAX_WAIT,
    )
    from rl_train import make_run_dir

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

    # ------------------------------------------------------------------ #
    # FIX 1: use DARPEnvV4.make() — applies v4 reward patch              #
    # ------------------------------------------------------------------ #
    def make_env(seed: int):
        def _init():
            env_cfg = SimulationConfig(
                seed=seed,
                fleet_size=sim_cfg.fleet_size,
                vehicle_capacity=sim_cfg.vehicle_capacity,
                depot_node=sim_cfg.depot_node,
                n_requests=sim_cfg.n_requests,
                demand_profile=sim_cfg.demand_profile,
                stochastic_arrivals=sim_cfg.stochastic_arrivals,
                travel_noise=0.0,
                n_nodes=sim_cfg.n_nodes,
            )
            return DARPEnvV4.make(
                cfg          = env_cfg,
                w_rejection  = cfg.get("w_rejection",  3.22),
                w_imbalance  = cfg.get("w_imbalance",  0.222),
                w_slack      = cfg.get("w_slack",      0.427),
            )
        return _init

    vec_env = SubprocVecEnv([make_env(42 + i) for i in range(n_envs)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = cfg.get("norm_obs",    False),
        norm_reward = cfg.get("norm_reward", True),
        clip_obs    = 10.0,
        clip_reward = 10.0,
        gamma       = cfg["gamma"],
    )

    # Resolve batch_size against actual buffer size
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
    elif lr_schedule == "cosine":
        import math as _math
        learning_rate = lambda progress: (
            lr_start * 0.5 * (1 + _math.cos(_math.pi * (1 - progress)))
        )
        print(f"  LR: cosine {lr_start:.6f} -> ~0.0")
    elif lr_schedule == "warmup_cosine":
        import math as _math
        _warmup_frac = 0.1
        def learning_rate(progress):
            elapsed = 1.0 - progress
            if elapsed < _warmup_frac:
                return lr_start * (elapsed / _warmup_frac)
            cos_progress = (elapsed - _warmup_frac) / (1.0 - _warmup_frac)
            return lr_start * 0.5 * (1 + _math.cos(_math.pi * cos_progress))
        print(f"  LR: warmup_cosine {lr_start:.6f} (10% warmup)")
    else:
        learning_rate = lr_start
        print(f"  LR: constant {lr_start:.6f}")

    run_dir     = make_run_dir("rl_outputs")
    ckpt_dir    = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Convert global-step frequencies to per-env-step counts.
    # SB3 callbacks fire on n_calls which counts steps per environment,
    # so we must divide global targets by n_envs.
    eval_freq_steps = args.eval_freq       // n_envs
    ckpt_freq_steps = args.checkpoint_freq // n_envs

    # ------------------------------------------------------------------ #
    # FIX 2 + FIX 3: combined eval/checkpoint callback                   #
    #                                                                     #
    # Two responsibilities:                                               #
    #   A) Thesis-grade eval every --eval-freq steps                      #
    #      • RL+TS metrics  (primary — matches Optuna objective)          #
    #      • Standalone RL  (diagnostic — shows setter/finisher gap)      #
    #      • load_std       (fleet balance — proves w_imbalance effect)   #
    #   B) Best-score checkpoint — saved whenever rl_ts_score improves.  #
    #      Also saves a periodic checkpoint every --checkpoint-freq steps #
    #      as a safety net against policy collapse on long runs.          #
    #                                                                     #
    # Both the model weights AND the VecNormalize running stats are saved #
    # together so checkpoints are fully self-contained for reload.        #
    # ------------------------------------------------------------------ #

    class EvalAndCheckpointCallback(BaseCallback):
        """
        Unified eval + checkpoint callback for v4 TS-Initialiser training.

        Saving strategy
        ---------------
        best/   — updated whenever rl_ts_score (= mean_wait + 0.1*rejected)
                  improves.  This is the checkpoint to use for the thesis.
        step_*/ — periodic safety snapshots every --checkpoint-freq steps.
                  Keep the last 3; older ones are deleted automatically.
        """

        def __init__(self, sim_cfg, eval_freq, eval_eps, ckpt_dir,
                     ckpt_freq, vec_env_ref):
            super().__init__(verbose=0)
            self.sim_cfg     = sim_cfg
            self.eval_freq   = eval_freq
            self.eval_eps    = eval_eps
            self.ckpt_dir    = ckpt_dir
            self.ckpt_freq   = ckpt_freq
            self.vec_env_ref = vec_env_ref      # needed to save VecNormalize stats

            self.best_score        = float("inf")
            self._periodic_ckpts   = []         # track last 3 periodic saves

        # ---------------------------------------------------------------- #
        # Helpers                                                           #
        # ---------------------------------------------------------------- #
        def _save_checkpoint(self, label: str):
            """Save model weights + VecNormalize stats to ckpt_dir/label/."""
            path = os.path.join(self.ckpt_dir, label)
            os.makedirs(path, exist_ok=True)
            self.model.save(os.path.join(path, "model"))
            self.vec_env_ref.save(os.path.join(path, "vec_normalize.pkl"))

        def _prune_periodic_ckpts(self, keep: int = 3):
            """Delete oldest periodic checkpoints beyond `keep`."""
            while len(self._periodic_ckpts) > keep:
                old = self._periodic_ckpts.pop(0)
                old_model  = os.path.join(self.ckpt_dir, old, "model.zip")
                old_vecnrm = os.path.join(self.ckpt_dir, old, "vec_normalize.pkl")
                for p in (old_model, old_vecnrm):
                    if os.path.exists(p):
                        os.remove(p)

        # ---------------------------------------------------------------- #
        # Main callback                                                     #
        # ---------------------------------------------------------------- #
        def _on_step(self) -> bool:

            # ---- A) Periodic safety checkpoint -------------------------
            if self.n_calls % self.ckpt_freq == 0:
                label = f"step_{self.num_timesteps:08d}"
                self._save_checkpoint(label)
                self._periodic_ckpts.append(label)
                self._prune_periodic_ckpts(keep=3)
                print(f"\n  [Checkpoint] periodic save → {label}")

            # ---- B) Eval (less frequent than checkpoint) ----------------
            if self.n_calls % self.eval_freq != 0:
                return True

            # ---- RL+TS evaluation (primary — matches Optuna objective) --
            rl_ts = _evaluate_rl_plus_ts(self.model, self.sim_cfg,
                                         n_episodes=self.eval_eps)
            svc  = rl_ts.get("service_rate")    or 0.0
            mw   = rl_ts.get("mean_wait")       or 0.0
            rej  = rl_ts.get("rejected")        or 0.0
            p95w = rl_ts.get("p95_wait")        or 0.0
            mr   = rl_ts.get("mean_ride")       or 0.0
            tsim = rl_ts.get("ts_improvements") or 0.0
            lstd = rl_ts.get("load_std")        or 0.0

            # Objective score — same formula as compute_objective()
            score = mw + 0.1 * rej

            # Thesis group 1 — Setter/Finisher synergy
            self.logger.record("eval/rl_ts_mean_wait",       mw)
            self.logger.record("eval/rl_ts_ts_improvements", tsim)

            # Thesis group 2 — Pareto / service
            self.logger.record("eval/rl_ts_service_rate",    svc)
            self.logger.record("eval/rl_ts_rejected",        rej)
            self.logger.record("eval/rl_ts_gap_vs_baseline", mw - GREEDY_TS_WAIT)
            self.logger.record("eval/rl_ts_score",           score)

            # Thesis group 3 — Fleet / equity
            self.logger.record("eval/load_std",              lstd)
            self.logger.record("eval/rl_ts_p95_wait",        p95w)

            # Supporting detail
            self.logger.record("eval/rl_ts_mean_ride",       mr)

            # ---- Standalone RL (no TS) — shows setter/finisher gap ------
            sa    = _evaluate_rl_standalone(self.model, self.sim_cfg,
                                            n_episodes=self.eval_eps)
            sa_sr = sa.get("service_rate") or 0.0
            sa_mw = sa.get("mean_wait")    or 0.0
            sa_rj = sa.get("rejected")     or 0.0

            # Thesis group 1 (companion to rl_ts_mean_wait)
            self.logger.record("eval/standalone_mean_wait",    sa_mw)
            self.logger.record("eval/standalone_service_rate", sa_sr)
            self.logger.record("eval/standalone_rejected",     sa_rj)

            # Derived: synergy gap — how much TS contributed this eval step
            self.logger.record("eval/synergy_gap_wait", sa_mw - mw)

            self.logger.dump(self.num_timesteps)

            print(f"\n  [Eval @ {self.num_timesteps:,}]"
                  f"  rl+ts: svc={svc:.1%}"
                  f"  wait={mw:.2f}"
                  f"  p95={p95w:.1f}"
                  f"  rej={rej:.0f}"
                  f"  score={score:.3f}"
                  f"  ts_impr={tsim:.0f}"
                  f"  load_std={lstd:.2f}"
                  f"  |  standalone: svc={sa_sr:.1%}"
                  f"  wait={sa_mw:.2f}"
                  f"  synergy_gap={sa_mw - mw:+.2f}")

            # ---- C) Best-score checkpoint --------------------------------
            if score < self.best_score:
                self.best_score = score
                self._save_checkpoint("best")
                print(f"  [Checkpoint] NEW BEST score={score:.3f}"
                      f"  (wait={mw:.2f}  rej={rej:.0f})"
                      f"  → checkpoints/best/")

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
        policy_kwargs = dict(net_arch=cfg.get("net_arch", [128, 128])),
        verbose       = 1,
        device        = device,
        tensorboard_log = os.path.join(run_dir, "tb_logs"),
    )

    callback = EvalAndCheckpointCallback(
        sim_cfg     = sim_cfg,
        eval_freq   = eval_freq_steps,
        eval_eps    = args.eval_eps,
        ckpt_dir    = ckpt_dir,
        ckpt_freq   = ckpt_freq_steps,
        vec_env_ref = vec_env,
    )

    print(f"\nTraining for {timesteps:,} steps  ({n_envs} envs)...")
    print(f"  Eval every {args.eval_freq:,} steps  "
          f"({args.eval_eps} eps, RL+TS + standalone)")
    print(f"  Periodic checkpoint every {args.checkpoint_freq:,} steps  "
          f"(keeps last 3)")
    print(f"  Best-score checkpoint → {ckpt_dir}/best/")
    print()
    t0 = time.time()
    model.learn(
        total_timesteps = timesteps,
        callback        = callback,
        progress_bar    = True,
    )
    train_time = time.time() - t0

    # ------------------------------------------------------------------ #
    # Final evaluation — load best checkpoint, not end-of-training model  #
    # This protects against the run ending after a collapse.              #
    # ------------------------------------------------------------------ #
    best_ckpt_model  = os.path.join(ckpt_dir, "best", "model.zip")
    best_ckpt_vecnrm = os.path.join(ckpt_dir, "best", "vec_normalize.pkl")

    if os.path.exists(best_ckpt_model):
        print(f"\nLoading best checkpoint for final evaluation...")
        eval_model = MaskablePPO.load(best_ckpt_model, device=device)
        print(f"  (best score during training: {callback.best_score:.3f})")
    else:
        print(f"\nNo best checkpoint found — using end-of-training model.")
        eval_model = model

    print("Final evaluation (10 episodes, RL+TS)...")
    rl_ts_final = _evaluate_rl_plus_ts(eval_model, sim_cfg, n_episodes=10)

    print("Final evaluation (10 episodes, standalone)...")
    sa_final = _evaluate_rl_standalone(eval_model, sim_cfg, n_episodes=10)

    svc  = rl_ts_final.get("service_rate")    or 0.0
    mw   = rl_ts_final.get("mean_wait")       or 0.0
    rej  = rl_ts_final.get("rejected")        or 0.0
    p95w = rl_ts_final.get("p95_wait")        or 0.0
    mr   = rl_ts_final.get("mean_ride")       or 0.0
    tsim = rl_ts_final.get("ts_improvements") or 0.0
    lstd = rl_ts_final.get("load_std")        or 0.0
    score_final = mw + 0.1 * rej

    # ------------------------------------------------------------------ #
    # FIX 3: baselines use v4 GREEDY_TS constants                        #
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS  (from best checkpoint)")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<24} {'This run':>10} {'Greedy+TS':>10} {'Gap':>8}")
    print(f"  {'-' * 54}")
    print(f"  {'service_rate':<24} {svc:>9.1%} {GREEDY_TS_SVC:>9.1%}"
          f" {svc - GREEDY_TS_SVC:>+8.1%}")
    print(f"  {'mean_wait (RL+TS)':<24} {mw:>10.2f} {GREEDY_TS_WAIT:>10.2f}"
          f" {mw - GREEDY_TS_WAIT:>+8.2f}")
    print(f"  {'p95_wait':<24} {p95w:>10.2f}")
    print(f"  {'mean_ride':<24} {mr:>10.2f}")
    print(f"  {'rejected':<24} {rej:>10.0f}")
    print(f"  {'ts_improvements':<24} {tsim:>10.0f}")
    print(f"  {'load_std (fleet)':<24} {lstd:>10.2f}")
    print(f"  {'objective_score':<24} {score_final:>10.3f}")
    print(f"\n  Standalone RL (no TS post-process):")
    print(f"    service_rate: {sa_final.get('service_rate', 0):.1%}")
    print(f"    mean_wait:    {sa_final.get('mean_wait', 0):.2f}")
    print(f"    rejected:     {sa_final.get('rejected', 0):.0f}")
    synergy = (sa_final.get('mean_wait') or 0) - mw
    print(f"    synergy_gap:  {synergy:+.2f} min  (TS contribution)")
    print(f"\n  Training time: {train_time:.0f}s ({train_time / 60:.1f} min)")

    # Save end-of-training model too (separate from best checkpoint)
    model.save(os.path.join(run_dir, "model_final"))
    vec_env.save(os.path.join(run_dir, "vec_normalize_final.pkl"))

    summary = {
        "source": "rl_train_from_tune.py (v4)",
        "tune_config": cfg,
        "final_rl_ts": {
            "service_rate":     round(svc,  4),
            "mean_wait":        round(mw,   2),
            "p95_wait":         round(p95w, 2),
            "mean_ride":        round(mr,   2),
            "rejected":         round(rej,  1),
            "ts_improvements":  round(tsim, 1),
            "load_std":         round(lstd, 3),
            "objective_score":  round(score_final, 3),
            "gap_vs_greedy_ts_wait": round(mw  - GREEDY_TS_WAIT, 2),
            "gap_vs_greedy_ts_svc":  round(svc - GREEDY_TS_SVC,  4),
        },
        "final_standalone": {
            "service_rate": round(sa_final.get("service_rate") or 0, 4),
            "mean_wait":    round(sa_final.get("mean_wait")    or 0, 2),
            "rejected":     round(sa_final.get("rejected")     or 0, 1),
            "synergy_gap":  round(synergy, 2),
        },
        "baselines": {
            "greedy_ts_wait": GREEDY_TS_WAIT,
            "greedy_ts_svc":  GREEDY_TS_SVC,
        },
        "best_score_during_training": round(callback.best_score, 3),
        "training_time_seconds":      round(train_time, 1),
    }
    summary_path = os.path.join(run_dir, "tune_retrain_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Best model: {ckpt_dir}/best/model.zip")
    print(f"  Final model: {run_dir}/model_final.zip")
    print(f"  Summary:     {summary_path}")
    print(f"  TB:          tensorboard --logdir {run_dir}/tb_logs")
    print()
    print("  TensorBoard thesis plots:")
    print("    Setter/Finisher:  eval/rl_ts_mean_wait + eval/standalone_mean_wait")
    print("                      eval/rl_ts_ts_improvements")
    print("    Pareto/Service:   eval/rl_ts_service_rate  eval/rl_ts_rejected")
    print("                      eval/rl_ts_gap_vs_baseline")
    print("    Fleet/Equity:     eval/load_std  eval/rl_ts_p95_wait")
    print("    Algorithm health: rollout/ep_rew_mean  train/entropy_loss"
          "  train/approx_kl")
    vec_env.close()


if __name__ == "__main__":
    main()