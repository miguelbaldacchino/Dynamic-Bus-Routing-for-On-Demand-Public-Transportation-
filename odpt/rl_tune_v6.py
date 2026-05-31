# rl_tune_v6.py
# Optuna hyperparameter search for v6 reward (wait_all + noise training).
# Canonical final tuner; writes best config to rl_tune_v6_best.json.
#
# python rl_tune_v6.py --trials 30 --jobs 4

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time
import types
from datetime import datetime

import numpy as np
import optuna
from optuna.samplers import TPESampler


MAX_WAIT         = 30.0
MAX_WAIT_PENALTY = 60.0
OBJ_W_P95        = 0.05
SVC_FLOOR        = 0.76   # min acceptable service rate — trials below this are heavily penalised

TRAIN_NOISE = 0.10
EVAL_NOISE  = 0.0

GREEDY_TS_MEAN_WAIT  = 13.9585
GREEDY_TS_REJECTED   = 74.6
GREEDY_TS_N_REQUESTS = 400
GREEDY_TS_P95_WAIT   = 30.529

_gts_n_served = GREEDY_TS_N_REQUESTS - GREEDY_TS_REJECTED
_gts_mw_all   = (
    _gts_n_served * GREEDY_TS_MEAN_WAIT + GREEDY_TS_REJECTED * MAX_WAIT_PENALTY
) / GREEDY_TS_N_REQUESTS
GREEDY_TS_SCORE = _gts_mw_all + OBJ_W_P95 * GREEDY_TS_P95_WAIT


def compute_objective(rl_ts_metrics: dict, n_requests: int) -> tuple:
    svc    = rl_ts_metrics.get("service_rate") or 0.0
    mw     = rl_ts_metrics.get("mean_wait")    or MAX_WAIT
    rej    = rl_ts_metrics.get("rejected")     or float(n_requests)
    p95w   = rl_ts_metrics.get("p95_wait")     or MAX_WAIT
    mr     = rl_ts_metrics.get("mean_ride")    or 0.0
    ts_imp = rl_ts_metrics.get("ts_improvements") or 0.0

    n_served = svc * n_requests
    mw_all   = (n_served * mw + rej * MAX_WAIT_PENALTY) / max(n_served + rej, 1)
    score    = mw_all + OBJ_W_P95 * p95w

    # Service floor: penalise trials below SVC_FLOOR heavily.
    # Prevents Optuna from rewarding rejection-gaming solutions.
    # Each 1% below floor costs +2 pts — dominates any wait improvement.
    if svc < SVC_FLOOR:
        score += (SVC_FLOOR - svc) * 200

    return score, {
        "rl_ts_service_rate":    round(svc,    4),
        "rl_ts_mean_wait":       round(mw,     2),
        "rl_ts_mean_wait_all":   round(mw_all, 2),
        "rl_ts_p95_wait":        round(p95w,   2),
        "rl_ts_mean_ride":       round(mr,     2),
        "rl_ts_rejected":        round(rej,    1),
        "rl_ts_improvements":    round(ts_imp, 1),
        "score":                 round(score,  3),
        "vs_greedy_ts_score":    round(score - GREEDY_TS_SCORE, 3),
        "vs_greedy_ts_wait":     round(mw     - GREEDY_TS_MEAN_WAIT, 2),
        "vs_greedy_ts_wait_all": round(mw_all - _gts_mw_all, 2),
    }


def _encode_v6_vehicle_features(v, request, travel_fn, sim_time, cfg, obs, base):
    from malta_travel import DEFAULT_COORDS
    lon_min, lon_max = 14.35, 14.55
    lat_min, lat_max = 35.85, 35.95

    def nc(node_id):
        if node_id in DEFAULT_COORDS:
            lo, la = DEFAULT_COORDS[node_id]
            x = (lo - lon_min) / (lon_max - lon_min) * 2 - 1
            y = (la - lat_min) / (lat_max - lat_min) * 2 - 1
            return float(np.clip(x, -1, 1)), float(np.clip(y, -1, 1))
        return 0.0, 0.0

    x, y = nc(v.location)
    obs[base + 0] = x
    obs[base + 1] = y
    obs[base + 2] = len(v.onboard) / max(v.capacity, 1)
    obs[base + 3] = min(len(v.plan) / 20.0, 1.0)
    obs[base + 4] = 1.0 if not v.plan and v.in_transit_stop is None else 0.0
    if v.plan:
        obs[base + 5] = min(travel_fn(v.location, v.plan[0].node, sim_time) / 30.0, 1.0)
    if request is not None:
        obs[base + 6] = min(travel_fn(v.location, request.pickup_node, sim_time) / 30.0, 1.0)
        obs[base + 7] = min(travel_fn(v.location, request.dropoff_node, sim_time) / 30.0, 1.0)

    if not v.plan:
        obs[base + 8] = 0.0
        obs[base + 9] = 1.0
        obs[base + 10] = 1.0
        obs[base + 11] = 0.0
        return

    cur_node = v.location
    cur_time = sim_time
    pu_slacks, wait_qualities, urgencies = [], [], []

    for stop in v.plan:
        cur_time += travel_fn(cur_node, stop.node, cur_time)
        if stop.kind == "PU":
            if stop.earliest and cur_time < stop.earliest:
                cur_time = stop.earliest
            latest = getattr(stop, "latest", None)
            if latest is not None:
                pu_slacks.append(max(0.0, latest - cur_time) / max(cfg.max_wait, 1.0))
            if stop.request_time is not None:
                est_wait = max(0.0, cur_time - stop.request_time)
                wait_qualities.append(1.0 - min(est_wait / max(cfg.max_wait, 1.0), 1.0))
                elapsed = sim_time - stop.request_time
                urgencies.append(min(max(elapsed, 0.0) / max(cfg.max_wait, 1.0), 1.0))
        cur_time += stop.service
        cur_node  = stop.node

    obs[base + 8]  = min((cur_time - sim_time) / max(cfg.service_end, 1), 1.0)
    obs[base + 9]  = min(pu_slacks,        default=1.0) if pu_slacks else 1.0
    obs[base + 10] = float(np.mean(wait_qualities)) if wait_qualities else 1.0
    obs[base + 11] = max(urgencies,         default=0.0) if urgencies else 0.0


class DARPEnvV6:

    @staticmethod
    def make(cfg, w_wait, w_wait_all, w_rejection, w_imbalance):
        import odpt.rl_env as _re
        from odpt.rl_env import DARPEnv
        from gymnasium import spaces

        _re.USE_V6_FEATURES           = True
        _re.USE_ANTICIPATORY_FEATURES = False

        env = DARPEnv(
            cfg=cfg, reward_mode="composite",
            w_acceptance=1.0, w_wait=w_wait, w_ride=0.8,
            w_ride_sq=0.0, w_detour=0.0, w_cost=0.1, w_rejection=w_rejection,
        )
        env._v6_w_wait_all   = w_wait_all
        env._v6_w_imbalance  = w_imbalance
        env._v6_n_requests   = cfg.n_requests
        env._v6_wait_accum   = 0.0
        env._v6_accept_count = 0

        env.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(_re.get_obs_size_v6(),),
            dtype=np.float32,
        )

        def _v6_encode_state(self):
            import odpt.rl_env as _r
            obs = np.zeros(_r.get_obs_size_v6(), dtype=np.float32)
            vf  = _r.OBS_PER_VEHICLE_V6
            for i, vid in enumerate(self._vehicle_ids[:_r.MAX_VEHICLES]):
                _encode_v6_vehicle_features(
                    self._vehicles[vid], self._current_request,
                    self._travel_fn, self._sim_time, self.cfg, obs, i * vf,
                )
            rb = _r.MAX_VEHICLES * vf
            lon_min, lon_max = 14.35, 14.55
            lat_min, lat_max = 35.85, 35.95
            def nc(n):
                from malta_travel import DEFAULT_COORDS
                if n in DEFAULT_COORDS:
                    lo, la = DEFAULT_COORDS[n]
                    x = (lo-lon_min)/(lon_max-lon_min)*2-1
                    y = (la-lat_min)/(lat_max-lat_min)*2-1
                    return float(np.clip(x,-1,1)), float(np.clip(y,-1,1))
                return 0.0, 0.0
            if self._current_request is not None:
                req = self._current_request
                px,py = nc(req.pickup_node); dx,dy = nc(req.dropoff_node)
                obs[rb+0]=px; obs[rb+1]=py; obs[rb+2]=dx; obs[rb+3]=dy
                obs[rb+4]=min((req.direct_time or 0)/30.0,1.0)
            g = rb + _r.OBS_REQUEST
            obs[g+0] = self._sim_time / max(self.cfg.service_end,1)
            obs[g+1] = sum(1 for v in self._vehicles.values() if v.plan or v.in_transit_stop is not None) / max(len(self._vehicles),1)
            obs[g+2] = sum(1 for r in self._requests.values() if r.status=="COMPLETED") / max(self.cfg.n_requests,1)
            from malta_travel import congestion_factor
            obs[g+3] = (congestion_factor(self._sim_time)-0.5)*2
            return obs

        def _v6_compute_reward(self, req, old_cost, new_cost, est_wait, est_ride):
            direct   = max(req.direct_time or 1.0, 1.0)
            max_ride = direct * self.cfg.ride_factor
            norm     = direct * 5.0
            wait_pen = -self.w_wait    * (est_wait / max(self.cfg.max_wait,1.0))
            ride_pen = -self.w_ride    * (est_ride / max(max_ride,1.0))
            cost_pen = -self.w_cost    * ((new_cost-old_cost) / max(norm,1.0))
            loads = [len(v.onboard)+len(v.plan)//2 for v in self._vehicles.values()]
            bal   = self._v6_w_imbalance * (1.0 - min(float(np.std(loads)) / max(self.cfg.vehicle_capacity,1), 1.0)) if loads else 0.0
            self._v6_wait_accum   += est_wait
            self._v6_accept_count += 1
            n_rej = self._n_rejections
            rwa   = (self._v6_wait_accum + n_rej*MAX_WAIT_PENALTY) / max(self._v6_accept_count+n_rej,1)
            wap   = -self._v6_w_wait_all * (rwa / max(self.cfg.max_wait,1.0))
            return self.w_acceptance + wait_pen + ride_pen + cost_pen + bal + wap

        def _v6_rejection_penalty(self, req):
            return -self.w_rejection * (1.0 + self._n_rejections / max(self._v6_n_requests,1))

        env._encode_state      = types.MethodType(_v6_encode_state,      env)
        env._compute_reward    = types.MethodType(_v6_compute_reward,     env)
        env._rejection_penalty = types.MethodType(_v6_rejection_penalty,  env)
        return env


def _eval_standalone(model, cfg, n_ep, w_wait, w_wait_all, w_rejection, w_imbalance):
    import odpt.rl_env as _re
    from config import SimulationConfig
    _re.USE_V6_FEATURES=True; _re.USE_ANTICIPATORY_FEATURES=False
    results = []
    for i in range(n_ep):
        ec = SimulationConfig(seed=4000+i, fleet_size=cfg.fleet_size,
            vehicle_capacity=cfg.vehicle_capacity, depot_node=cfg.depot_node,
            n_requests=cfg.n_requests, demand_profile=cfg.demand_profile,
            stochastic_arrivals=cfg.stochastic_arrivals,
            travel_noise=EVAL_NOISE, n_nodes=cfg.n_nodes)
        env = DARPEnvV6.make(ec, w_wait, w_wait_all, w_rejection, w_imbalance)
        obs,_ = env.reset(seed=ec.seed); done=False
        while not done:
            mask=env.action_masks(); a,_=model.predict(obs,deterministic=True,action_masks=mask)
            obs,_,t,tr,_=env.step(int(a)); done=t or tr
        env._advance_vehicles_to(cfg.service_end+500)
        results.append(env.episode_summary())
    def m(k): v=[r[k] for r in results if r.get(k) is not None]; return float(np.mean(v)) if v else None
    return {k:m(k) for k in ["service_rate","mean_wait","p95_wait","mean_ride","rejected"]}


def _eval_rl_ts(model, cfg, n_ep, w_wait, w_wait_all, w_rejection, w_imbalance):
    import odpt.rl_env as _re
    from ts import TSPolicy
    from feasibility import check_feasibility
    from config import SimulationConfig
    from copy import deepcopy
    import random as _r
    _re.USE_V6_FEATURES=True; _re.USE_ANTICIPATORY_FEATURES=False
    ts = TSPolicy(tabu_tenure=7,max_neighbours=50,iterations=200,patience=30,
                  decision_time_limit=0.3,rng=_r.Random(999))
    results=[]
    for i in range(n_ep):
        ec = SimulationConfig(seed=4000+i, fleet_size=cfg.fleet_size,
            vehicle_capacity=cfg.vehicle_capacity, depot_node=cfg.depot_node,
            n_requests=cfg.n_requests, demand_profile=cfg.demand_profile,
            stochastic_arrivals=cfg.stochastic_arrivals,
            travel_noise=EVAL_NOISE, n_nodes=cfg.n_nodes)
        env=DARPEnvV6.make(ec,w_wait,w_wait_all,w_rejection,w_imbalance)
        obs,_=env.reset(seed=ec.seed); done=False; ts_imp=0
        while not done:
            mask=env.action_masks(); a,_=model.predict(obs,deterministic=True,action_masks=mask)
            obs,_,t,tr,_=env.step(int(a)); done=t or tr
            if not (t or tr):
                ss={**env._system_state,"vehicles":{}}
                for vid,v in env._vehicles.items():
                    vs=v.to_state_dict(env._sim_time); nc=1 if v.in_transit_stop is not None else 0
                    ss["vehicles"][vid]={**vs,"plan":deepcopy(vs["plan_snapshot"]),"n_committed":nc}
                chg=ts.propose(ss,check_feasibility,ec.weights)
                if chg:
                    ts_imp+=1
                    for vid,np_ in chg.items():
                        vv=env._vehicles[vid]; nc=1 if vv.in_transit_stop is not None else 0
                        vv.plan=np_[nc:]
        env._advance_vehicles_to(cfg.service_end+500)
        s=env.episode_summary(); s["ts_improvements"]=ts_imp; results.append(s)
    def m(k): v=[r[k] for r in results if r.get(k) is not None]; return float(np.mean(v)) if v else None
    return {k:m(k) for k in ["service_rate","mean_wait","p95_wait","mean_ride","rejected","ts_improvements"]}


def run_trial(trial, timesteps, n_envs, tb_base):
    import torch
    import odpt.rl_env as _re
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    from config import SimulationConfig

    w_wait      = trial.suggest_float("w_wait",      1.0,  4.0)
    w_wait_all  = trial.suggest_float("w_wait_all",  0.5,  2.0)
    w_rejection = trial.suggest_float("w_rejection", 4.0,  9.0)
    w_imbalance = trial.suggest_float("w_imbalance", 0.1,  1.0)
    lr_start    = trial.suggest_float("lr_start",    1e-4, 5e-4, log=True)
    gamma       = trial.suggest_float("gamma",       0.988, 0.997)
    ent_coef    = trial.suggest_float("ent_coef",    0.005, 0.05, log=True)
    n_steps     = trial.suggest_categorical("n_steps", [1024, 2048])

    lr_fn = lambda progress: lr_start * progress

    print(f"\n  --- Trial {trial.number+1} ---")
    print(f"    w_wait={w_wait:.3f}  w_wait_all={w_wait_all:.3f}  w_rej={w_rejection:.3f}  w_imbal={w_imbalance:.3f}")
    print(f"    lr={lr_start:.5f}  gamma={gamma:.4f}  ent={ent_coef:.5f}  n_steps={n_steps}")

    _re.USE_V6_FEATURES=True; _re.USE_ANTICIPATORY_FEATURES=False

    cfg = SimulationConfig(seed=42, fleet_size=6, vehicle_capacity=16,
        depot_node=0, n_requests=400, demand_profile="malta",
        stochastic_arrivals=True, travel_noise=TRAIN_NOISE, n_nodes=71)

    def mk(seed):
        def _f():
            ec=SimulationConfig(seed=seed, fleet_size=cfg.fleet_size,
                vehicle_capacity=cfg.vehicle_capacity, depot_node=cfg.depot_node,
                n_requests=cfg.n_requests, demand_profile=cfg.demand_profile,
                stochastic_arrivals=cfg.stochastic_arrivals,
                travel_noise=TRAIN_NOISE, n_nodes=cfg.n_nodes)
            return DARPEnvV6.make(ec,w_wait,w_wait_all,w_rejection,w_imbalance)
        return _f

    vec_env = SubprocVecEnv([mk(42+i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=gamma)

    bs=128; buf=n_steps*n_envs
    if buf%bs!=0:
        cands=[b for b in [64,128,256] if buf%b==0 and b<=bs]
        bs=max(cands) if cands else 64

    device="cuda" if torch.cuda.is_available() else "cpu"

    model=MaskablePPO("MlpPolicy",vec_env,
        learning_rate=lr_fn, gamma=gamma, ent_coef=ent_coef,
        n_steps=n_steps, batch_size=bs, n_epochs=5,
        vf_coef=1.0, gae_lambda=0.95, clip_range=0.2, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256,128]),
        verbose=0, tensorboard_log=os.path.join(tb_base,f"trial_{trial.number+1:03d}"),
        device=device)

    t0=time.time()
    class TO(BaseCallback):
        def __init__(self): super().__init__(verbose=0)
        def _on_step(self): return time.time()-t0 < 60*60
    model.learn(timesteps, callback=TO())
    train_time=time.time()-t0

    print(f"    Evaluating standalone (5 eps)...")
    sa=_eval_standalone(model,cfg,5,w_wait,w_wait_all,w_rejection,w_imbalance)
    print(f"    svc={sa.get('service_rate',0):.1%}  wait={sa.get('mean_wait',0):.2f}  rej={sa.get('rejected',0):.0f}")

    print(f"    Evaluating rl+ts (15 eps)...")
    rl_ts=_eval_rl_ts(model,cfg,15,w_wait,w_wait_all,w_rejection,w_imbalance)
    print(f"    svc={rl_ts.get('service_rate',0):.1%}  wait={rl_ts.get('mean_wait',0):.2f}  rej={rl_ts.get('rejected',0):.0f}  ts_impr={rl_ts.get('ts_improvements',0):.1f}")

    vec_env.close()
    score,metrics=compute_objective(rl_ts,cfg.n_requests)
    for k,v in metrics.items():
        if v is not None: trial.set_user_attr(k,v)
    for k,v in sa.items():
        if v is not None: trial.set_user_attr(f"sa_{k}",round(v,4) if isinstance(v,float) else v)
    trial.set_user_attr("train_time_s",round(train_time,1))

    delta=metrics.get("vs_greedy_ts_score",0)
    status="BEATS BASELINE" if score<GREEDY_TS_SCORE else "below baseline"
    print(f"    Score={score:.3f}  wait_all={metrics.get('rl_ts_mean_wait_all',0):.2f}  delta={delta:+.3f}  [{status}]")
    return score


def main():
    parser=argparse.ArgumentParser(description="rl_tune_v6")
    parser.add_argument("--samples",    type=int,  default=30)
    parser.add_argument("--timesteps",  type=int,  default=400_000)
    parser.add_argument("--n-envs",     type=int,  default=6)
    parser.add_argument("--output-dir", default="rl_outputs/tune_v6")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--study-name", default="darp_ppo_v6")
    args=parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tb_base=os.path.join(args.output_dir,"tb")
    study_path=os.path.join(args.output_dir,"study.pkl")
    os.makedirs(tb_base, exist_ok=True)

    if args.resume and os.path.exists(study_path):
        with open(study_path,"rb") as f: study=pickle.load(f)
        completed=len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE])
        print(f"Resuming: {completed} trials complete")
    else:
        study=optuna.create_study(
            study_name=args.study_name, direction="minimize",
            sampler=TPESampler(seed=42, n_startup_trials=8))

        # Seed trial 1: v5 actual best (trial 12/30, score=10.47)
        # w_wait_all=1.0 neutral — no prior for this new dimension
        study.enqueue_trial({
            "w_wait":2.876, "w_wait_all":1.0, "w_rejection":7.223,
            "w_imbalance":0.675, "lr_start":3.89e-4,
            "gamma":0.9919, "ent_coef":0.0170, "n_steps":1024,
        })
        # Seed trial 2: moderate hedge, n_steps=2048
        study.enqueue_trial({
            "w_wait":2.2, "w_wait_all":0.7, "w_rejection":6.0,
            "w_imbalance":0.45, "lr_start":2.5e-4,
            "gamma":0.993, "ent_coef":0.020, "n_steps":2048,
        })
        # Seed trial 3: high w_wait_all — aggressively test new reward axis
        study.enqueue_trial({
            "w_wait":2.4, "w_wait_all":1.8, "w_rejection":7.5,
            "w_imbalance":0.5, "lr_start":2.0e-4,
            "gamma":0.993, "ent_coef":0.018, "n_steps":1024,
        })
        print(f"New study: {args.study_name}  (3 seed trials, v5 best as trial 1)")

    def cb(study, trial):
        a=trial.user_attrs; d=a.get("vs_greedy_ts_score",0)
        print(f"\n  Trial {trial.number+1}: score={trial.value:.3f}  delta={d:+.3f}")
        print(f"    svc={a.get('rl_ts_service_rate',0):.1%}  wait={a.get('rl_ts_mean_wait',0):.2f}"
              f"  wait_all={a.get('rl_ts_mean_wait_all',0):.2f}  p95={a.get('rl_ts_p95_wait',0):.2f}"
              f"  rej={a.get('rl_ts_rejected',0):.0f}  ts_impr={a.get('rl_ts_improvements',0):.1f}")
        print(f"    standalone: svc={a.get('sa_service_rate',0):.1%}  wait={a.get('sa_mean_wait',0):.2f}")
        print(f"    train_time: {a.get('train_time_s',0):.0f}s")
        valid=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE and t.value!=float("inf")]
        if valid:
            b=min(valid,key=lambda t:t.value)
            print(f"  >>> Best: trial {b.number+1}  score={b.value:.3f}  delta={b.user_attrs.get('vs_greedy_ts_score',0):+.3f}"
                  f"  wait_all={b.user_attrs.get('rl_ts_mean_wait_all',0):.2f}  svc={b.user_attrs.get('rl_ts_service_rate',0):.1%}")
        with open(study_path,"wb") as f: pickle.dump(study,f)

    completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    n_rem=args.samples-len(completed)
    if n_rem<=0:
        print(f"All {args.samples} trials complete."); return

    print("="*65)
    print("rl_tune_v6 — richer obs + noise training")
    print("="*65)
    print(f"  Changes from v5:")
    print(f"    travel_noise  {TRAIN_NOISE} train / {EVAL_NOISE} eval  (was 0.0/0.0)")
    print(f"    obs_per_veh   12  (was 8 — adds makespan, slack, quality, urgency)")
    print(f"    w_wait_all    searched 0.5-2.0  (new running mean_wait_all penalty)")
    print(f"    net_arch      [256,128]  (was [128,128])")
    print(f"    n_steps       [1024,2048]  (was fixed 1024)")
    print(f"    eval seeds    4000+i")
    print(f"    MAX_WAIT_PENALTY {MAX_WAIT_PENALTY:.0f}  (unchanged)")
    print(f"  Baseline (seeds 100-104, greedy+ts):")
    print(f"    wait={GREEDY_TS_MEAN_WAIT:.4f}  rej={GREEDY_TS_REJECTED}  score={GREEDY_TS_SCORE:.3f}")
    print(f"  Trials: {args.samples} ({n_rem} remaining)  Timesteps: {args.timesteps:,}  Envs: {args.n_envs}")
    print("="*65)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    t_start=time.time()
    study.optimize(lambda t:run_trial(t,args.timesteps,args.n_envs,tb_base),
                   n_trials=n_rem, callbacks=[cb], show_progress_bar=False)
    total_time=time.time()-t_start

    valid=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE and t.value!=float("inf")]
    if not valid:
        print("No valid trials."); return

    best=min(valid,key=lambda t:t.value)
    bp=best.params; ba=best.user_attrs

    print("\n"+"="*65)
    print("SEARCH v6 COMPLETE")
    print("="*65)
    print(f"  Time: {total_time/3600:.1f}h  |  Trials: {len(valid)}/{len(study.trials)}")
    print(f"  Best trial #{best.number+1}  score={best.value:.3f}  delta={ba.get('vs_greedy_ts_score',0):+.3f}")
    print(f"    svc={ba.get('rl_ts_service_rate',0):.1%}  wait={ba.get('rl_ts_mean_wait',0):.2f}"
          f"  wait_all={ba.get('rl_ts_mean_wait_all',0):.2f}  rej={ba.get('rl_ts_rejected',0):.0f}")
    print(f"    w_wait={bp['w_wait']:.4f}  w_wait_all={bp['w_wait_all']:.4f}"
          f"  w_rej={bp['w_rejection']:.4f}  w_imbal={bp['w_imbalance']:.4f}")
    print(f"    lr={bp['lr_start']:.5f}  gamma={bp['gamma']:.4f}"
          f"  ent={bp['ent_coef']:.5f}  n_steps={bp['n_steps']}")

    cfg_out={
        "source":"rl_tune_v6.py",
        "objective":"mean_wait_all_penalised(MAX_WAIT_PENALTY=60)+0.05*p95_wait",
        "reward_mode":"composite",
        "w_acceptance":1.0, "w_wait":bp["w_wait"], "w_wait_all":bp["w_wait_all"],
        "w_ride":0.8, "w_ride_sq":0.0, "w_detour":0.0, "w_cost":0.1,
        "w_rejection":bp["w_rejection"], "w_imbalance":bp["w_imbalance"],
        "lr_start":bp["lr_start"], "lr_schedule":"linear",
        "gamma":bp["gamma"], "ent_coef":bp["ent_coef"],
        "n_steps":bp["n_steps"], "n_epochs":5, "batch_size":128,
        "vf_coef":1.0, "gae_lambda":0.95, "clip_range":0.2, "max_grad_norm":0.5,
        "net_arch":[256,128], "norm_obs":False, "norm_reward":True,
        "train_noise":TRAIN_NOISE, "eval_noise":EVAL_NOISE,
        "use_v6_features":True, "timesteps":args.timesteps, "n_envs":args.n_envs,
        "best_trial_number":best.number+1, "n_trials":len(study.trials),
        "n_valid_trials":len(valid), "best_score":round(best.value,4),
        "achieved_rl_ts_service_rate":ba.get("rl_ts_service_rate"),
        "achieved_rl_ts_mean_wait":ba.get("rl_ts_mean_wait"),
        "achieved_rl_ts_mean_wait_all":ba.get("rl_ts_mean_wait_all"),
        "achieved_rl_ts_p95_wait":ba.get("rl_ts_p95_wait"),
        "achieved_rl_ts_rejected":ba.get("rl_ts_rejected"),
        "achieved_standalone_svc":ba.get("sa_service_rate"),
        "achieved_standalone_wait":ba.get("sa_mean_wait"),
        "greedy_ts_mean_wait":GREEDY_TS_MEAN_WAIT,
        "greedy_ts_score":round(GREEDY_TS_SCORE,3),
        "max_wait_penalty":MAX_WAIT_PENALTY,
    }
    cp=os.path.join(args.output_dir,"best_config.json")
    with open(cp,"w",encoding="utf-8") as f: json.dump(cfg_out,f,indent=2)

    fn=["trial","score","vs_greedy_ts_score",
        "rl_ts_service_rate","rl_ts_mean_wait","rl_ts_mean_wait_all",
        "rl_ts_p95_wait","rl_ts_mean_ride","rl_ts_rejected","rl_ts_improvements",
        "sa_service_rate","sa_mean_wait",
        "w_wait","w_wait_all","w_rejection","w_imbalance",
        "lr_start","gamma","ent_coef","n_steps","train_time_s"]
    csvp=os.path.join(args.output_dir,"all_trials.csv")
    with open(csvp,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fn); w.writeheader()
        for t in sorted([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE],
                        key=lambda x:x.value if x.value!=float("inf") else 999):
            row={"trial":t.number+1,"score":round(t.value,4)}
            row.update(t.user_attrs); row.update(t.params)
            w.writerow({k:row.get(k,"") for k in fn})

    top5=sorted(valid,key=lambda t:t.value)[:5]
    print(f"\n  TOP 5:")
    for t in top5:
        a=t.user_attrs
        print(f"  #{t.number+1}  score={t.value:.3f}  delta={a.get('vs_greedy_ts_score',0):+.3f}"
              f"  wait_all={a.get('rl_ts_mean_wait_all',0):.2f}  wait={a.get('rl_ts_mean_wait',0):.2f}"
              f"  svc={a.get('rl_ts_service_rate',0):.1%}  rej={a.get('rl_ts_rejected',0):.0f}"
              f"  ts_impr={a.get('rl_ts_improvements',0):.1f}")

    print(f"\n  Config: {cp}")
    print(f"  CSV:    {csvp}")
    print(f"\n  Next: python rl_train_from_tune_v6.py --config {cp}")


if __name__ == "__main__":
    main()