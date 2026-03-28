#!/usr/bin/env python3
# patch_study.py
# Recalculates all stored Optuna trial scores using the corrected
# objective — no explicit rejection penalty, mean_wait_all recomputed
# from raw components using MAX_WAIT=120 to make rejection genuinely costly.
#
# KEY FIX: mean_wait_all is recomputed from scratch using stored
# mean_wait_served, service_rate, and rejected — NOT read from the
# pre-computed mean_wait_all stored in user_attrs, which was calculated
# with the old MAX_WAIT value and cannot be changed retroactively.
#
# Objective:
#   score = mean_wait_all(MAX_WAIT=120) + 0.5*mean_ride + 0.05*p95_wait
#   No explicit rejection penalty — handled by mean_wait_all.
#   MAX_WAIT=120 makes each rejection cost 120/400=0.30 min to mean_wait_all.
#
# Usage:
#   python patch_study.py --dry-run   (preview without saving)
#   python patch_study.py             (apply and save)

from __future__ import annotations

import argparse
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import optuna


# ---------------------------------------------------------------------------
# Objective constants — must match rl_tune.py exactly
# ---------------------------------------------------------------------------
NEW_WEIGHT_RIDE = 0.5
NEW_WEIGHT_P95  = 0.05
MAX_WAIT        = 120.0   # penalty for rejected passengers in mean_wait_all
                          # 120 = 2x the 60-min service window — reflects that
                          # a rejected passenger received zero service, worse
                          # than the worst possible served experience
N_REQUESTS      = 400     # fixed thesis constant


def compute_score(attrs: dict) -> float:
    """
    Recompute trial score from raw stored components.

    IMPORTANT: reads mean_wait_served (not mean_wait_all) and recomputes
    mean_wait_all using MAX_WAIT=120. This is necessary because the stored
    mean_wait_all was computed with the old MAX_WAIT value and cannot be
    changed retroactively.
    """
    service_rate     = attrs.get("service_rate",     0.0)
    mean_wait_served = attrs.get("mean_wait_served", 0.0)
    rejected         = attrs.get("rejected",         0.0)
    mean_ride        = attrs.get("mean_ride",        0.0)
    p95_wait         = attrs.get("p95_wait",         0.0)

    if mean_wait_served == 0.0 and mean_ride == 0.0:
        return float("inf")

    # Recompute mean_wait_all from raw components using new MAX_WAIT
    n_served      = service_rate * N_REQUESTS
    mean_wait_all = (
        (n_served * mean_wait_served) + (rejected * MAX_WAIT)
    ) / max(n_served + rejected, 1)

    return (
        mean_wait_all
        + NEW_WEIGHT_RIDE * mean_ride
        + NEW_WEIGHT_P95  * p95_wait
    )


def main():
    parser = argparse.ArgumentParser(
        description="Patch Optuna study.pkl with corrected objective weights"
    )
    parser.add_argument(
        "--study",
        default="rl_outputs/tune_results/study.pkl",
        help="Path to study.pkl (default: rl_outputs/tune_results/study.pkl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview new scores without saving",
    )
    args = parser.parse_args()

    study_path = Path(args.study)

    if not study_path.exists():
        print(f"ERROR: {study_path} not found.")
        print("Make sure rl_tune.py has run at least one trial first.")
        return

    # ------------------------------------------------------------------
    # Load study
    # ------------------------------------------------------------------
    with open(study_path, "rb") as f:
        study = pickle.load(f)

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("=" * 65)
    print("Optuna Study Patch — Corrected Objective Weights")
    print("=" * 65)
    print(f"  Study:    {study_path}")
    print(f"  Trials:   {len(completed)} completed")
    print()
    print(f"  Old weights:  p95=0.3  rejection=explicit penalty")
    print(f"  New weights:  p95={NEW_WEIGHT_P95}  MAX_WAIT={MAX_WAIT:.0f}")
    print(f"  mean_wait_all recomputed from raw components (not stored value)")
    print()
    print(f"  {'#':<5} {'old_score':>10} {'new_score':>10}"
          f" {'svc':>7} {'wait_svd':>9} {'wait_all_new':>13} {'p95_w':>7}"
          f" {'rej':>5} {'verdict'}")
    print(f"  {'-'*80}")

    # ------------------------------------------------------------------
    # Recalculate scores
    # ------------------------------------------------------------------
    old_best_score = float("inf")
    old_best_trial = None
    new_best_score = float("inf")
    new_best_trial = None

    changes = []

    for trial in completed:
        attrs     = trial.user_attrs
        old_score = trial.values[0] if trial.values else float("inf")
        new_score = compute_score(attrs)

        svc      = attrs.get("service_rate",     0)
        wait_svd = attrs.get("mean_wait_served", 0)
        p95w     = attrs.get("p95_wait",         0)
        rej      = attrs.get("rejected",         0)

        # Recompute wait_all for display using new MAX_WAIT
        n_served      = svc * N_REQUESTS
        new_wait_all  = ((n_served * wait_svd) + (rej * MAX_WAIT)) / max(n_served + rej, 1)

        if old_score < new_score:
            verdict = "↑ worse"
        elif old_score > new_score:
            verdict = "↓ better"
        else:
            verdict = "─ same"

        print(f"  {trial.number+1:<5}"
              f" {old_score:>10.3f}"
              f" {new_score:>10.3f}"
              f" {svc:>6.1%}"
              f" {wait_svd:>8.2f}"
              f" {new_wait_all:>12.2f}"
              f" {p95w:>7.2f}"
              f" {rej:>5.0f}"
              f"  {verdict}")

        changes.append((trial, old_score, new_score))

        if old_score < old_best_score:
            old_best_score = old_score
            old_best_trial = trial
        if new_score < new_best_score:
            new_best_score = new_score
            new_best_trial = trial

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(f"  Old best: trial {old_best_trial.number+1}"
          f"  score={old_best_score:.3f}"
          f"  svc={old_best_trial.user_attrs.get('service_rate',0):.1%}"
          f"  rej={old_best_trial.user_attrs.get('rejected',0):.0f}")
    print(f"  New best: trial {new_best_trial.number+1}"
          f"  score={new_best_score:.3f}"
          f"  svc={new_best_trial.user_attrs.get('service_rate',0):.1%}"
          f"  rej={new_best_trial.user_attrs.get('rejected',0):.0f}")

    if args.dry_run:
        print()
        print("  DRY RUN — no changes saved.")
        print("  Run without --dry-run to apply patch.")
        return

    # ------------------------------------------------------------------
    # Backup original study before patching
    # ------------------------------------------------------------------
    backup_path = study_path.with_suffix(
        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    )
    shutil.copy2(study_path, backup_path)
    print()
    print(f"  Backup saved: {backup_path}")

    # ------------------------------------------------------------------
    # Apply patches — update stored trial values
    # ------------------------------------------------------------------
    for trial, old_score, new_score in changes:
        # Optuna stores trial values as a list internally
        # We patch _values directly — the only reliable way to
        # update a completed trial's stored score
        trial._values = [new_score]

    # Save patched study
    with open(study_path, "wb") as f:
        pickle.dump(study, f)

    print(f"  Patched:  {study_path}")
    print()
    print("=" * 65)
    print("PATCH COMPLETE")
    print("=" * 65)
    print(f"  {len(completed)} trial scores updated.")
    print(f"  New best: trial {new_best_trial.number+1}"
          f"  score={new_best_score:.3f}"
          f"  svc={new_best_trial.user_attrs.get('service_rate',0):.1%}")
    print()
    print("  Next step:")
    print("    python rl_tune.py --resume")


if __name__ == "__main__":
    main()