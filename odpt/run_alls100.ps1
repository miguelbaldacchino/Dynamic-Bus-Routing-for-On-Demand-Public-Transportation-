# run_missing.ps1
# Rename your existing results folders before running this.
# e.g. results\fleet_4 -> results\fleet_4_old


# --- fleet_8: all RL models ---
python odpt\benchmark.py --n-seeds 3 --fleet-size 8 --rl-model rl_v3ant rl_base rl_v3 rl_v4 rl_v5 --no-greedy --out results/fleet_8

# --- all other scenarios: rl_base, rl_v3, rl_v4, rl_v5 only ---
python odpt\benchmark.py --n-seeds 5 --rl-model rl_base rl_v3 rl_v4 rl_v5 --no-greedy --out results/baseline

python odpt\benchmark.py --n-seeds 3 --capacity 8 --rl-model rl_base rl_v3 rl_v4 rl_v5 --no-greedy --out results/capacity_8

python odpt\benchmark.py --n-seeds 3 --capacity 24 --rl-model rl_base rl_v3 rl_v4 rl_v5 --no-greedy --out results/capacity_24

python odpt\benchmark.py --n-seeds 3 --inter-arrival 5.0 --n-requests 9999 --rl-model rl_base rl_v3 rl_v4 rl_v5 --no-greedy --out results/demand_quiet

python odpt\benchmark.py --n-seeds 3 --inter-arrival 2.0 --n-requests 9999 --rl-model rl_base rl_v3 rl_v4 rl_v5 --no-greedy --out results/demand_busy

python odpt\benchmark.py --n-seeds 3 --max-wait 15 --rl-model rl_base rl_v3 rl_v4 rl_v5 --no-greedy --out results/maxwait_15

python odpt\benchmark.py --n-seeds 3 --max-wait 45 --rl-model rl_base rl_v3 rl_v4 rl_v5 --no-greedy --out results/maxwait_45

python odpt\benchmark.py --n-seeds 3 --ride-factor 1.5 --rl-model rl_base rl_v3 rl_v4 rl_v5 --no-greedy --out results/ridefactor_15

python odpt\benchmark.py --n-seeds 3 --ride-factor 3.0 --rl-model rl_base rl_v3 rl_v4 rl_v5 --no-greedy --out results/ridefactor_30