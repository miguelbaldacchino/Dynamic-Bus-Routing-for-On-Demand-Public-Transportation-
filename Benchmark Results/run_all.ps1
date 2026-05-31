# run_all.ps1 - Windows PowerShell version of the benchmark suite

# 1. Baseline — all 20 policies, 5 seeds
python odpt\benchmark.py --n-seeds 5 --out results/baseline

# 2. Fleet 4 — stress test (20 policies, 3 seeds)
python odpt\benchmark.py --n-seeds 3 --fleet-size 4 --out results/fleet_4

# 3. Fleet 8 — abundance test (20 policies, 3 seeds)
python odpt\benchmark.py --n-seeds 3 --fleet-size 8 --out results/fleet_8

# 4. Demand Quiet (10 policies, 3 seeds)
python odpt\benchmark.py --n-seeds 3 --rl-model rl_v4 --inter-arrival 5.0 --n-requests 9999 --out results/demand_quiet

# 5. Demand Busy (20 policies, 3 seeds)
python odpt\benchmark.py --n-seeds 3 --inter-arrival 2.0 --n-requests 9999 --out results/demand_busy

# 6. Max Wait 15 — Tight constraint (20 policies, 3 seeds)
python odpt\benchmark.py --n-seeds 3 --max-wait 15 --out results/maxwait_15

# 7. Max Wait 45 (10 policies, 3 seeds)
python odpt\benchmark.py --n-seeds 3 --rl-model rl_v4 --max-wait 45 --out results/maxwait_45

# 8. Capacity 8 (10 policies, 3 seeds)
python odpt\benchmark.py --n-seeds 3 --rl-model rl_v4 --capacity 8 --out results/capacity_8

# 9. Capacity 24 (10 policies, 3 seeds)
python odpt\benchmark.py --n-seeds 3 --rl-model rl_v4 --capacity 24 --out results/capacity_24

# 10. Ride Factor 1.5 (10 policies, 3 seeds)
python odpt\benchmark.py --n-seeds 3 --rl-model rl_v4 --ride-factor 1.5 --out results/ridefactor_15

# 11. Ride Factor 3.0 (10 policies, 3 seeds)
python odpt\benchmark.py --n-seeds 3 --rl-model rl_v4 --ride-factor 3.0 --out results/ridefactor_30