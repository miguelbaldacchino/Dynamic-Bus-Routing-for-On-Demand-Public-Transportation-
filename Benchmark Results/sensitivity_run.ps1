# run_sensitivity.ps1
# DARP Sensitivity Sweep - RQ6
#
# 7 conditions x 15 policies x 5 seeds = 525 runs total.
# Each condition is fully isolated in its own output folder.
#
# Policy set (15):
#   Greedy : greedy, greedy+sa, greedy+ts, greedy+ga, greedy+alns
#   RL1.1  : rl(v3),   rl+ts(v3)
#   RL1.0  : rl(base), rl+ts(base)
#   RL2.x  : rl(v4),   rl+ts(v4), rl+alns(v4),
#             rl+ts(v5), rl+ts(v6), rl+ga(v6)
#
# Conditions (stress only - no easy direction):
#   baseline        default config                          reference
#   fleet_4         fleet-size 4    (default: 6)           fewer vehicles
#   demand_busy     inter-arrival 1.5, no cap (default: 3) high demand
#   maxwait_15      max-wait 15     (default: 30)          tight window
#   capacity_8      capacity 8      (default: 16)          tight vehicles
#   ridefactor_20   ride-factor 2.0 (default: 2.5)        tight ride bound
#   combined_stress fleet=4 + demand_busy + maxwait=15     worst case
#
# Note: ridefactor uses 2.0 not 1.5 - direct travel time is free-flow
# (uncongested OSRM), so 1.5x would be trivially infeasible during peak
# hours where congestion already makes actual travel 2.2x free-flow.
#
# Output structure:
#   sensitivity_results\
#     baseline\        runs\  aggregated.csv  aggregated.json  report.txt
#     fleet_4\         runs\  ...
#     ...
#
# Resume safety:
#   Condition already has aggregated.csv  ->  skipped entirely.
#   Partial runs\ JSONs                   ->  benchmark.py skips completed pairs.
#
# Usage (from odpt\ directory, venv active):
#   powershell -ExecutionPolicy Bypass -File .\sensitivity_run.ps1
#
# Background overnight:
#   Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File .\sensitivity_run.ps1" -RedirectStandardOutput "sensitivity_log.txt" -RedirectStandardError "sensitivity_err.txt" -NoNewWindow

$SEEDS    = "--seeds 100 101 102 103 104"
$POLICIES = "--policies greedy:none greedy+sa:none greedy+ts:none greedy+ga:none greedy+alns:none rl:rl_base rl+ts:rl_base rl:rl_v3 rl+ts:rl_v3 rl:rl_v4 rl+ts:rl_v4 rl+alns:rl_v4 rl+ts:rl_v5 rl+ts:rl_v6 rl+ga:rl_v6"
$OUTROOT  = "sensitivity_results"

$CONDITIONS = @(
    @{
        name = "baseline"
        args = ""
        desc = "Default config - reference point"
    },
    @{
        name = "fleet_4"
        args = "--fleet-size 4"
        desc = "Fleet stress - 4 vehicles (default: 6)"
    },
    @{
        name = "demand_busy"
        args = "--inter-arrival 1.5 --n-requests 9999"
        desc = "Demand stress - inter-arrival 1.5 min (default: 3.0), cap removed"
    },
    @{
        name = "maxwait_15"
        args = "--max-wait 15"
        desc = "Constraint stress - max wait 15 min (default: 30)"
    },
    @{
        name = "capacity_8"
        args = "--capacity 8"
        desc = "Capacity stress - vehicle capacity 8 (default: 16)"
    },
    @{
        name = "ridefactor_20"
        args = "--ride-factor 2.0"
        desc = "Ride-time stress - ride factor 2.0x (default: 2.5x)"
    },
    @{
        name = "combined_stress"
        args = "--fleet-size 4 --inter-arrival 1.5 --n-requests 9999 --max-wait 15"
        desc = "Combined worst-case - fleet=4 + demand_busy + maxwait=15"
    }
)

New-Item -ItemType Directory -Force -Path $OUTROOT | Out-Null

$total    = $CONDITIONS.Count
$done     = 0
$skipped  = 0
$failed   = 0
$start_ts = Get-Date

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DARP Sensitivity Sweep - RQ6" -ForegroundColor Cyan
Write-Host "  Conditions : $total" -ForegroundColor Cyan
Write-Host "  Seeds      : 100-104 (5 per condition)" -ForegroundColor Cyan
Write-Host "  Started    : $start_ts" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$idx = 0
foreach ($cond in $CONDITIONS) {
    $idx++
    $name    = $cond.name
    $extra   = $cond.args
    $desc    = $cond.desc
    $outdir  = "$OUTROOT\$name"
    $csvpath = "$outdir\aggregated.csv"

    Write-Host ""
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "  [$idx / $total]  $name" -ForegroundColor Yellow
    Write-Host "  $desc" -ForegroundColor DarkGray

    if (Test-Path $csvpath) {
        Write-Host "  SKIP - aggregated.csv already exists." -ForegroundColor DarkGray
        $skipped++
        continue
    }

    $cmd = "python benchmark.py $SEEDS $POLICIES $extra --out `"$outdir`""
    Write-Host "  CMD : $cmd" -ForegroundColor DarkGray
    Write-Host ""

    $t0 = Get-Date
    Invoke-Expression $cmd
    $exit_code = $LASTEXITCODE
    $elapsed   = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

    if ($exit_code -ne 0) {
        Write-Host "  FAILED (exit $exit_code) after ${elapsed} min" -ForegroundColor Red
        $failed++
    } else {
        Write-Host "  DONE - ${elapsed} min" -ForegroundColor Green
        $done++
    }
}

$total_elapsed = [math]::Round(((Get-Date) - $start_ts).TotalMinutes, 1)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Sweep complete" -ForegroundColor Cyan
Write-Host "  Done    : $done"    -ForegroundColor Green
Write-Host "  Skipped : $skipped" -ForegroundColor DarkGray
Write-Host "  Failed  : $failed"  -ForegroundColor Red
Write-Host "  Total   : ${total_elapsed} min" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan