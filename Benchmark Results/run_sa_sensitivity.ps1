$SEEDS   = "--seeds 100 101 102 103 104"
$OUTROOT = "sa_sensitivity_results"

$CONDITIONS = @(
    @{ name = "baseline";       args = "";                                                               desc = "Default config" },
    @{ name = "fleet_4";        args = "--fleet-size 4";                                                 desc = "Fleet stress - 4 vehicles" },
    @{ name = "demand_busy";    args = "--inter-arrival 1.5 --n-requests 9999";                          desc = "Demand stress - inter-arrival 1.5 min" },
    @{ name = "maxwait_15";     args = "--max-wait 15";                                                  desc = "Constraint stress - max wait 15 min" },
    @{ name = "capacity_8";     args = "--capacity 8";                                                   desc = "Capacity stress - capacity 8" },
    @{ name = "ridefactor_20";  args = "--ride-factor 2.0";                                              desc = "Ride-time stress - ride factor 2.0x" },
    @{ name = "combined_stress";args = "--fleet-size 4 --inter-arrival 1.5 --n-requests 9999 --max-wait 15"; desc = "Combined worst-case" }
)

New-Item -ItemType Directory -Force -Path $OUTROOT | Out-Null

$total    = $CONDITIONS.Count
$done     = 0
$skipped  = 0
$failed   = 0
$start_ts = Get-Date

Write-Host "SA Sensitivity Sweep - greedy+sa and rl+sa:rl_v4" -ForegroundColor Cyan
Write-Host "Conditions : $total   Seeds: 100-104   Started: $start_ts" -ForegroundColor Cyan

$idx = 0
foreach ($cond in $CONDITIONS) {
    $idx++
    $name    = $cond.name
    $extra   = $cond.args
    $desc    = $cond.desc
    $outdir  = "$OUTROOT\$name"
    $csvpath = "$outdir\aggregated.csv"

    Write-Host ""
    Write-Host "[$idx/$total] $name - $desc" -ForegroundColor Yellow

    if (Test-Path $csvpath) {
        Write-Host "  SKIP - aggregated.csv already exists." -ForegroundColor DarkGray
        $skipped++
        continue
    }

    # Pass 1: greedy+sa only
    $cmd1 = "python benchmark.py $SEEDS --no-rl $extra --out `"$outdir`""
    Write-Host "  [1/2] $cmd1" -ForegroundColor DarkGray
    $t0 = Get-Date
    Invoke-Expression $cmd1
    $exit1 = $LASTEXITCODE
    $e1 = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

    if ($exit1 -ne 0) {
        Write-Host "  FAILED greedy+sa (exit $exit1) after ${e1} min" -ForegroundColor Red
        $failed++
        continue
    }
    Write-Host "  greedy+sa done in ${e1} min" -ForegroundColor Green

    # Pass 2: rl_v4 variants (includes rl+sa:rl_v4)
    $cmd2 = "python benchmark.py $SEEDS --no-greedy --rl-model rl_v4 $extra --out `"$outdir`""
    Write-Host "  [2/2] $cmd2" -ForegroundColor DarkGray
    $t0 = Get-Date
    Invoke-Expression $cmd2
    $exit2 = $LASTEXITCODE
    $e2 = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

    if ($exit2 -ne 0) {
        Write-Host "  FAILED rl+sa:rl_v4 (exit $exit2) after ${e2} min" -ForegroundColor Red
        $failed++
        continue
    }
    Write-Host "  rl_v4 done in ${e2} min" -ForegroundColor Green

    $done++
}

$total_elapsed = [math]::Round(((Get-Date) - $start_ts).TotalMinutes, 1)
Write-Host ""
Write-Host "Sweep complete - Done: $done  Skipped: $skipped  Failed: $failed  Total: ${total_elapsed} min" -ForegroundColor Cyan
