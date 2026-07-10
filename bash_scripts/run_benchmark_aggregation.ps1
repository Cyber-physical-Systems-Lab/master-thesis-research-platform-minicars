# ══════════════════════════════════════════════════════════════════════════════
# aggregate_benchmarks.ps1 — Group exp-log-*.json files by Scenario / FOV / calib
#                             and run benchmark_plot.py over each group.
#                             (PowerShell / Windows)
#
# benchmark_plot.py's ACTUAL CLI (see file header) takes only positional JSON
# paths plus an optional "-f N" flag meaning "average the first N files given":
#     python benchmark_plot.py file1.json file2.json ...       (compare runs)
#     python benchmark_plot.py -f N file1.json ... fileN.json  (average N runs)
# There is NO --scenario / --fov / --calib / --policy / --out flag -- output
# folders are derived automatically from each file's embedded meta block via
# _results_dir()/_multi_run_dir() (./exp/results/{scenario}-{dfov}dFOV-{calib}-{policy}/).
#
# This script GROUPS files by the (Scenario, FOV, HeightCM, Calib, Policy)
# combination encoded in their filenames, then invokes benchmark_plot.py once
# per group with "-f <N>" (N = however many files exist for that group) so all
# reps in a group are averaged together in one run.
#
# CROSS-GROUP COMPARISON (-Compare switch):
# Each "-f N ..." call is its OWN process, so benchmark_plot.py's internal
# multi-run comparison (policy_comparison_bar.png, combined summary table)
# never sees more than one averaged group at a time. Passing -Compare instead
# runs compare_groups.py once, at the end, with EVERY file across ALL groups --
# it averages each group internally (mirroring "-f N" behaviour) and then
# feeds all per-group averaged results into benchmark_plot's own multi-run
# comparison path in a single process, producing ONE combined
# policy_comparison_bar.png + summary table across every group
# (e.g. calib vs. non-calib, or S1 vs. S2 vs. S3 vs. S4).
# compare_groups.py must sit next to benchmark_plot.py.
#
# Usage:  .\aggregate_benchmarks.ps1 [-LogDir <path>] [-Scenario S1] [-Fov 90]
#                                    [-Calib calib] [-Policy cooperative]
#                                    [-MinFiles 1] [-Compare] [-DryRun]
#
#   -LogDir     Folder containing exp-log-*.json files (default: current dir)
#   -Scenario   Optional filter: only aggregate this scenario (S1..S4)
#   -Fov        Optional filter: only aggregate this FOV tag (90 / 78 / ...)
#   -Calib      Optional filter: "calib" or "non-calib"
#   -Policy     Optional filter: "cooperative" or "non_cooperative"
#   -MinFiles   Skip groups with fewer than this many matching files (default 1)
#   -Compare    After per-group averaging, also run compare_groups.py once
#               across ALL matched files to produce a cross-group comparison
#               chart + combined summary table
#   -DryRun     Print the resolved groups and exact command lines without
#               actually invoking python
#
# Log filename convention (produced by player_launcher.py / experiments_script.ps1):
#   exp-log-{Scenario}-r{Repetition}-{FOV}fov-{HeightCM}cm-{calib|non-calib}-{Policy}[-{N}].json
#
#   e.g. exp-log-S1-r7-90fov-170cm-non-calib-cooperative.json
#        exp-log-S2-r1-90fov-170cm-non-calib-cooperative-2.json   (trailing -N = dup/retake tag)
#
# A "group" = every log file sharing the same (Scenario, FOV, HeightCM, Calib,
# Policy) combination, regardless of repetition number or trailing -N tag.
# N (the -f count passed to benchmark_plot.py) is fully variable -- however
# many repetition files exist on disk for that combination.
#
# Examples:
#   .\aggregate_benchmarks.ps1
#   .\aggregate_benchmarks.ps1 -LogDir .\logs
#   .\aggregate_benchmarks.ps1 -Scenario S1 -Fov 90 -Calib calib
#   .\aggregate_benchmarks.ps1 -Policy cooperative -MinFiles 3
#   .\aggregate_benchmarks.ps1 -Compare
#   .\aggregate_benchmarks.ps1 -DryRun
# ══════════════════════════════════════════════════════════════════════════════

param(
    [string]$LogDir   = ".",
    [string]$Scenario = $null,
    [string]$Fov      = $null,
    [string]$Calib    = $null,
    [string]$Policy   = $null,
    [int]   $MinFiles = 1,
    [switch]$Compare,
    [switch]$DryRun
)

$PYTHON        = "python"
$SCRIPT        = ".\exp\benchmark_plot.py"
$COMPARE_SCRIPT = ".\bash_scripts\compare_groups.py"

# ── Filename pattern ─────────────────────────────────────────────────────────
# Captures: Scenario, Repetition, FOV, HeightCM, Calib, Policy, (optional) DupTag
$FilePattern = '^exp-log-(?<scenario>S\d+)-r(?<rep>\d+)-(?<fov>\d+)fov-(?<cm>\d+)cm-(?<calib>non-calib|calib)-(?<policy>cooperative|non_cooperative)(?:-(?<dup>\d+))?\.json$'

if (-not (Test-Path $LogDir)) {
    Write-Host "[aggregate_benchmarks.ps1] LogDir not found: $LogDir"
    exit 1
}

$allFiles = Get-ChildItem -Path $LogDir -Filter "exp-log-*.json" -File
if ($allFiles.Count -eq 0) {
    Write-Host "[aggregate_benchmarks.ps1] No exp-log-*.json files found in $LogDir"
    exit 1
}

Write-Host "[aggregate_benchmarks.ps1] Found $($allFiles.Count) log file(s) in $LogDir"

# ── Parse + filter ───────────────────────────────────────────────────────────
$parsed = @()
foreach ($f in $allFiles) {
    $m = [regex]::Match($f.Name, $FilePattern)
    if (-not $m.Success) {
        Write-Host "[aggregate_benchmarks.ps1] Skipping (name doesn't match convention): $($f.Name)"
        continue
    }

    $g = $m.Groups
    $entry = [PSCustomObject]@{
        Path     = $f.FullName
        Name     = $f.Name
        Scenario = $g["scenario"].Value
        Rep      = [int]$g["rep"].Value
        Fov      = $g["fov"].Value
        HeightCM = $g["cm"].Value
        Calib    = $g["calib"].Value
        Policy   = $g["policy"].Value
    }

    if ($Scenario -and $entry.Scenario -ne $Scenario) { continue }
    if ($Fov      -and $entry.Fov      -ne $Fov)      { continue }
    if ($Calib    -and $entry.Calib    -ne $Calib)    { continue }
    if ($Policy   -and $entry.Policy   -ne $Policy)   { continue }

    $parsed += $entry
}

if ($parsed.Count -eq 0) {
    Write-Host "[aggregate_benchmarks.ps1] No files matched the requested filters."
    exit 1
}

# ── Group by (Scenario, Fov, HeightCM, Calib, Policy) — N is variable per
#    group, exactly as many files as exist on disk for that combination. ────
$groups = $parsed | Group-Object -Property Scenario, Fov, HeightCM, Calib, Policy

Write-Host "[aggregate_benchmarks.ps1] Built $($groups.Count) group(s)."
Write-Host ""

$keptFilesForCompare = @()
$groupIndex = 0
foreach ($grp in $groups) {
    $groupIndex++
    $sample = $grp.Group[0]
    $files  = $grp.Group | Sort-Object Rep
    $n      = $files.Count

    $tag = "$($sample.Scenario)-$($sample.Fov)fov-$($sample.HeightCM)cm-$($sample.Calib)-$($sample.Policy)"

    if ($n -lt $MinFiles) {
        Write-Host "[aggregate_benchmarks.ps1] ($groupIndex) Skipping group '$tag' — $n file(s) < MinFiles=$MinFiles"
        continue
    }

    Write-Host "[aggregate_benchmarks.ps1] ($groupIndex/$($groups.Count)) Group: $tag  -> $n file(s), averaging via -f $n"
    foreach ($f in $files) {
        Write-Host "    r$($f.Rep): $($f.Name)"
    }

    # benchmark_plot.py's real signature: "-f N file1 file2 ... fileN" averages
    # the first N files given. Passing every file in the group with -f = count
    # averages the whole group in one run; output folder is auto-derived by
    # the script itself from each file's meta (scenario/dfov/calib/policy),
    # so no --out/--scenario/--fov/--calib/--policy flags are passed here.
    $cmdArgs = @($SCRIPT, "-f", "$n") + ($files.Path)

    if ($DryRun) {
        Write-Host "    [dry-run] $PYTHON $($cmdArgs -join ' ')"
    } else {
        Write-Host "    Running: $PYTHON $($cmdArgs -join ' ')"
        & $PYTHON @cmdArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    [aggregate_benchmarks.ps1] WARNING: benchmark_plot.py exited with code $LASTEXITCODE for group $tag"
        }
    }

    $keptFilesForCompare += $files.Path
    Write-Host ""
}

if ($Compare) {
    if ($keptFilesForCompare.Count -eq 0) {
        Write-Host "[aggregate_benchmarks.ps1] -Compare requested but no groups met MinFiles; skipping."
    } else {
        Write-Host "[aggregate_benchmarks.ps1] Running cross-group comparison across $($keptFilesForCompare.Count) file(s)..."
        $compareArgs = @($COMPARE_SCRIPT, "--min-files", "$MinFiles") + $keptFilesForCompare
        if ($DryRun) {
            Write-Host "    [dry-run] $PYTHON $($compareArgs -join ' ')"
        } else {
            Write-Host "    Running: $PYTHON $($compareArgs -join ' ')"
            & $PYTHON @compareArgs
            if ($LASTEXITCODE -ne 0) {
                Write-Host "    [aggregate_benchmarks.ps1] WARNING: compare_groups.py exited with code $LASTEXITCODE"
            }
        }
    }
    Write-Host ""
}

Write-Host "[aggregate_benchmarks.ps1] Done."
