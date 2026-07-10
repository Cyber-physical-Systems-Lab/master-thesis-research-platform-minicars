#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# aggregate_benchmarks.sh — Group exp-log-*.json files by Scenario / FOV / calib
#                            and run benchmark_plot.py over each group.
#                            (Bash / Linux)
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
# CROSS-GROUP COMPARISON (-C switch):
# Each "-f N ..." call is its OWN process, so benchmark_plot.py's internal
# multi-run comparison (policy_comparison_bar.png, combined summary table)
# never sees more than one averaged group at a time. Passing -C instead
# runs compare_groups.py once, at the end, with EVERY file across ALL groups --
# it averages each group internally (mirroring "-f N" behaviour) and then
# feeds all per-group averaged results into benchmark_plot's own multi-run
# comparison path in a single process, producing ONE combined
# policy_comparison_bar.png + summary table across every group
# (e.g. calib vs. non-calib, or S1 vs. S2 vs. S3 vs. S4).
# compare_groups.py must sit in ./bash_scripts/ (or adjust COMPARE_SCRIPT).
#
# Usage:  ./aggregate_benchmarks.sh [-d <LogDir>] [-s <Scenario>] [-F <Fov>]
#                                    [-c <Calib>] [-p <Policy>] [-m <MinFiles>]
#                                    [-C] [-D]
#
#   -d <path>   Folder containing exp-log-*.json files (default: current dir)
#   -s <tag>    Optional filter: only aggregate this scenario (S1..S4)
#   -F <tag>    Optional filter: only aggregate this FOV tag (90 / 78 / ...)
#   -c <tag>    Optional filter: "calib" or "non-calib"
#   -p <tag>    Optional filter: "cooperative" or "non_cooperative"
#   -m <num>    Skip groups with fewer than this many matching files (default 1)
#   -C          After per-group averaging, also run compare_groups.py once
#               across ALL matched files to produce a cross-group comparison
#               chart + combined summary table
#   -D          Dry-run: print resolved groups and commands without executing
#
# Log filename convention:
#   exp-log-{Scenario}-r{Repetition}-{FOV}fov-{HeightCM}cm-{calib|non-calib}-{Policy}[-{N}].json
#
# Examples:
#   ./aggregate_benchmarks.sh
#   ./aggregate_benchmarks.sh -d ./logs
#   ./aggregate_benchmarks.sh -s S1 -F 90 -c calib
#   ./aggregate_benchmarks.sh -p cooperative -m 3
#   ./aggregate_benchmarks.sh -C
#   ./aggregate_benchmarks.sh -D
# ══════════════════════════════════════════════════════════════════════════════

set -u

# Defaults
LogDir="."
Scenario=""
Fov=""
Calib=""
Policy=""
MinFiles=1
Compare=0
DryRun=0

PYTHON="python"
SCRIPT="./exp/benchmark_plot.py"
COMPARE_SCRIPT="./bash_scripts/compare_groups.py"

# Parse options
while getopts "d:s:F:c:p:m:CD" opt; do
    case $opt in
        d) LogDir="$OPTARG" ;;
        s) Scenario="$OPTARG" ;;
        F) Fov="$OPTARG" ;;
        c) Calib="$OPTARG" ;;
        p) Policy="$OPTARG" ;;
        m) MinFiles="$OPTARG" ;;
        C) Compare=1 ;;
        D) DryRun=1 ;;
        *) echo "Usage: $0 [-d LogDir] [-s Scenario] [-F Fov] [-c Calib] [-p Policy] [-m MinFiles] [-C] [-D]" >&2; exit 1 ;;
    esac
done

if [ ! -d "$LogDir" ]; then
    echo "[aggregate_benchmarks.sh] LogDir not found: $LogDir"
    exit 1
fi

# Find matching JSON files
shopt -s nullglob
allFiles=("$LogDir"/exp-log-*.json)
if [ ${#allFiles[@]} -eq 0 ]; then
    echo "[aggregate_benchmarks.sh] No exp-log-*.json files found in $LogDir"
    exit 1
fi
echo "[aggregate_benchmarks.sh] Found ${#allFiles[@]} log file(s) in $LogDir"

# Regex pattern to capture fields
pattern='^exp-log-(S[0-9]+)-r([0-9]+)-([0-9]+)fov-([0-9]+)cm-(non-calib|calib)-(cooperative|non_cooperative)(-([0-9]+))?\.json$'

declare -A groups_files
declare -A groups_count

# Filtered files for -Compare (those that pass all filters and MinFiles later)
all_matched_files=()

for f in "${allFiles[@]}"; do
    filename=$(basename "$f")
    if [[ ! $filename =~ $pattern ]]; then
        echo "[aggregate_benchmarks.sh] Skipping (name doesn't match convention): $filename"
        continue
    fi

    scenario="${BASH_REMATCH[1]}"
    rep="${BASH_REMATCH[2]}"
    fov="${BASH_REMATCH[3]}"
    height="${BASH_REMATCH[4]}"
    calib="${BASH_REMATCH[5]}"
    policy="${BASH_REMATCH[6]}"
    # duplicate tag is BASH_REMATCH[8] (optional) – we ignore it for grouping

    # Apply filters
    if [ -n "$Scenario" ] && [ "$scenario" != "$Scenario" ]; then continue; fi
    if [ -n "$Fov" ] && [ "$fov" != "$Fov" ]; then continue; fi
    if [ -n "$Calib" ] && [ "$calib" != "$Calib" ]; then continue; fi
    if [ -n "$Policy" ] && [ "$policy" != "$Policy" ]; then continue; fi

    # Group key: scenario-fov-height-calib-policy
    key="${scenario}-${fov}-${height}-${calib}-${policy}"

    # Append file to group
    if [ -z "${groups_files[$key]:-}" ]; then
        groups_files[$key]="$f"
        groups_count[$key]=1
    else
        groups_files[$key]="${groups_files[$key]} $f"
        groups_count[$key]=$((groups_count[$key] + 1))
    fi

    # Store for -Compare (will be filtered later by MinFiles)
    all_matched_files+=("$f")
done

if [ ${#groups_files[@]} -eq 0 ]; then
    echo "[aggregate_benchmarks.sh] No files matched the requested filters."
    exit 1
fi

echo "[aggregate_benchmarks.sh] Built ${#groups_files[@]} group(s)."
echo ""

kept_files_for_compare=()
group_index=0
total_groups=${#groups_files[@]}

for key in "${!groups_files[@]}"; do
    ((group_index++))
    files_str="${groups_files[$key]}"
    # Convert to array (space-separated; filenames have no spaces)
    IFS=' ' read -r -a file_array <<< "$files_str"
    n=${#file_array[@]}

    # Build a human-readable tag
    IFS='-' read -r -a parts <<< "$key"
    scenario="${parts[0]}"
    fov="${parts[1]}"
    height="${parts[2]}"
    calib="${parts[3]}"
    policy="${parts[4]}"
    tag="${scenario}-${fov}fov-${height}cm-${calib}-${policy}"

    if [ $n -lt $MinFiles ]; then
        echo "[aggregate_benchmarks.sh] ($group_index/$total_groups) Skipping group '$tag' — $n file(s) < MinFiles=$MinFiles"
        continue
    fi

    echo "[aggregate_benchmarks.sh] ($group_index/$total_groups) Group: $tag  -> $n file(s), averaging via -f $n"
    for f in "${file_array[@]}"; do
        echo "    $(basename "$f")"
    done

    # Build command: python benchmark_plot.py -f N file1 ... fileN
    cmd=("$PYTHON" "$SCRIPT" "-f" "$n" "${file_array[@]}")

    if [ $DryRun -eq 1 ]; then
        echo "    [dry-run] ${cmd[*]}"
    else
        echo "    Running: ${cmd[*]}"
        "${cmd[@]}"
        if [ $? -ne 0 ]; then
            echo "    [aggregate_benchmarks.sh] WARNING: benchmark_plot.py exited with code $? for group $tag"
        fi
    fi

    # Keep these files for -Compare
    kept_files_for_compare+=("${file_array[@]}")
    echo ""
done

if [ $Compare -eq 1 ]; then
    if [ ${#kept_files_for_compare[@]} -eq 0 ]; then
        echo "[aggregate_benchmarks.sh] -C requested but no groups met MinFiles; skipping."
    else
        echo "[aggregate_benchmarks.sh] Running cross-group comparison across ${#kept_files_for_compare[@]} file(s)..."
        cmd=("$PYTHON" "$COMPARE_SCRIPT" "--min-files" "$MinFiles" "${kept_files_for_compare[@]}")
        if [ $DryRun -eq 1 ]; then
            echo "    [dry-run] ${cmd[*]}"
        else
            echo "    Running: ${cmd[*]}"
            "${cmd[@]}"
            if [ $? -ne 0 ]; then
                echo "    [aggregate_benchmarks.sh] WARNING: compare_groups.py exited with code $?"
            fi
        fi
    fi
    echo ""
fi

echo "[aggregate_benchmarks.sh] Done."