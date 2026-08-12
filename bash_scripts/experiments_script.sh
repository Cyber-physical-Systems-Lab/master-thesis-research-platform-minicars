#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# experiments_script.sh — Experiment launcher for player_launcher.py  (Bash / Linux)
#
# Usage:  ./experiments_script.sh <command> [repetition]
#
#   <command>     One of the named scenarios below (./experiments_script.sh list for full ref)
#   [repetition]  Optional integer — default 1.
#                 Sets --repetition N -> log: exp-log-S1-r3-90fov-calib-cooperative.json
#                 No counter file is read or written.
#
# Examples:
#   ./experiments_script.sh s1-coop-90-calib
#   ./experiments_script.sh s1-coop-90-calib 3
#   ./experiments_script.sh s3-ncoop-78-calib-vid 5
#   ./experiments_script.sh list
#
# First run — make script executable:
#   chmod +x experiments_script.sh
#
# Command naming convention:
#   s{1-4}         -> scenario S1-S4
#   -coop/-ncoop   -> cooperative / non-cooperative policy
#   -90/-78        -> FOV tag (also selects calibration file)
#   -calib         -> load calibration file for that FOV
#   -vid           -> also records --video-name
#   -noboot        -> skip SSH boot (cars already running / vis-only)
#   Car sets per scenario:
#     S1         -> -n 2
#     S2, S3     -> -n 0 2
#     S4         -> -n 0 1 2
# ══════════════════════════════════════════════════════════════════════════════

set -u

# Defaults
COMMAND="${1:-}"
REPETITION="${2:-1}"

if [ -z "$COMMAND" ]; then
    echo "Usage: $0 <command> [repetition]"
    echo "Run '$0 list' for available commands."
    exit 1
fi

PYTHON="python"
SCRIPT="./src/player_launcher.py"
CALIB_90="./calib_files/calib-90_RMS-1p71.npz"
CALIB_78="./calib_files/calib-78_RMS-2p02.npz"
REP="$REPETITION"

# Helper: launch player_launcher.py with given args
launch() {
    local args=("$@")
    echo "[experiments_script.sh] $PYTHON $SCRIPT ${args[*]} --repetition $REP"
    "$PYTHON" "$SCRIPT" "${args[@]}" --repetition "$REP"
}

case "$COMMAND" in

    # ── SCENARIO S1  (car: 2) ─────────────────────────────────────────────────
    "s1-coop-90")            launch -n 2 --scenario S1 --policy cooperative --fov 90 --run-name 90 ;;
    "s1-coop-90-vid")        launch -n 2 --scenario S1 --policy cooperative --fov 90 --run-name 90 --video-name S1-coop-90-nocalib ;;
    "s1-coop-90-calib")      launch -n 2 --scenario S1 --policy cooperative --run-name 90 --calib-file "$CALIB_90" ;;
    "s1-coop-90-calib-vid")  launch -n 2 --scenario S1 --policy cooperative --run-name 90 --calib-file "$CALIB_90" --video-name S1-coop-90 ;;
    "s1-coop-78")            launch -n 2 --scenario S1 --policy cooperative --fov 78 --run-name 78 ;;
    "s1-coop-78-calib")      launch -n 2 --scenario S1 --policy cooperative --run-name 78 --calib-file "$CALIB_78" ;;
    "s1-coop-78-calib-vid")  launch -n 2 --scenario S1 --policy cooperative --run-name 78 --calib-file "$CALIB_78" --video-name S1-coop-78 ;;
    "s1-ncoop-90")           launch -n 2 --scenario S1 --policy non_cooperative --fov 90 --run-name 90 ;;
    "s1-ncoop-90-calib")     launch -n 2 --scenario S1 --policy non_cooperative --run-name 90 --calib-file "$CALIB_90" ;;
    "s1-ncoop-90-calib-vid") launch -n 2 --scenario S1 --policy non_cooperative --run-name 90 --calib-file "$CALIB_90" --video-name S1-ncoop-90 ;;
    "s1-ncoop-78")           launch -n 2 --scenario S1 --policy non_cooperative --fov 78 --run-name 78 ;;
    "s1-ncoop-78-calib")     launch -n 2 --scenario S1 --policy non_cooperative --run-name 78 --calib-file "$CALIB_78" ;;
    "s1-ncoop-78-calib-vid") launch -n 2 --scenario S1 --policy non_cooperative --run-name 78 --calib-file "$CALIB_78" --video-name S1-ncoop-78 ;;

    # ── SCENARIO S2  (cars: 0 2) ──────────────────────────────────────────────
    "s2-coop-90")            launch -n 1 2 --scenario S2 --policy cooperative --fov 90 --run-name 90 ;;
    "s2-coop-90-calib")      launch -n 1 2 --scenario S2 --policy cooperative --run-name 90 --calib-file "$CALIB_90" ;;
    "s2-coop-90-calib-vid")  launch -n 1 2 --scenario S2 --policy cooperative --run-name 90 --calib-file "$CALIB_90" --video-name S2-coop-90 ;;
    "s2-coop-78")            launch -n 1 2 --scenario S2 --policy cooperative --fov 78 --run-name 78 ;;
    "s2-coop-78-calib")      launch -n 1 2 --scenario S2 --policy cooperative --run-name 78 --calib-file "$CALIB_78" ;;
    "s2-coop-78-calib-vid")  launch -n 1 2 --scenario S2 --policy cooperative --run-name 78 --calib-file "$CALIB_78" --video-name S2-coop-78 ;;
    "s2-ncoop-90")           launch -n 1 2 --scenario S2 --policy non_cooperative --fov 90 --run-name 90 ;;
    "s2-ncoop-90-calib")     launch -n 1 2 --scenario S2 --policy non_cooperative --run-name 90 --calib-file "$CALIB_90" ;;
    "s2-ncoop-90-calib-vid") launch -n 1 2 --scenario S2 --policy non_cooperative --run-name 90 --calib-file "$CALIB_90" --video-name S2-ncoop-90 ;;
    "s2-ncoop-78")           launch -n 1 2 --scenario S2 --policy non_cooperative --fov 78 --run-name 78 ;;
    "s2-ncoop-78-calib")     launch -n 1 2 --scenario S2 --policy non_cooperative --run-name 78 --calib-file "$CALIB_78" ;;
    "s2-ncoop-78-calib-vid") launch -n 1 2 --scenario S2 --policy non_cooperative --run-name 78 --calib-file "$CALIB_78" --video-name S2-ncoop-78 ;;

    # ── SCENARIO S3  (cars: 0 2) ──────────────────────────────────────────────
    "s3-coop-90")            launch -n 1 2 --scenario S3 --policy cooperative --fov 90 --run-name 90 ;;
    "s3-coop-90-calib")      launch -n 1 2 --scenario S3 --policy cooperative --run-name 90 --calib-file "$CALIB_90" ;;
    "s3-coop-90-calib-vid")  launch -n 1 2 --scenario S3 --policy cooperative --run-name 90 --calib-file "$CALIB_90" --video-name S3-coop-90 ;;
    "s3-coop-78")            launch -n 1 2 --scenario S3 --policy cooperative --fov 78 --run-name 78 ;;
    "s3-coop-78-calib")      launch -n 1 2 --scenario S3 --policy cooperative --run-name 78 --calib-file "$CALIB_78" ;;
    "s3-coop-78-calib-vid")  launch -n 1 2 --scenario S3 --policy cooperative --run-name 78 --calib-file "$CALIB_78" --video-name S3-coop-78 ;;
    "s3-ncoop-90")           launch -n 1 2 --scenario S3 --policy non_cooperative --fov 90 --run-name 90 ;;
    "s3-ncoop-90-calib")     launch -n 1 2 --scenario S3 --policy non_cooperative --run-name 90 --calib-file "$CALIB_90" ;;
    "s3-ncoop-90-calib-vid") launch -n 1 2 --scenario S3 --policy non_cooperative --run-name 90 --calib-file "$CALIB_90" --video-name S3-ncoop-90 ;;
    "s3-ncoop-78")           launch -n 1 2 --scenario S3 --policy non_cooperative --fov 78 --run-name 78 ;;
    "s3-ncoop-78-calib")     launch -n 1 2 --scenario S3 --policy non_cooperative --run-name 78 --calib-file "$CALIB_78" ;;
    "s3-ncoop-78-calib-vid") launch -n 1 2 --scenario S3 --policy non_cooperative --run-name 78 --calib-file "$CALIB_78" --video-name S3-ncoop-78 ;;

    # ── SCENARIO S4  (cars: 0 1 2) ────────────────────────────────────────────
    "s4-coop-90")            launch -n 0 1 2 --scenario S4 --policy cooperative --fov 90 --run-name 90 ;;
    "s4-coop-90-vid")        launch -n 0 1 2 --scenario S4 --policy cooperative --fov 90 --run-name 90 --video-name S4-coop-90-nocalib ;;
    "s4-coop-90-calib")      launch -n 0 1 2 --scenario S4 --policy cooperative --run-name 90 --calib-file "$CALIB_90" ;;
    "s4-coop-90-calib-vid")  launch -n 0 1 2 --scenario S4 --policy cooperative --run-name 90 --calib-file "$CALIB_90" --video-name S4-coop-90 ;;
    "s4-coop-78")            launch -n 0 1 2 --scenario S4 --policy cooperative --fov 78 --run-name 78 ;;
    "s4-coop-78-vid")        launch -n 0 1 2 --scenario S4 --policy cooperative --fov 78 --run-name 78 --video-name S4-coop-78-nocalib ;;
    "s4-coop-78-calib")      launch -n 0 1 2 --scenario S4 --policy cooperative --run-name 78 --calib-file "$CALIB_78" ;;
    "s4-coop-78-calib-vid")  launch -n 0 1 2 --scenario S4 --policy cooperative --run-name 78 --calib-file "$CALIB_78" --video-name S4-coop-78 ;;
    "s4-ncoop-90")           launch -n 0 1 2 --scenario S4 --policy non_cooperative --fov 90 --run-name 90 ;;
    "s4-ncoop-90-vid")       launch -n 0 1 2 --scenario S4 --policy non_cooperative --fov 90 --run-name 90 --video-name S4-ncoop-90-nocalib ;;
    "s4-ncoop-90-calib")     launch -n 0 1 2 --scenario S4 --policy non_cooperative --run-name 90 --calib-file "$CALIB_90" ;;
    "s4-ncoop-90-calib-vid") launch -n 0 1 2 --scenario S4 --policy non_cooperative --run-name 90 --calib-file "$CALIB_90" --video-name S4-ncoop-90 ;;
    "s4-ncoop-78")           launch -n 0 1 2 --scenario S4 --policy non_cooperative --fov 78 --run-name 78 ;;
    "s4-ncoop-78-vid")       launch -n 0 1 2 --scenario S4 --policy non_cooperative --fov 78 --run-name 78 --video-name S4-ncoop-78-nocalib ;;
    "s4-ncoop-78-calib")     launch -n 0 1 2 --scenario S4 --policy non_cooperative --run-name 78 --calib-file "$CALIB_78" ;;
    "s4-ncoop-78-calib-vid") launch -n 0 1 2 --scenario S4 --policy non_cooperative --run-name 78 --calib-file "$CALIB_78" --video-name S4-ncoop-78 ;;

    # ── NO-BOOT VARIANTS ──────────────────────────────────────────────────────
    "s1-coop-90-noboot")     launch -n 2 --scenario S1 --policy cooperative --fov 90 --run-name 90 --no-boot ;;
    "s1-coop-78-noboot")     launch -n 2 --scenario S1 --policy cooperative --fov 78 --run-name 78 --no-boot ;;
    "s2-coop-90-noboot")     launch -n 1 2 --scenario S2 --policy cooperative --fov 90 --run-name 90 --no-boot ;;
    "s2-coop-78-noboot")     launch -n 0 2 --scenario S2 --policy cooperative --fov 78 --run-name 78 --no-boot ;;
    "s3-coop-90-noboot")     launch -n 0 2 --scenario S3 --policy cooperative --fov 90 --run-name 90 --no-boot ;;
    "s3-coop-78-noboot")     launch -n 0 2 --scenario S3 --policy cooperative --fov 78 --run-name 78 --no-boot ;;
    "s4-coop-90-noboot")     launch -n 0 1 2 --scenario S4 --policy cooperative --fov 90 --run-name 90 --no-boot ;;
    "s4-coop-78-noboot")     launch -n 0 1 2 --scenario S4 --policy cooperative --fov 78 --run-name 78 --no-boot ;;

    # ── CALIBRATION ───────────────────────────────────────────────────────────
    "calib-90")
        echo "[experiments_script.sh] Calibration wizard (90 FOV)"
        "$PYTHON" "$SCRIPT" -n 0 --calibrate --run-name 90
        ;;
    "calib-78")
        echo "[experiments_script.sh] Calibration wizard (78 FOV)"
        "$PYTHON" "$SCRIPT" -n 0 --calibrate --run-name 78
        ;;

    # ── LIST ──────────────────────────────────────────────────────────────────
    "list")
        cat << EOF

Usage:  $0 <command> [repetition]

  repetition  -- integer, default 1.
                Sets --repetition N -> log: exp-log-S1-r3-90fov-calib-cooperative.json
                No counter file is read or written.

Commands:  s{1-4}-{coop|ncoop}-{90|78}[-calib][-vid][-noboot]

  Car sets per scenario:
    S1           -> -n 2
    S2, S3       -> -n 0 2
    S4           -> -n 0 1 2

  Suffixes:
    (none)       no calib file, --fov passed explicitly
    -calib       load calibration file for the given FOV
    -vid         record annotated video (--video-name S{n}-{policy}-{fov})
    -noboot      --no-boot  (skip SSH launch / visualisation-only)

  Examples:
    $0 s1-coop-90-calib
    $0 s1-coop-90-calib 3
    $0 s3-ncoop-78-calib-vid 5
    $0 s4-coop-90-noboot

Calibration:
  calib-{90|78}   run calibration wizard

EOF
        ;;

    *)
        echo "[experiments_script.sh] Unknown command: '$COMMAND'"
        echo "Run '$0 list' for available commands."
        exit 1
        ;;
esac