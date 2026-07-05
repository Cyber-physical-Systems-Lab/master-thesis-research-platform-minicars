# ══════════════════════════════════════════════════════════════════════════════
# experiments_script.ps1 — Experiment launcher for player_launcher.py  (PowerShell / Windows)
#
# Usage:  .\experiments_script.ps1 <command> [repetition]
#
#   <command>     One of the named scenarios below (.\experiments_script.ps1 list for full ref)
#   [repetition]  Optional integer — default 1.
#                 Sets --repetition N -> log: exp-log-S1-r3-90fov-calib-cooperative.json
#                 No counter file is read or written.
#
# Examples:
#   .\experiments_script.ps1 s1-coop-90-calib
#   .\experiments_script.ps1 s1-coop-90-calib 3
#   .\experiments_script.ps1 s3-ncoop-78-calib-vid 5
#   .\experiments_script.ps1 list
#
# First run — allow local scripts in VSCode terminal:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
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

param(
    [Parameter(Position=0, Mandatory=$true)]  [string]$Command,
    [Parameter(Position=1, Mandatory=$false)] [int]   $Repetition = 1
)

$PYTHON   = "python"
$SCRIPT   = "player_launcher.py"
$CALIB_90 = "calib-90_RMS-1p71.npz"
$CALIB_78 = "calib-78_RMS-2p02.npz"
$REP      = $Repetition

function Invoke-Launch ([string[]]$CmdArgs) {
    Write-Host "[experiments_script.ps1] $PYTHON $SCRIPT $CmdArgs --repetition $REP"
    & $PYTHON $SCRIPT @CmdArgs --repetition $REP
}

switch ($Command) {

    # ── SCENARIO S1  (car: 2) ─────────────────────────────────────────────────
    "s1-coop-90"            { Invoke-Launch "-n","2","--scenario","S1","--policy","cooperative","--fov","90","--run-name","90" }
    "s1-coop-90-vid"        { Invoke-Launch "-n","2","--scenario","S1","--policy","cooperative","--fov","90","--run-name","90", "--video-name","S1-coop-90-nocalib" }
    "s1-coop-90-calib"      { Invoke-Launch "-n","2","--scenario","S1","--policy","cooperative","--run-name","90","--calib-file",$CALIB_90 }
    "s1-coop-90-calib-vid"  { Invoke-Launch "-n","2","--scenario","S1","--policy","cooperative","--run-name","90","--calib-file",$CALIB_90,"--video-name","S1-coop-90" }
    "s1-coop-78"            { Invoke-Launch "-n","2","--scenario","S1","--policy","cooperative","--fov","78","--run-name","78" }
    "s1-coop-78-calib"      { Invoke-Launch "-n","2","--scenario","S1","--policy","cooperative","--run-name","78","--calib-file",$CALIB_78 }
    "s1-coop-78-calib-vid"  { Invoke-Launch "-n","2","--scenario","S1","--policy","cooperative","--run-name","78","--calib-file",$CALIB_78,"--video-name","S1-coop-78" }
    "s1-ncoop-90"           { Invoke-Launch "-n","2","--scenario","S1","--policy","non_cooperative","--fov","90","--run-name","90" }
    "s1-ncoop-90-calib"     { Invoke-Launch "-n","2","--scenario","S1","--policy","non_cooperative","--run-name","90","--calib-file",$CALIB_90 }
    "s1-ncoop-90-calib-vid" { Invoke-Launch "-n","2","--scenario","S1","--policy","non_cooperative","--run-name","90","--calib-file",$CALIB_90,"--video-name","S1-ncoop-90" }
    "s1-ncoop-78"           { Invoke-Launch "-n","2","--scenario","S1","--policy","non_cooperative","--fov","78","--run-name","78" }
    "s1-ncoop-78-calib"     { Invoke-Launch "-n","2","--scenario","S1","--policy","non_cooperative","--run-name","78","--calib-file",$CALIB_78 }
    "s1-ncoop-78-calib-vid" { Invoke-Launch "-n","2","--scenario","S1","--policy","non_cooperative","--run-name","78","--calib-file",$CALIB_78,"--video-name","S1-ncoop-78" }

    # ── SCENARIO S2  (cars: 0 2) ──────────────────────────────────────────────
    "s2-coop-90"            { Invoke-Launch "-n","1","2","--scenario","S2","--policy","cooperative","--fov","90","--run-name","90" }
    "s2-coop-90-calib"      { Invoke-Launch "-n","1","2","--scenario","S2","--policy","cooperative","--run-name","90","--calib-file",$CALIB_90 }
    "s2-coop-90-calib-vid"  { Invoke-Launch "-n","1","2","--scenario","S2","--policy","cooperative","--run-name","90","--calib-file",$CALIB_90,"--video-name","S2-coop-90" }
    "s2-coop-78"            { Invoke-Launch "-n","1","2","--scenario","S2","--policy","cooperative","--fov","78","--run-name","78" }
    "s2-coop-78-calib"      { Invoke-Launch "-n","1","2","--scenario","S2","--policy","cooperative","--run-name","78","--calib-file",$CALIB_78 }
    "s2-coop-78-calib-vid"  { Invoke-Launch "-n","1","2","--scenario","S2","--policy","cooperative","--run-name","78","--calib-file",$CALIB_78,"--video-name","S2-coop-78" }
    "s2-ncoop-90"           { Invoke-Launch "-n","1","2","--scenario","S2","--policy","non_cooperative","--fov","90","--run-name":"90" }
    "s2-ncoop-90-calib"     { Invoke-Launch "-n","1","2","--scenario","S2","--policy","non_cooperative","--run-name","90","--calib-file",$CALIB_90 }
    "s2-ncoop-90-calib-vid" { Invoke-Launch "-n","1","2","--scenario","S2","--policy","non_cooperative","--run-name","90","--calib-file",$CALIB_90,"--video-name","S2-ncoop-90" }
    "s2-ncoop-78"           { Invoke-Launch "-n","1","2","--scenario","S2","--policy","non_cooperative","--fov","78","--run-name","78" }
    "s2-ncoop-78-calib"     { Invoke-Launch "-n","1","2","--scenario","S2","--policy","non_cooperative","--run-name","78","--calib-file",$CALIB_78 }
    "s2-ncoop-78-calib-vid" { Invoke-Launch "-n","1","2","--scenario","S2","--policy","non_cooperative","--run-name":"78","--calib-file",$CALIB_78,"--video-name","S2-ncoop-78" }

    # ── SCENARIO S3  (cars: 0 2) ──────────────────────────────────────────────
    "s3-coop-90"            { Invoke-Launch "-n","1","2","--scenario","S3","--policy","cooperative","--fov","90","--run-name","90" }
    "s3-coop-90-calib"      { Invoke-Launch "-n","1","2","--scenario","S3","--policy","cooperative","--run-name","90","--calib-file",$CALIB_90 }
    "s3-coop-90-calib-vid"  { Invoke-Launch "-n","1","2","--scenario","S3","--policy","cooperative","--run-name","90","--calib-file",$CALIB_90,"--video-name","S3-coop-90" }
    "s3-coop-78"            { Invoke-Launch "-n","1","2","--scenario","S3","--policy","cooperative","--fov","78","--run-name","78" }
    "s3-coop-78-calib"      { Invoke-Launch "-n","1","2","--scenario","S3","--policy","cooperative","--run-name","78","--calib-file",$CALIB_78 }
    "s3-coop-78-calib-vid"  { Invoke-Launch "-n","1","2","--scenario","S3","--policy","cooperative","--run-name","78","--calib-file",$CALIB_78,"--video-name","S3-coop-78" }
    "s3-ncoop-90"           { Invoke-Launch "-n","1","2","--scenario","S3","--policy","non_cooperative","--fov","90","--run-name","90" }
    "s3-ncoop-90-calib"     { Invoke-Launch "-n","1","2","--scenario","S3","--policy","non_cooperative","--run-name","90","--calib-file",$CALIB_90 }
    "s3-ncoop-90-calib-vid" { Invoke-Launch "-n","1","2","--scenario","S3","--policy","non_cooperative","--run-name","90","--calib-file",$CALIB_90,"--video-name","S3-ncoop-90" }
    "s3-ncoop-78"           { Invoke-Launch "-n","1","2","--scenario","S3","--policy","non_cooperative","--fov","78","--run-name","78" }
    "s3-ncoop-78-calib"     { Invoke-Launch "-n","1","2","--scenario","S3","--policy","non_cooperative","--run-name","78","--calib-file",$CALIB_78 }
    "s3-ncoop-78-calib-vid" { Invoke-Launch "-n","1","2","--scenario","S3","--policy","non_cooperative","--run-name","78","--calib-file",$CALIB_78,"--video-name","S3-ncoop-78" }

    # ── SCENARIO S4  (cars, 0 1 2) ────────────────────────────────────────────
    "s4-coop-90"            { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","cooperative","--fov","90","--run-name","90" }
    "s4-coop-90-vid"        { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","cooperative","--fov","90","--run-name","90", "--video-name","S4-coop-90-nocalib" }
    "s4-coop-90-calib"      { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","cooperative","--run-name","90","--calib-file",$CALIB_90 }
    "s4-coop-90-calib-vid"  { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","cooperative","--run-name","90","--calib-file",$CALIB_90,"--video-name","S4-coop-90" }
    "s4-coop-78"            { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","cooperative","--fov","78","--run-name","78" }
    "s4-coop-78-vid"        { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","cooperative","--fov","78","--run-name","78", "--video-name","S4-coop-78-nocalib" }
    "s4-coop-78-calib"      { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","cooperative","--run-name","78","--calib-file",$CALIB_78 }
    "s4-coop-78-calib-vid"  { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","cooperative","--run-name","78","--calib-file",$CALIB_78,"--video-name","S4-coop-78" }
    "s4-ncoop-90"           { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","non_cooperative","--fov","90","--run-name","90" }
    "s4-ncoop-90-vid"       { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","non_cooperative","--fov","90","--run-name","90", "--video-name","S4-ncoop-90-nocalib" }
    "s4-ncoop-90-calib"     { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","non_cooperative","--run-name","90","--calib-file",$CALIB_90 }
    "s4-ncoop-90-calib-vid" { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","non_cooperative","--run-name","90","--calib-file",$CALIB_90,"--video-name","S4-ncoop-90" }
    "s4-ncoop-78"           { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","non_cooperative","--fov","78","--run-name","78" }
    "s4-ncoop-78-vid"       { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","non_cooperative","--fov","78","--run-name","78", "--video-name","S4-ncoop-78-nocalib" }
    "s4-ncoop-78-calib"     { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","non_cooperative","--run-name","78","--calib-file",$CALIB_78 }
    "s4-ncoop-78-calib-vid" { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","non_cooperative","--run-name","78","--calib-file",$CALIB_78,"--video-name","S4-ncoop-78" }

    # ── NO-BOOT VARIANTS ──────────────────────────────────────────────────────
    "s1-coop-90-noboot"     { Invoke-Launch "-n","2","--scenario","S1","--policy","cooperative","--fov","90","--run-name","90","--no-boot" }
    "s1-coop-78-noboot"     { Invoke-Launch "-n","2","--scenario","S1","--policy","cooperative","--fov","78","--run-name","78","--no-boot" }
    "s2-coop-90-noboot"     { Invoke-Launch "-n","1","2","--scenario","S2","--policy","cooperative","--fov","90","--run-name","90","--no-boot" }
    "s2-coop-78-noboot"     { Invoke-Launch "-n","0","2","--scenario","S2","--policy","cooperative","--fov","78","--run-name","78","--no-boot" }
    "s3-coop-90-noboot"     { Invoke-Launch "-n","0","2","--scenario","S3","--policy","cooperative","--fov","90","--run-name","90","--no-boot" }
    "s3-coop-78-noboot"     { Invoke-Launch "-n","0","2","--scenario","S3","--policy","cooperative","--fov","78","--run-name","78","--no-boot" }
    "s4-coop-90-noboot"     { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","cooperative","--fov","90","--run-name","90","--no-boot" }
    "s4-coop-78-noboot"     { Invoke-Launch "-n","0","1","2","--scenario","S4","--policy","cooperative","--fov","78","--run-name","78","--no-boot" }

    # ── CALIBRATION ───────────────────────────────────────────────────────────
    "calib-90" {
        Write-Host "[experiments_script.ps1] Calibration wizard (90 FOV)"
        & $PYTHON $SCRIPT "-n","0","--calibrate","--run-name","90"
    }
    "calib-78" {
        Write-Host "[experiments_script.ps1] Calibration wizard (78 FOV)"
        & $PYTHON $SCRIPT "-n","0","--calibrate","--run-name","78"
    }

    # ── LIST ──────────────────────────────────────────────────────────────────
    "list" {
        Write-Host ""
        Write-Host "Usage:  .\experiments_script.ps1 <command> [repetition]"
        Write-Host ""
        Write-Host "  repetition  -- integer, default 1."
        Write-Host "                 Sets --repetition N -> log: exp-log-S1-r3-90fov-calib-cooperative.json"
        Write-Host "                 No counter file is read or written."
        Write-Host ""
        Write-Host "Commands:  s{1-4}-{coop|ncoop}-{90|78}[-calib][-vid][-noboot]"
        Write-Host ""
        Write-Host "  Car sets per scenario:"
        Write-Host "    S1           -> -n 2"
        Write-Host "    S2, S3       -> -n 0 2"
        Write-Host "    S4           -> -n 0 1 2"
        Write-Host ""
        Write-Host "  Suffixes:"
        Write-Host "    (none)       no calib file, --fov passed explicitly"
        Write-Host "    -calib       load calibration file for the given FOV"
        Write-Host "    -vid         record annotated video (--video-name S{n}-{policy}-{fov})"
        Write-Host "    -noboot      --no-boot  (skip SSH launch / visualisation-only)"
        Write-Host ""
        Write-Host "  Examples:"
        Write-Host "    .\experiments_script.ps1 s1-coop-90-calib"
        Write-Host "    .\experiments_script.ps1 s1-coop-90-calib 3"
        Write-Host "    .\experiments_script.ps1 s3-ncoop-78-calib-vid 5"
        Write-Host "    .\experiments_script.ps1 s4-coop-90-noboot"
        Write-Host ""
        Write-Host "Calibration:"
        Write-Host "  calib-{90|78}   run calibration wizard"
        Write-Host ""
    }

    default {
        Write-Host "[experiments_script.ps1] Unknown command: '$Command'"
        Write-Host "Run '.\experiments_script.ps1 list' for available commands."
        exit 1
    }
}
