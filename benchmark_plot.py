#!/usr/bin/env python3
"""
benchmark_plot.py -- Analysis & visualisation tool for experiment JSON files.

Input : one or more .json files produced by auto_control.save_log
Output: PNG charts + a Markdown/CSV summary table

Usage
-----
python benchmark_plot.py experiment_log.json               # single file
python benchmark_plot.py S1_coop.json S1_noncoop.json      # compare runs
python benchmark_plot.py *.json                            # batch (output auto-derived from each file's metadata)
python benchmark_plot.py -f 5 exp-log-S1-1-calib.json  # average first 5 files
    exp-log-S1-2-calib.json exp-log-S1-3-calib.json \
    exp-log-S1-4-calib.json exp-log-S1-5-calib.json

JSON schema (actual auto_control.py output — adapted here, not the other
way around; see notes below for every field this script now reads)
----------------------------------------
Top-level keys
  meta               : {scenario, policy, calibration, dfov, camera_height_cm,
                        coordinate_origin, unit, px_per_cm, track_long_axis_cm,
                        cross_check, saved_at, n_frames, n_exp}
                       NOTE: auto_control.py does NOT write car_ids, d_col_px,
                       d_safe_px, or d_warn_px into meta. car_ids is derived
                       here from frames (_car_ids()); the D_* pixel thresholds
                       are hardcoded below as _D_COL/_D_WARN/_D_SAFE, matching
                       the controller's actual constants (25 / 57 / 115), with
                       d_col < d_warn < d_safe ordering (methodology-aligned
                       naming: D_WARN is the *stricter* near-miss boundary,
                       D_SAFE is the *farther*, most permissive decision zone).
  frames             : list of frame dicts
  track_ground_truth : {lane1_ref, lane2_ref} -- list of [x, y] pairs (top-level,
                        written once by save_log; also mirrored per-frame)
  summary / summary_by_car : populated here and written back

Per-frame dict
  t          : float timestamp (s)
  k          : int frame counter
  distances  : {"1-2": {euclidean_px, same_lane, lane_a, lane_b, seg_delta}, ...}
  cars       : {
    "<id>": {
      policy         : str,
      pose           : [x_px, y_px, theta_deg],
      lane           : int,
      segment        : str,
      command        : {servo: float, motor: float},
      waiting        : bool,
      lateral_error  : float,
      heading_error  : float,
      obstacle       : {driving_state: str, distance_px: float|null},
      events         : {minicar_events: [str], safety_events: [str]}  # NESTED,
                        not a flat list -- see _safety() below for the actual
                        tag vocabulary used by auto_control.py.
      emergency_stop : bool
    }
  }

There is no discrete "interaction_zones" block written by auto_control.py
(no {car_id, taus} list exists anywhere in the log). Waiting time is instead
derived here from contiguous True-runs in the per-frame "waiting" boolean
(see _waiting_durations()).

Charts produced (per file / per averaged group)
--------------------------
1. lateral_error_timeseries.png   lateral error over time + mean
2. heading_error_timeseries.png   heading error over time + mean
3. obstacle_dist_timeseries.png   obstacle dist + D_SAFE / D_WARN bands
4. safety_event_pie.png           collision / near-miss / safe proportions
5. commands_timeseries.png        servo + motor step commands over time
6. iv_distance_timeseries.png     estimated vs reference inter-vehicle dist
7. waiting_time_bar.png           per-interaction waiting time (derived)
8. error_cdf.png                  empirical CDF of lateral error (RQ1.2)
10. lane_timeline.png             lane assignment over time
11. trajectory_coverage.png       ground-truth track + car position scatter
12. emergency_stop_timeline.png   emergency-stop active frames (skipped when none occurred)
Multi-file
9. policy_comparison_bar.png      side-by-side summary across runs
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# -- Colour palette (Nexus design system) --------------------------------------
PAL = {
    "cooperative":     "#01696f",
    "non_cooperative": "#964219",
    "egocentric":      "#964219",  # alias kept for backward compat
    "safe":            "#437a22",
    "decision":        "#5591c7",
    "near_miss":       "#d19900",
    "collision":       "#a12c7b",
    "no_collision":    "#437a22",  # alias used by the split car/object safety_events schema
    "servo":           "#006494",
    "motor":           "#da7101",
    "lateral":         "#01696f",
    "heading":         "#7a39bb",
    "obs_dist":        "#a13544",
    "d_safe":          "#a12c7b",
    "d_warn":          "#d19900",
    "default":         "#28251d",
}

# Per-run scatter colours for trajectory chart (one per car ID)
_RUN_COLOURS = [
    "#01696f", "#964219", "#437a22", "#006494",
    "#7a39bb", "#a12c7b", "#d19900", "#a13544",
]

# -- Real controller safety thresholds (px) -------------------------------------
# auto_control.py never writes these into meta, so they are hardcoded here to
# match the actual D_COL / D_WARN / D_SAFE constants in auto_control.py.
# Ordering (methodology-aligned): d_col < d_warn < d_safe.
#   d_col  = physical-contact radius (collision)
#   d_warn = near-miss boundary (stricter, closer to collision)
#   d_safe = decision/yield boundary (farthest, most permissive)
_D_COL  = 25
_D_WARN = 57
_D_SAFE = 115

# ── I/O helpers ───────────────────────────────────────────────────────────────
def _results_dir(meta: dict) -> str:
    """Derive nested output path: ./exp/results/{scenario}-{dfov}fov-{calib}-{policy}/"""
    scenario = meta.get("scenario", "S?")
    policy   = meta.get("policy", "unknown")
    calib    = meta.get("calibration", "non-calib")
    dfov     = meta.get("dfov")
    dfov_part = f"{dfov}dFOV" if dfov else ""
    folder = f"{scenario}-{dfov_part}-{calib}-{policy}"
    path = os.path.join(".", "exp", "results", folder)
    os.makedirs(path, exist_ok=True)
    return path

def _multi_run_dir(runs: list) -> str:
    """
    Derive a shared output directory for multi-run comparison outputs
    (policy_comparison_bar.png, aggregated summary_table.*).

    The folder is placed alongside the per-run directories:
        ./exp/results/{scenario}-{dfov}-{calib}-multi/

    If all runs share the same policy, append that policy instead of "multi".
    Falls back to ./exp/results/multi/ when runs is empty.
    """
    if not runs:
        path = os.path.join(".", "exp", "results", "multi")
        os.makedirs(path, exist_ok=True)
        return path

    metas = [r.get("meta", {}) for r in runs]
    scenario  = metas[0].get("scenario", "S?")
    calib     = metas[0].get("calibration", "non-calib")
    dfov      = metas[0].get("dfov")
    dfov_part = f"{dfov}dFOV" if dfov else ""
    policies  = sorted({m.get("policy", "unknown") for m in metas})
    pol_tag   = policies[0] if len(policies) == 1 else "multi"
    folder = f"{scenario}-{dfov_part}-{calib}-{pol_tag}"
    path = os.path.join(".", "exp", "results", folder)
    os.makedirs(path, exist_ok=True)
    return path

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

# ── Schema helpers ────────────────────────────────────────────────────────────

def _car_ids(frames: List[dict]) -> List[str]:
    """Return sorted list of car_id strings present across all frames.

    auto_control.py never writes meta.car_ids, so this is always the
    authoritative source -- called fresh for every run instead of trusting
    a meta key that will never exist.
    """
    ids = set()
    for f in frames:
        ids.update(f.get("cars", {}).keys())
    return sorted(ids)

def _car_field(f: dict, car_id: str, *path, default=None):
    """Safely navigate frame -> cars -> car_id -> nested keys."""
    node = f.get("cars", {}).get(str(car_id), {})
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
    if node is None:
        return default
    return node

# ── Array extraction (per car_id) ─────────────────────────────────────────────

def _safety(f: dict, car_id: str) -> str:
    """
    Classify a single frame's COMBINED (car + object) safety state for
    *car_id*. Kept for backward compatibility with older logs / callers
    that still expect one flat safety label; prefer _safety_car() /
    _safety_object() for source-separated analysis.

    auto_control.py (>= v9) writes safety_events as a SPLIT dict:
        {"minicar_events": [...],
         "safety_events": {"car": [...], "object": [...]}}
    Older logs (<= v8) wrote safety_events as a flat list -- treated here
    as car-only, since that was the only source mixed into it at the time.
    minicar_events carries the emergency-stop / obstacle-avoidance tags
    actually used by the controller (e.g. "emergency_stop_straddle",
    "emergency_stop_both_lanes_blocked", "safety_stop",
    "obstacle_near_slowdown", "both_lanes_near_slowdown") and is checked
    as a fallback for emergency conditions that may not always propagate
    into safety_events.
    """
    ev = _car_field(f, car_id, "events", default={}) or {}
    raw_se = ev.get("safety_events", []) or []
    if isinstance(raw_se, dict):
        safety_tags = list(raw_se.get("car", [])) + list(raw_se.get("object", []))
    else:
        safety_tags = raw_se
    minicar_tags = set(ev.get("minicar_events", []) or [])

    if "collision" in safety_tags:
        return "collision"
    if ("emergency_stop_straddle" in minicar_tags or
            "emergency_stop_both_lanes_blocked" in minicar_tags):
        return "collision"
    if "near_miss" in safety_tags:
        return "near_miss"
    if "decision" in safety_tags:
        return "near_miss"
    if ("safety_stop" in minicar_tags or
            "obstacle_near_slowdown" in minicar_tags or
            "both_lanes_near_slowdown" in minicar_tags):
        return "near_miss"
    return "safe"

def _safety_source(f: dict, car_id: str, source: str) -> str:
    """
    Classify a single frame's safety state for *car_id*, restricted to one
    interaction SOURCE: "car" (car-to-car) or "object" (car-to-obstacle).

    Only meaningful for logs written by auto_control.py >= v9, where
    safety_events is split into {"car": [...], "object": [...]}. Older
    (<= v8) flat-list logs have no way to separate the two sources:
    - source="car"    falls back to the full flat list (legacy behaviour
                       treated everything as car-relevant).
    - source="object" falls back to "safe" (no data available).
    minicar_events fallback tags are only applied to the "object" source,
    since safety_stop / obstacle_near_slowdown / both_lanes_near_slowdown
    and the emergency_stop_* straddle/both-lanes-blocked tags are all
    obstacle-driven in the controller.
    """
    ev = _car_field(f, car_id, "events", default={}) or {}
    raw_se = ev.get("safety_events", []) or []
    minicar_tags = set(ev.get("minicar_events", []) or [])

    if isinstance(raw_se, dict):
        tags = raw_se.get(source, []) or []
    elif source == "car":
        tags = raw_se
    else:
        tags = []

    if "collision" in tags:
        return "collision"
    if source == "object" and ("emergency_stop_straddle" in minicar_tags or
                                "emergency_stop_both_lanes_blocked" in minicar_tags):
        return "collision"
    if "near_miss" in tags:
        return "near_miss"
    if "decision" in tags:
        return "decision"
    if source == "object" and ("safety_stop" in minicar_tags or
                                "obstacle_near_slowdown" in minicar_tags or
                                "both_lanes_near_slowdown" in minicar_tags):
        return "near_miss"
    return "safe"

def _safety_car(f: dict, car_id: str) -> str:
    """Car-to-car safety state for this frame (4-way: collision/near_miss/decision/safe)."""
    return _safety_source(f, car_id, "car")

def _safety_object(f: dict, car_id: str) -> str:
    """Car-to-object (obstacle) safety state for this frame (4-way)."""
    return _safety_source(f, car_id, "object")

def frames_to_arrays(frames: List[dict], car_id: str) -> dict:
    """
    Extract numpy arrays from the frame list for one car.

    All benchmark metrics -- pose tracking error, obstacle distance, safety
    events, waiting time -- are derived from these arrays.
    """
    t0 = frames[0]["t"] if frames else 0.0
    t = np.array([f["t"] - t0 for f in frames])
    lat = np.array([_car_field(f, car_id, "lateral_error", default=np.nan) for f in frames])
    hdg = np.array([_car_field(f, car_id, "heading_error", default=np.nan) for f in frames])
    servo = np.array([_car_field(f, car_id, "command", "servo", default=np.nan) for f in frames])
    motor = np.array([_car_field(f, car_id, "command", "motor", default=np.nan) for f in frames])
    # auto_control.py nests this under obstacle.distance_px (state key is
    # "driving_state", not "state" -- distance_px is unaffected either way).
    obs_dist = np.array([_car_field(f, car_id, "obstacle", "distance_px", default=np.nan) for f in frames])

    safety = [_safety(f, car_id) for f in frames]
    safety_car = [_safety_car(f, car_id) for f in frames]
    safety_object = [_safety_object(f, car_id) for f in frames]
    waiting = np.array([int(_car_field(f, car_id, "waiting", default=False) or False) for f in frames])
    lane = np.array([_car_field(f, car_id, "lane", default=1) for f in frames])

    emstop = np.array([int(_car_field(f, car_id, "emergency_stop", default=False) or False)
                        for f in frames])

    return dict(t=t, lat=lat, hdg=hdg, servo=servo, motor=motor,
                obs_dist=obs_dist, safety=safety, safety_car=safety_car,
                safety_object=safety_object, waiting=waiting, lane=lane,
                emstop=emstop)

def frames_to_iv_arrays(frames: List[dict]) -> dict:
    """
    Extract inter-vehicle distance arrays from the frame-level 'distances' dict.

    Handles both the legacy scalar format::

        {"1-2": 123.4}

    and the rich-dict format actually written by auto_control.py::

        {"1-2": {"euclidean_px": 123.4, "interaction_state": "safe",
                  "same_lane": True, "lane_a": 1, "lane_b": 1, "seg_delta": 42}}

    Returns a dict keyed by pair string -> dict with keys:
        tarr          - np.array of relative timestamps (s)
        distarr       - np.array of Euclidean pixel distances
        same_lane     - bool | None (majority vote over the run)
        lane_a/lane_b - int | None (lane of each car, from first frame)
        seg_delta_arr - np.array of |seg_idx_a - seg_idx_b| (NaN if missing)
    """
    t0 = frames[0]["t"] if frames else 0.0
    pairs: Dict[str, dict] = {}
    for f in frames:
        t_rel = f["t"] - t0
        for pair, raw in (f.get("distances") or {}).items():
            if isinstance(raw, dict):
                d = raw.get("euclidean_px", float("nan"))
                sl = raw.get("same_lane")
                la = raw.get("lane_a")
                lb = raw.get("lane_b")
                sdelta = raw.get("seg_delta")
            else:
                d, sl, la, lb, sdelta = float(raw), None, None, None, None
            rec = pairs.setdefault(pair, {"t": [], "d": [], "sl": [],
                                           "la": None, "lb": None, "sd": []})
            rec["t"].append(t_rel)
            rec["d"].append(d)
            rec["sl"].append(sl)
            rec["sd"].append(sdelta)
            if rec["la"] is None and la is not None:
                rec["la"] = la
            if rec["lb"] is None and lb is not None:
                rec["lb"] = lb
    out = {}
    for p, rec in pairs.items():
        sl_vals = [v for v in rec["sl"] if v is not None]
        sl_majority = (sum(sl_vals) / len(sl_vals)) >= 0.5 if sl_vals else None
        sd_arr = np.array(
            [v if v is not None else float("nan") for v in rec["sd"]], dtype=float
        )
        out[p] = {
            "tarr": np.array(rec["t"]),
            "distarr": np.array(rec["d"]),
            "same_lane": sl_majority,
            "lane_a": rec["la"],
            "lane_b": rec["lb"],
            "seg_delta_arr": sd_arr,
        }
    return out

def _waiting_durations(frames: List[dict], car_id: str) -> List[float]:
    """
    Derive discrete waiting-time durations (seconds) from contiguous
    True-runs in the per-frame "waiting" boolean.

    auto_control.py has no discrete interaction_zones/taus log -- only a
    per-frame per-car "waiting" flag (from _pp_waiting). Each maximal
    consecutive run of waiting=True is treated as one "interaction" and its
    wall-clock duration (t[end] - t[start of run]) is one tau sample. This
    replaces the old zones["interactions"][*]["taus"] lookup.
    """
    arrs = frames_to_arrays(frames, car_id)
    w = arrs["waiting"].astype(bool)
    t = arrs["t"]
    durations: List[float] = []
    in_run = False
    start_t = None
    for i in range(len(w)):
        if w[i] and not in_run:
            in_run, start_t = True, t[i]
        elif not w[i] and in_run:
            durations.append(float(t[i] - start_t))
            in_run = False
    if in_run and start_t is not None and len(t) > 0:
        durations.append(float(t[-1] - start_t))
    return durations

# ── Summary metrics ───────────────────────────────────────────────────────────

def compute_summary(frames: List[dict], meta: dict, car_id: str) -> dict:
    """
    Note: the *zones* parameter from the old interaction_zones-based schema
    has been removed -- waiting time and interaction counts are now derived
    directly from frames via _waiting_durations().
    """
    arrs = frames_to_arrays(frames, car_id)

    lat_errs = np.abs(arrs["lat"]); lat_errs = lat_errs[~np.isnan(lat_errs)]
    hdg_errs = np.abs(arrs["hdg"]); hdg_errs = hdg_errs[~np.isnan(hdg_errs)]
    iv_errors: List[float] = []

    n_col = arrs["safety"].count("collision")
    n_near = arrs["safety"].count("near_miss")

    n_col_car  = arrs["safety_car"].count("collision")
    n_near_car = arrs["safety_car"].count("near_miss")
    n_dec_car  = arrs["safety_car"].count("decision")
    n_col_obj  = arrs["safety_object"].count("collision")
    n_near_obj = arrs["safety_object"].count("near_miss")
    n_dec_obj  = arrs["safety_object"].count("decision")

    taus = _waiting_durations(frames, car_id)
    n_exp = max(len(taus), 1)

    n_emstop = int(np.sum(arrs["emstop"]))  # frames with emergency_stop=True

    obs_dists = arrs["obs_dist"]
    obs_valid = obs_dists[~np.isnan(obs_dists)]
    mean_obs = round(float(np.mean(obs_valid)), 4) if len(obs_valid) else None

    return dict(
        mean_lateral_error_px  = round(float(np.mean(lat_errs)), 4) if len(lat_errs) else None,
        mean_heading_error_deg = round(float(np.mean(hdg_errs)), 4) if len(hdg_errs) else None,
        mean_iv_dist_error_px  = round(float(np.mean(iv_errors)), 4) if iv_errors else None,
        mean_obs_dist_px       = mean_obs,
        n_collision            = n_col,
        n_near_miss            = n_near,
        n_collision_car        = n_col_car,
        n_near_miss_car        = n_near_car,
        n_decision_car         = n_dec_car,
        n_collision_object     = n_col_obj,
        n_near_miss_object     = n_near_obj,
        n_decision_object      = n_dec_obj,
        n_emergency_stop       = n_emstop,
        collision_rate         = round(n_col / n_exp, 4),
        near_miss_rate         = round(n_near / n_exp, 4),
        collision_rate_car     = round(n_col_car / n_exp, 4),
        near_miss_rate_car     = round(n_near_car / n_exp, 4),
        collision_rate_object  = round(n_col_obj / n_exp, 4),
        near_miss_rate_object  = round(n_near_obj / n_exp, 4),
        emergency_stop_rate    = round(n_emstop / max(len(frames), 1), 4),
        mean_waiting_time_s    = round(float(np.mean(taus)), 4) if taus else None,
    )

# ── Chart helpers ─────────────────────────────────────────────────────────────
FIGW, FIGH = 10, 4
DPI = 150

def _fig(h: float = FIGH):
    fig, ax = plt.subplots(figsize=(FIGW, h), dpi=DPI)
    fig.patch.set_facecolor("#f7f6f2")
    ax.set_facecolor("#f9f8f5")
    for spine in ax.spines.values():
        spine.set_edgecolor("#d4d1ca")
    ax.tick_params(colors="#7a7974")
    ax.title.set_color("#28251d")
    ax.xaxis.label.set_color("#7a7974")
    ax.yaxis.label.set_color("#7a7974")
    return fig, ax

def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart -> {path}")

# ── Per-run charts ────────────────────────────────────────────────────────────

def plot_lateral_error(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 1 -- lateral error time-series (all cars overlaid)."""
    fig, ax = _fig()
    for car_id, arrs in arrs_by_car.items():
        lat = np.abs(arrs["lat"])
        ax.plot(arrs["t"], lat, lw=1.2, alpha=0.85, label=f"Car {car_id}")
        ax.axhline(float(np.nanmean(lat)), lw=1.2, ls="--",
                   label=f"Mean car {car_id} ({np.nanmean(lat):.1f} px)")
    ax.set_xlabel("Time s"); ax.set_ylabel("Lateral error px")
    ax.set_title(f"Pose Tracking - Lateral Error {meta.get('scenario','')} {meta.get('policy','')}")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "lateral_error_timeseries.png"))

def plot_heading_error(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 2 -- heading error time-series (all cars overlaid)."""
    fig, ax = _fig()
    for car_id, arrs in arrs_by_car.items():
        hdg = np.abs(arrs["hdg"])
        ax.plot(arrs["t"], hdg, lw=1.2, alpha=0.85, label=f"Car {car_id}")
        ax.axhline(float(np.nanmean(hdg)), lw=1.2, ls="--",
                   label=f"Mean car {car_id} ({np.nanmean(hdg):.1f} deg)")
    ax.set_xlabel("Time s"); ax.set_ylabel("Heading error deg")
    ax.set_title(f"Pose Tracking - Heading Error {meta.get('scenario','')} {meta.get('policy','')}")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "heading_error_timeseries.png"))

def plot_obstacle_dist(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 3 -- obstacle distance with D_COL / D_WARN / D_SAFE reference bands.

    Uses the hardcoded _D_COL/_D_WARN/_D_SAFE (matching auto_control.py's
    real constants) since meta never carries d_col_px/d_warn_px/d_safe_px.
    Skipped (with informational text) when no obstacle data is present --
    e.g. S1 scenarios where no obstacles are placed on the track.
    """
    has_obs = any(
        not np.all(np.isnan(arrs["obs_dist"]))
        for arrs in arrs_by_car.values()
    )
    d_col, d_warn, d_safe = _D_COL, _D_WARN, _D_SAFE
    fig, ax = _fig()
    if not has_obs:
        ax.text(0.5, 0.5,
                "No obstacle data in this scenario\n(obstacles not present - e.g. S1)",
                transform=ax.transAxes, ha="center", va="center",
                color="#7a7974", fontsize=10)
        ax.set_title(f"Obstacle Distance {meta.get('scenario','')} {meta.get('policy','')}")
        _save(fig, os.path.join(outdir, "obstacle_dist_timeseries.png"))
        return
    for car_id, arrs in arrs_by_car.items():
        ax.plot(arrs["t"], arrs["obs_dist"], lw=1.2, alpha=0.85,
                color=PAL["obs_dist"], label=f"Obs dist car {car_id}")
    ax.axhline(d_col, color="#28251d", lw=1.2, ls="-", label=f"D_COL {d_col} px")
    ax.axhline(d_warn, color=PAL["d_warn"], lw=1.5, ls="--", label=f"D_WARN {d_warn} px")
    ax.axhline(d_safe, color=PAL["d_safe"], lw=1.5, ls=":", label=f"D_SAFE {d_safe} px")
    t_all = np.concatenate([a["t"] for a in arrs_by_car.values()])
    if len(t_all):
        ax.fill_between(t_all, 0, d_col, alpha=0.12, color=PAL["collision"])
        ax.fill_between(t_all, d_col, d_warn, alpha=0.08, color=PAL["d_warn"])
        ax.fill_between(t_all, d_warn, d_safe, alpha=0.06, color=PAL["d_safe"])
    ax.set_xlabel("Time s"); ax.set_ylabel("Distance px")
    ax.set_title(f"Obstacle Distance {meta.get('scenario','')} {meta.get('policy','')}")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "obstacle_dist_timeseries.png"))

def _plot_one_safety_pie(all_safety: list, title: str, fname: str,
                          meta: dict, outdir: str) -> None:
    """Shared pie-chart renderer for a single safety-state series (either
    combined, car-to-car, or car-to-object), 4-way: collision / near_miss /
    decision / safe."""
    counts = {k: all_safety.count(k) for k in ("collision", "near_miss", "decision", "safe")}
    labels = [k for k, v in counts.items() if v > 0]
    vals = [counts[k] for k in labels]
    colors = [PAL[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    fig.patch.set_facecolor("#f7f6f2")
    if not vals:
        ax.text(0.5, 0.5, "No safety events recorded\n(all frames: safe)",
                transform=ax.transAxes, ha="center", va="center",
                color="#7a7974", fontsize=10)
        ax.set_title(f"{title} {meta.get('scenario','')} {meta.get('policy','')}", color="#28251d")
        _save(fig, os.path.join(outdir, fname))
        return
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors, autopct="%.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor="#f7f6f2", linewidth=1.5))
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(9)
    ax.set_title(f"{title} {meta.get('scenario','')} {meta.get('policy','')}", color="#28251d")
    _save(fig, os.path.join(outdir, fname))

def plot_safety_pie(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 4 -- combined (legacy) safety event distribution, aggregated
    over all cars and BOTH sources (car-to-car + car-to-object). Kept for
    backward compatibility; prefer plot_safety_pie_car / plot_safety_pie_object
    for source-separated views."""
    all_safety = [s for arrs in arrs_by_car.values() for s in arrs["safety"]]
    _plot_one_safety_pie(all_safety, "Safety Event Distribution (combined)",
                          "safety_event_pie.png", meta, outdir)

def plot_safety_pie_car(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 4a -- car-to-car safety event distribution (aggregated over all cars)."""
    all_safety = [s for arrs in arrs_by_car.values() for s in arrs.get("safety_car", [])]
    _plot_one_safety_pie(all_safety, "Safety Event Distribution - Car-to-Car",
                          "safety_event_pie_car.png", meta, outdir)

def plot_safety_pie_object(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 4b -- car-to-object (obstacle) safety event distribution
    (aggregated over all cars)."""
    all_safety = [s for arrs in arrs_by_car.values() for s in arrs.get("safety_object", [])]
    _plot_one_safety_pie(all_safety, "Safety Event Distribution - Car-to-Object",
                          "safety_event_pie_object.png", meta, outdir)

def plot_commands(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 5 -- quantised servo + motor commands (one subplot per car)."""
    n = len(arrs_by_car)
    if n == 0:
        return
    fig, axes = plt.subplots(n * 2, 1, figsize=(FIGW, 3 * n), dpi=DPI, sharex=True)
    fig.patch.set_facecolor("#f7f6f2")
    if n * 2 == 1:
        axes = [axes, axes]
    axes = list(np.array(axes).flatten())
    for idx, (car_id, arrs) in enumerate(arrs_by_car.items()):
        ax1, ax2 = axes[idx * 2], axes[idx * 2 + 1]
        for ax in (ax1, ax2):
            ax.set_facecolor("#f9f8f5")
            for sp in ax.spines.values(): sp.set_edgecolor("#d4d1ca")
            ax.tick_params(colors="#7a7974")
        ax1.step(arrs["t"], arrs["servo"], color=PAL["servo"], lw=1.2, where="post")
        ax1.set_ylabel(f"Servo rad\n(car {car_id})")
        ax1.axhline(0, color="#d4d1ca", lw=0.8)
        ax2.step(arrs["t"], arrs["motor"], color=PAL["motor"], lw=1.2, where="post")
        ax2.set_ylabel(f"Motor\n(car {car_id})")
    axes[-1].set_xlabel("Time s")
    axes[0].set_title(f"Commands - Servo & Motor {meta.get('scenario','')} {meta.get('policy','')}",
                       color="#28251d")
    _save(fig, os.path.join(outdir, "commands_timeseries.png"))

def plot_iv_distance(frames: List[dict], meta: dict, outdir: str) -> None:
    """Chart 6 -- inter-vehicle distance over time.

    Two sub-plots:
      Top    - same-lane pairs (safety-critical: D_WARN / D_SAFE bands)
      Bottom - cross-lane pairs (lateral proximity between lanes)
    Each pair is plotted as Euclidean px distance.
    A secondary y-axis on the same-lane sub-plot shows the mean
    segment-index delta (|seg_a - seg_b|) as a dashed grey trace,
    giving a path-distance approximation without requiring the full curve.
    """
    iv = frames_to_iv_arrays(frames)
    if not iv:
        return
    d_col, d_warn, d_safe = _D_COL, _D_WARN, _D_SAFE

    same_pairs = {p: v for p, v in iv.items() if v["same_lane"] is True}
    cross_pairs = {p: v for p, v in iv.items() if v["same_lane"] is False}
    unk_pairs = {p: v for p, v in iv.items() if v["same_lane"] is None}
    same_pairs.update(unk_pairs)

    n_panels = (1 if same_pairs else 0) + (1 if cross_pairs else 0)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(FIGW, FIGH * n_panels),
                              dpi=DPI, sharex=False)
    fig.patch.set_facecolor("#f7f6f2")
    if n_panels == 1:
        axes = [axes]
    for ax in axes:
        ax.set_facecolor("#f9f8f5")
        for sp in ax.spines.values():
            sp.set_edgecolor("#d4d1ca")
        ax.tick_params(colors="#7a7974")

    panel = 0
    title_sfx = f"{meta.get('scenario','')} {meta.get('policy','')}"

    if same_pairs:
        ax = axes[panel]; panel += 1
        for pair, v in same_pairs.items():
            lbl = f"d({pair}) L{v['lane_a']}->L{v['lane_b']}"
            ax.plot(v["tarr"], v["distarr"], lw=1.4, label=lbl)
        all_t = np.concatenate([v["tarr"] for v in same_pairs.values()])
        ax.fill_between(all_t, 0, d_col, alpha=0.10, color=PAL["collision"])
        ax.fill_between(all_t, d_col, d_warn, alpha=0.08, color=PAL["d_warn"])
        ax.fill_between(all_t, d_warn, d_safe, alpha=0.06, color=PAL["d_safe"])
        ax.axhline(d_warn, color=PAL["d_warn"], lw=1.4, ls="--",
                   label=f"D_WARN {d_warn} px")
        ax.axhline(d_safe, color=PAL["d_safe"], lw=1.2, ls=":",
                   label=f"D_SAFE {d_safe} px")
        has_seg = any(not np.all(np.isnan(v["seg_delta_arr"])) for v in same_pairs.values())
        if has_seg:
            ax2 = ax.twinx()
            ax2.set_facecolor("none")
            for pair, v in same_pairs.items():
                if not np.all(np.isnan(v["seg_delta_arr"])):
                    ax2.plot(v["tarr"], v["seg_delta_arr"],
                             lw=1.0, ls="--", alpha=0.55, color="#c8c6c0",
                             label=f"dseg({pair})")
            ax2.set_ylabel("Segment-index gap (samples)", color="#c8c6c0", fontsize=8)
            ax2.tick_params(colors="#c8c6c0")
            ax2.legend(loc="upper right", fontsize=7, framealpha=0.5)
        ax.set_xlabel("Time s")
        ax.set_ylabel("Euclidean distance px")
        ax.set_title(f"IV Distance - Same-Lane {title_sfx}")
        ax.legend(loc="upper left", framealpha=0.7, fontsize=8)

    if cross_pairs:
        ax = axes[panel]
        for pair, v in cross_pairs.items():
            lbl = f"d({pair}) L{v['lane_a']}<->L{v['lane_b']}"
            ax.plot(v["tarr"], v["distarr"], lw=1.4, ls="-.", label=lbl)
        ax.set_xlabel("Time s")
        ax.set_ylabel("Euclidean distance px")
        ax.set_title(f"IV Distance - Cross-Lane {title_sfx}")
        ax.legend(loc="upper left", framealpha=0.7, fontsize=8)

    _save(fig, os.path.join(outdir, "iv_distance_timeseries.png"))

def plot_waiting_times(frames: List[dict], car_ids: List[str], meta: dict, outdir: str) -> None:
    """Chart 7 -- per-interaction waiting time (derived from the "waiting" flag).

    Replaces the old interaction_zones/taus lookup, which does not exist in
    auto_control.py's output. All waiting-run durations across all cars are
    pooled and plotted as one bar per interaction (in first-seen order).
    """
    taus: List[float] = []
    for cid in car_ids:
        taus.extend(_waiting_durations(frames, cid))
    if not taus:
        return
    idxs = list(range(1, len(taus) + 1))
    fig, ax = _fig(4)
    pol_color = PAL.get(meta.get("policy", "cooperative"), PAL["default"])
    ax.bar(idxs, taus, color=pol_color, alpha=0.85,
           edgecolor="#f7f6f2", linewidth=1.2)
    mean_t = float(np.mean(taus))
    ax.axhline(mean_t, color="#28251d", lw=1.5, ls="--", label=f"Mean {mean_t:.2f} s")
    ax.set_xlabel("Interaction"); ax.set_ylabel("Waiting time s")
    ax.set_title(f"Per-Interaction Waiting Time {meta.get('scenario','')} {meta.get('policy','')}")
    ax.set_xticks(idxs); ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "waiting_time_bar.png"))

def plot_error_cdf(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 8 -- empirical CDF of lateral error (supports RQ1.2)."""
    fig, ax = _fig()
    any_plotted = False
    for car_id, arrs in arrs_by_car.items():
        errs = np.sort(np.abs(arrs["lat"]))
        errs = errs[~np.isnan(errs)]
        if len(errs) == 0:
            continue
        cdf = np.arange(1, len(errs) + 1) / len(errs)
        ax.plot(errs, cdf, lw=1.8, label=f"Car {car_id}")
        any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "No lateral error data available",
                transform=ax.transAxes, ha="center", va="center",
                color="#7a7974", fontsize=10)
    ax.set_xlabel("Lateral error px"); ax.set_ylabel("Cumulative probability")
    ax.set_title(f"CDF - Lateral Error {meta.get('scenario','')} {meta.get('policy','')}")
    ax.grid(True, alpha=0.3, color="#d4d1ca"); ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "error_cdf.png"))

def plot_lane_timeline(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 10 -- lane assignment over time per car."""
    fig, ax = _fig()
    for car_id, arrs in arrs_by_car.items():
        lane = np.where(np.isnan(arrs["lane"].astype(float)), 1, arrs["lane"])
        ax.step(arrs["t"], lane, lw=1.4, where="post", label=f"Car {car_id}")
    ax.set_yticks([1, 2]); ax.set_yticklabels(["Lane 1 (inner)", "Lane 2 (outer)"])
    ax.set_xlabel("Time s"); ax.set_ylabel("Lane")
    ax.set_title(f"Lane Assignment Over Time {meta.get('scenario','')} {meta.get('policy','')}")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "lane_timeline.png"))

def plot_emergency_stop_timeline(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 12 -- emergency stop active frames over time.

    Plots a binary (0/1) step trace per car showing frames where
    emergency_stop was True. Skipped when no emergency stops occurred
    (e.g. nominal S1 runs with no obstacles and no lane straddling).
    """
    any_emstop = any(
        bool(np.any(arrs.get("emstop", np.zeros(1)) > 0))
        for arrs in arrs_by_car.values()
    )
    if not any_emstop:
        return

    fig, ax = _fig(3)
    for car_id, arrs in arrs_by_car.items():
        em = arrs.get("emstop", np.zeros_like(arrs["t"]))
        ax.step(arrs["t"], em, lw=1.4, where="post", label=f"Car {car_id}")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "EMERGENCY"])
    ax.set_xlabel("Time s"); ax.set_ylabel("Emergency stop active")
    ax.set_title(f"Emergency Stop Active Frames {meta.get('scenario','')} {meta.get('policy','')}")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "emergency_stop_timeline.png"))

# ── Trajectory coverage chart (ground-truth + car positions) ─────────────────

def _extract_car_positions(frames: List[dict], car_id: str) -> List[tuple]:
    """Return list of (x_px, y_px) from the pose field for *car_id*."""
    pts = []
    for f in frames:
        p = _car_field(f, car_id, "pose", default=None)
        if p and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
    return pts

def plot_trajectory(runs_data: List[dict], outdir: str, avg_mode: bool = False) -> None:
    """
    Chart 11 -- trajectory coverage.

    Draws the track ground-truth curves (lane centrelines) on a dark
    background, then overlays the car position scatter points from one or
    more runs. In avg_mode (multi-file averaging) all runs are merged so
    the point cloud density reflects the full N-repetition dataset.

    The ground-truth geometry is taken from the first run that has a
    non-empty top-level 'track_ground_truth' key (written once by save_log;
    auto_control.py also mirrors the same object into every frame, but this
    chart only needs the top-level copy).
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=DPI)
    fig.patch.set_facecolor("#f7f6f2")
    ax.set_facecolor("#171614")
    for spine in ax.spines.values():
        spine.set_edgecolor("#d4d1ca")
    ax.tick_params(colors="#7a7974")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    gt = next((r.get("track_ground_truth") for r in runs_data
               if r.get("track_ground_truth")), None)

    GT_STYLE = {
        "lane1_ref": ("#6daa45", 2.0, "Lane 1 ideal path"),
        "lane2_ref": ("#5591c7", 2.0, "Lane 2 ideal path"),
    }
    if gt:
        for key, (col, lw, lbl) in GT_STYLE.items():
            pts = gt.get(key, [])
            if pts:
                arr = np.array(pts, dtype=float)
                arr_closed = np.vstack([arr, arr[[0]]])
                ax.plot(arr_closed[:, 0], arr_closed[:, 1],
                        color=col, lw=lw, alpha=0.75, label=lbl)
    else:
        ax.text(0.5, 0.5, "No track_ground_truth in log\n(run at least one frame first)",
                transform=ax.transAxes, ha="center", va="center",
                color="#7a7974", fontsize=9)

    all_car_ids = sorted(set(
        cid
        for r in runs_data
        for f in r.get("frames", [])
        for cid in f.get("cars", {}).keys()
    ))
    for ci, car_id in enumerate(all_car_ids):
        colour = _RUN_COLOURS[ci % len(_RUN_COLOURS)]
        all_x, all_y = [], []
        for r in runs_data:
            for px, py in _extract_car_positions(r.get("frames", []), car_id):
                all_x.append(px); all_y.append(py)
        if all_x:
            ax.scatter(all_x, all_y, s=2, alpha=0.35, color=colour,
                       label=f"Car {car_id} positions", rasterized=True)

    meta0 = runs_data[0].get("meta", {}) if runs_data else {}
    n_runs = len(runs_data)
    suffix = f" ({n_runs} runs averaged)" if avg_mode and n_runs > 1 else ""
    _unit = meta0.get("unit", "px")
    _dfov = meta0.get("dfov")
    _dfov_lbl = f" dFOV={_dfov}deg" if _dfov else ""
    ax.set_title(
        f"Trajectory Coverage {meta0.get('scenario', '')} "
        f"{meta0.get('policy', '')}{_dfov_lbl}{suffix}",
        color="#28251d",
    )
    ax.set_xlabel(f"x ({_unit})", color="#7a7974")
    ax.set_ylabel(f"y ({_unit})", color="#7a7974")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6,
              facecolor="#1c1b19", labelcolor="white")
    _save(fig, os.path.join(outdir, "trajectory_coverage.png"))

# ── Multi-run comparison chart ────────────────────────────────────────────────

def plot_policy_comparison(runs: List[dict], outdir: str) -> None:
    """Chart 9 -- side-by-side summary metrics: cooperative vs non-cooperative."""
    metrics = [
        ("mean_lateral_error_px",  "Mean lat. error px"),
        ("mean_heading_error_deg", "Mean hdg error deg"),
        ("mean_waiting_time_s",    "Mean waiting time s"),
        ("collision_rate",         "Collision rate"),
        ("near_miss_rate",         "Near-miss rate"),
    ]
    labels = []
    for r in runs:
        for cid in r.get("meta", {}).get("car_ids", ["?"]):
            labels.append(f"{r['meta'].get('scenario','')}\n{r['meta'].get('policy','?')[:4].upper()} c{cid}")

    n_met = len(metrics)
    fig, axes = plt.subplots(1, n_met, figsize=(3.5 * n_met, 5), dpi=DPI)
    fig.patch.set_facecolor("#f7f6f2")
    if n_met == 1:
        axes = [axes]

    for ax, (key, ylabel) in zip(axes, metrics):
        vals, colors = [], []
        for r in runs:
            pol = r["meta"].get("policy", "cooperative")
            for cid in r.get("meta", {}).get("car_ids", ["?"]):
                summ = r.get("summary_by_car", {}).get(str(cid), r.get("summary", {}))
                vals.append(summ.get(key))
                colors.append(PAL.get(pol, PAL["default"]))

        has_data = [v is not None for v in vals]
        bar_vals = [v if v is not None else 0 for v in vals]
        x = np.arange(len(labels))
        bars = ax.bar(
            [xi for xi, h in enumerate(has_data) if h],
            [bar_vals[i] for i in range(len(labels)) if has_data[i]],
            width=0.65,
            color=[colors[i] for i in range(len(labels)) if has_data[i]],
            alpha=0.88, edgecolor="#f7f6f2", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8, color="#7a7974")
        ax.set_facecolor("#f9f8f5")
        for sp in ax.spines.values(): sp.set_edgecolor("#d4d1ca")
        ax.tick_params(colors="#7a7974")
        shown_vals = [bar_vals[i] for i in range(len(labels)) if has_data[i]]
        for bar, val in zip(bars, shown_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(bar_vals + [1e-9]),
                    f"{val:.3g}", ha="center", va="bottom",
                    fontsize=7, color="#28251d")

    fig.suptitle("Policy / Scenario Comparison - Summary Metrics",
                 color="#28251d", fontsize=11)
    _save(fig, os.path.join(outdir, "policy_comparison_bar.png"))

# ── Summary table ─────────────────────────────────────────────────────────────

SUMMARY_COLS = [
    "scenario", "policy", "calibration", "car_id", "n_frames",
    "mean_lateral_error_px", "mean_heading_error_deg", "mean_iv_dist_error_px",
    "mean_obs_dist_px",
    "n_collision", "n_near_miss", "n_emergency_stop",
    "collision_rate", "near_miss_rate", "emergency_stop_rate",
    "n_collision_car", "n_near_miss_car", "n_decision_car",
    "collision_rate_car", "near_miss_rate_car",
    "n_collision_object", "n_near_miss_object", "n_decision_object",
    "collision_rate_object", "near_miss_rate_object",
    "mean_waiting_time_s",
]

def write_summary_table(runs: List[dict], outdir: str) -> None:
    rows = []
    for r in runs:
        for cid in r.get("meta", {}).get("car_ids", ["?"]):
            summ = r.get("summary_by_car", {}).get(str(cid), r.get("summary", {}))
            row = {**r["meta"], "car_id": cid, **summ}
            rows.append({c: row.get(c) for c in SUMMARY_COLS})

    csv_path = os.path.join(outdir, "summary_table.csv")
    def _fmt(v, col: str) -> str:
        """
        Rules:
        - Count columns (n_*) and rate columns -> "0" when value is 0 (not blank)
        - Obstacle / waiting metrics -> "-" when None (not present in this scenario)
        - All other None -> "-"
        """
        COUNT_COLS = {"n_collision", "n_near_miss", "n_emergency_stop",
                      "collision_rate", "near_miss_rate", "emergency_stop_rate",
                      "n_collision_car", "n_near_miss_car", "n_decision_car",
                      "collision_rate_car", "near_miss_rate_car",
                      "n_collision_object", "n_near_miss_object", "n_decision_object",
                      "collision_rate_object", "near_miss_rate_object"}
        if v is None:
            return "0" if col in COUNT_COLS else "-"
        return str(v)

    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(SUMMARY_COLS) + "\n")
        for row in rows:
            fh.write(",".join(_fmt(row[c], c) for c in SUMMARY_COLS) + "\n")
    print(f"  table -> {csv_path}")

    md_path = os.path.join(outdir, "summary_table.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(" | ".join(SUMMARY_COLS) + "\n")
        fh.write(" | ".join(["---"] * len(SUMMARY_COLS)) + "\n")
        for row in rows:
            fh.write(" | ".join(_fmt(row[c], c) for c in SUMMARY_COLS) + "\n")
    print(f"  table -> {md_path}")

# ── Multi-file averaging ──────────────────────────────────────────────────────

def _resample(arr: np.ndarray, n: int = 1000) -> np.ndarray:
    """Resample *arr* to exactly *n* points using linear interpolation."""
    if len(arr) == 0:
        return np.full(n, np.nan)
    idx = np.linspace(0, len(arr) - 1, n)
    return np.interp(idx, np.arange(len(arr)), arr)

def process_averaged_files(paths: List[str]) -> dict:
    """
    Load N JSON files, average their per-car metrics, produce charts from
    the averaged arrays, and return a synthetic run dict.

    Averaging strategy
    ------------------
    Each per-car numeric array (lateral_error, heading_error, etc.) is
    resampled to N_RESAMPLE=1000 points (normalised time axis) before
    stacking. np.nanmean across the N runs then produces a single averaged
    trace, which is used for all charts.

    Safety labels use majority-vote per time step.
    Summary statistics are recomputed from each file's frames, then averaged
    numerically across runs.
    """
    N_RESAMPLE = 1000
    print(f"\nAveraging {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    loaded = [load_json(p) for p in paths]

    meta = dict(loaded[0]["meta"])
    meta["n_frames"] = int(round(np.mean([len(d["frames"]) for d in loaded])))
    meta["averaged_runs"] = len(paths)
    meta["source_files"] = [Path(p).name for p in paths]

    all_car_ids: List[str] = sorted(set(
        cid for d in loaded for f in d["frames"]
        for cid in f.get("cars", {}).keys()
    ))
    meta["car_ids"] = all_car_ids

    avg_arrs_by_car: dict = {}
    for car_id in all_car_ids:
        run_arrs = [frames_to_arrays(d["frames"], car_id) for d in loaded]

        numeric_keys = ["lat", "hdg", "servo", "motor", "obs_dist", "waiting", "lane", "emstop"]
        averaged: dict = {}
        for k in numeric_keys:
            stacked = np.stack(
                [_resample(ra[k].astype(float), N_RESAMPLE) for ra in run_arrs],
                axis=0)
            averaged[k] = np.nanmean(stacked, axis=0)

        mean_dur = float(np.mean([
            ra["t"][-1] if len(ra["t"]) > 0 else 1.0 for ra in run_arrs
        ]))
        averaged["t"] = np.linspace(0, mean_dur, N_RESAMPLE)

        def _majority_vote(step_i: int, key: str) -> str:
            votes = []
            for ra in run_arrs:
                series = ra.get(key, [])
                n = len(series)
                if n == 0:
                    votes.append("safe")
                    continue
                idx_r = min(int(round(step_i * (n - 1) / (N_RESAMPLE - 1))), n - 1)
                votes.append(series[idx_r])
            return Counter(votes).most_common(1)[0][0]

        averaged["safety"] = [_majority_vote(i, "safety") for i in range(N_RESAMPLE)]
        averaged["safety_car"] = [_majority_vote(i, "safety_car") for i in range(N_RESAMPLE)]
        averaged["safety_object"] = [_majority_vote(i, "safety_object") for i in range(N_RESAMPLE)]
        avg_arrs_by_car[car_id] = averaged

    rundir = _results_dir(meta)

    plot_lateral_error(avg_arrs_by_car, meta, rundir)
    plot_heading_error(avg_arrs_by_car, meta, rundir)
    plot_obstacle_dist(avg_arrs_by_car, meta, rundir)
    plot_safety_pie(avg_arrs_by_car, meta, rundir)
    plot_safety_pie_car(avg_arrs_by_car, meta, rundir)
    plot_safety_pie_object(avg_arrs_by_car, meta, rundir)
    plot_commands(avg_arrs_by_car, meta, rundir)
    plot_iv_distance(loaded[0]["frames"], meta, rundir)
    plot_waiting_times(loaded[0]["frames"], all_car_ids, meta, rundir)
    plot_error_cdf(avg_arrs_by_car, meta, rundir)
    plot_lane_timeline(avg_arrs_by_car, meta, rundir)
    plot_emergency_stop_timeline(avg_arrs_by_car, meta, rundir)

    plot_trajectory(loaded, rundir, avg_mode=True)

    all_run_summaries: Dict[str, List[dict]] = {cid: [] for cid in all_car_ids}
    for d in loaded:
        for cid in all_car_ids:
            all_run_summaries[cid].append(
                compute_summary(d["frames"], meta, cid))

    summary_by_car: Dict[str, dict] = {}
    for cid in all_car_ids:
        summ_list = all_run_summaries[cid]
        avg_summ: dict = {}
        for key in summ_list[0]:
            vals = [s[key] for s in summ_list if s[key] is not None]
            avg_summ[key] = round(float(np.mean(vals)), 4) if vals else None
        summary_by_car[cid] = avg_summ

    return {
        "meta": meta,
        "frames": loaded[0]["frames"],
        "summary_by_car": summary_by_car,
        "summary": summary_by_car.get(all_car_ids[0], {}) if all_car_ids else {},
        "track_ground_truth": next(
            (d.get("track_ground_truth") for d in loaded if d.get("track_ground_truth")),
            None),
    }

# ── Entry points ──────────────────────────────────────────────────────────────

def process_file(path: str) -> dict:
    print(f"\nProcessing {path}")
    data = load_json(path)
    meta = data["meta"]
    frames = data["frames"]

    rundir = _results_dir(meta)

    car_ids = _car_ids(frames)
    meta["car_ids"] = car_ids

    if not car_ids:
        print(f"  [warn] No car data found in frames -- skipping charts for {path}")
        return data

    arrs_by_car = {cid: frames_to_arrays(frames, cid) for cid in car_ids}
    summary_by_car = {cid: compute_summary(frames, meta, cid) for cid in car_ids}
    data["summary_by_car"] = summary_by_car
    data["summary"] = summary_by_car.get(car_ids[0], {}) if car_ids else {}

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    plot_lateral_error(arrs_by_car, meta, rundir)
    plot_heading_error(arrs_by_car, meta, rundir)
    plot_obstacle_dist(arrs_by_car, meta, rundir)
    plot_safety_pie(arrs_by_car, meta, rundir)
    plot_safety_pie_car(arrs_by_car, meta, rundir)
    plot_safety_pie_object(arrs_by_car, meta, rundir)
    plot_commands(arrs_by_car, meta, rundir)
    plot_iv_distance(frames, meta, rundir)
    plot_waiting_times(frames, car_ids, meta, rundir)
    plot_error_cdf(arrs_by_car, meta, rundir)
    plot_lane_timeline(arrs_by_car, meta, rundir)
    plot_emergency_stop_timeline(arrs_by_car, meta, rundir)
    plot_trajectory([data], rundir, avg_mode=False)

    return data

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark plotter for experiment JSON files")
    parser.add_argument(
        "files", nargs="*",
        help="Paths to experiment .json files")
    parser.add_argument(
        "-f", "--avg-files", type=int, default=None, metavar="N",
        help=(
            "Average the first N positional files as repeated runs of the "
            "same scenario. The averaged result is treated as one run entry "
            "alongside any remaining files. "
            "Example: -f 5 (with 5 matching .json paths listed)"
        ))
    args = parser.parse_args()

    if not args.files:
        parser.print_help(); sys.exit(0)

    os.makedirs(os.path.join(".", "exp", "results"), exist_ok=True)

    if args.avg_files and len(args.files) >= args.avg_files:
        avg_group = args.files[:args.avg_files]
        remaining = args.files[args.avg_files:]
        avg_run = process_averaged_files(avg_group)
        ind_runs = [process_file(p) for p in remaining]
        runs = [avg_run] + ind_runs
    else:
        runs = [process_file(p) for p in args.files]

    for _r in runs:
        write_summary_table([_r], _results_dir(_r.get("meta", {})))

    if len(runs) > 1:
        multi_dir = _multi_run_dir(runs)
        write_summary_table(runs, multi_dir)
        plot_policy_comparison(runs, multi_dir)
        print(f"  Multi-run outputs -> {multi_dir}")

    print(f"\nAll per-run outputs written to ./exp/results/")

if __name__ == "__main__":
    main()
