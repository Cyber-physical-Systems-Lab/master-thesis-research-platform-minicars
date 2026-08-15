"""benchmark_plot.py -- charts + summary tables for auto_control.py experiment logs.

Usage:
    python benchmark_plot.py log.json                  single file
    python benchmark_plot.py a.json b.json              compare runs
    python benchmark_plot.py *.json                     batch
    python benchmark_plot.py -f 5 r1.json r2.json ...    average first N as one run

Reads the JSON schema written by auto_control.save_log (meta/frames/track_ground_truth).
D_COL/D_WARN/D_SAFE below mirror auto_control.py's real thresholds (not stored in meta).
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style ────────────────────────────────────────────────────────────────
PAL = {
    "cooperative": "#01696f", "non_cooperative": "#964219", "egocentric": "#964219",
    "safe": "#437a22", "decision": "#5591c7", "near_miss": "#d19900",
    "collision": "#a12c7b", "no_collision": "#437a22",
    "servo": "#006494", "motor": "#da7101", "lateral": "#01696f", "heading": "#7a39bb",
    "obs_dist": "#a13544", "d_safe": "#a12c7b", "d_warn": "#d19900", "default": "#28251d",
}
RUN_COLOURS = ["#01696f", "#964219", "#437a22", "#006494", "#7a39bb", "#a12c7b", "#d19900", "#a13544"]
BG, PANEL, GRID, MUTED, DARK = "#f7f6f2", "#f9f8f5", "#d4d1ca", "#7a7974", "#28251d"

D_COL, D_WARN, D_SAFE = 25, 57, 115  # px, matches auto_control.py's real constants
FIGW, FIGH, DPI = 10, 4, 150
EVENT_STATES = {"decision", "near_miss", "collision", "hold_gap", "small_gap"}
EMERGENCY_TAGS = {"emergency_stop_straddle", "emergency_stop_both_lanes_blocked"}
SLOWDOWN_TAGS = {"safety_stop", "obstacle_near_slowdown", "both_lanes_near_slowdown"}

# ── I/O helpers ──────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

def results_dir(meta: dict) -> str:
    dfov = f"{meta.get('dfov')}dFOV" if meta.get("dfov") else ""
    folder = f"{meta.get('scenario', 'S?')}-{dfov}-{meta.get('calibration', 'non-calib')}-{meta.get('policy', 'unknown')}"
    path = os.path.join(".", "exp", "results", folder)
    os.makedirs(path, exist_ok=True)
    return path

def multi_run_dir(runs: list) -> str:
    if not runs:
        path = os.path.join(".", "exp", "results", "multi")
        os.makedirs(path, exist_ok=True)
        return path
    metas = [r.get("meta", {}) for r in runs]
    dfov = f"{metas[0].get('dfov')}dFOV" if metas[0].get("dfov") else ""
    policies = sorted({m.get("policy", "unknown") for m in metas})
    pol_tag = policies[0] if len(policies) == 1 else "multi"
    folder = f"{metas[0].get('scenario', 'S?')}-{dfov}-{metas[0].get('calibration', 'non-calib')}-{pol_tag}"
    path = os.path.join(".", "exp", "results", folder)
    os.makedirs(path, exist_ok=True)
    return path

# ── Schema helpers ───────────────────────────────────────────────────────

def car_ids(frames: List[dict]) -> List[str]:
    """Sorted car_id list, derived from frames (not stored in meta)."""
    ids = set()
    for f in frames:
        ids.update(f.get("cars", {}).keys())
    return sorted(ids)

def car_field(f: dict, cid: str, *path, default=None):
    """Navigate frame -> cars -> cid -> nested keys, returning default on miss."""
    node = f.get("cars", {}).get(str(cid), {})
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is None:
            return default
    return node

def safety_state(f: dict, cid: str, source: Optional[str] = None) -> str:
    """4-way safety bucket for one frame.

    source=None -> combined car+object tags, "decision" folds into "near_miss" (legacy view).
    source="car"/"object" -> restricted to one interaction source, keeps "decision" distinct.
    The "object" source also checks minicar_events for the emergency-stop/slowdown fallback tags.
    """
    ev = car_field(f, cid, "events", default={}) or {}
    raw = ev.get("safety_events", []) or []
    minicar = set(ev.get("minicar_events", []) or [])

    if source is None:
        tags = list(raw.get("car", []) or []) + list(raw.get("object", []) or []) if isinstance(raw, dict) else raw
        check_minicar = True
    else:
        tags = (raw.get(source, []) or []) if isinstance(raw, dict) else (raw if source == "car" else [])
        check_minicar = source == "object"

    if "collision" in tags or (check_minicar and minicar & EMERGENCY_TAGS):
        return "collision"
    if "near_miss" in tags:
        return "near_miss"
    if "decision" in tags:
        return "near_miss" if source is None else "decision"
    if check_minicar and minicar & SLOWDOWN_TAGS:
        return "near_miss"
    return "safe"

def frames_to_arrays(frames: List[dict], cid: str) -> dict:
    """Per-car numpy arrays for all benchmark metrics (pose error, obstacle dist, safety, waiting)."""
    t0 = frames[0]["t"] if frames else 0.0
    return dict(
        t=np.array([f["t"] - t0 for f in frames]),
        lat=np.array([car_field(f, cid, "lateral_error", default=np.nan) for f in frames]),
        hdg=np.array([car_field(f, cid, "heading_error", default=np.nan) for f in frames]),
        servo=np.array([car_field(f, cid, "command", "servo", default=np.nan) for f in frames]),
        motor=np.array([car_field(f, cid, "command", "motor", default=np.nan) for f in frames]),
        obs_dist=np.array([car_field(f, cid, "obstacle", "distance_px", default=np.nan) for f in frames]),
        safety=[safety_state(f, cid) for f in frames],
        safety_car=[safety_state(f, cid, "car") for f in frames],
        safety_object=[safety_state(f, cid, "object") for f in frames],
        waiting=np.array([int(car_field(f, cid, "waiting", default=False) or False) for f in frames]),
        lane=np.array([car_field(f, cid, "lane", default=1) for f in frames]),
        emstop=np.array([int(car_field(f, cid, "emergency_stop", default=False) or False) for f in frames]),
    )

def frames_to_iv_arrays(frames: List[dict]) -> dict:
    """Per-pair car-to-car distance arrays from each frame's 'distances' dict.

    Returns {pair: {tarr, distarr, same_lane, lane_a, lane_b, seg_delta_arr,
    state_list, event_t, event_v}}. Handles both the legacy scalar format and
    the rich-dict format ({"euclidean_px", "same_lane", "lane_a", "lane_b",
    "seg_delta", "interaction_state"}).
    """
    t0 = frames[0]["t"] if frames else 0.0
    pairs: Dict[str, dict] = {}
    for f in frames:
        t_rel = f["t"] - t0
        for pair, raw in (f.get("distances") or {}).items():
            if isinstance(raw, dict):
                d, sl, la, lb = raw.get("euclidean_px", float("nan")), raw.get("same_lane"), raw.get("lane_a"), raw.get("lane_b")
                sdelta, state = raw.get("seg_delta"), raw.get("interaction_state")
            else:
                d, sl, la, lb, sdelta, state = float(raw), None, None, None, None, None
            rec = pairs.setdefault(pair, {"t": [], "d": [], "sl": [], "la": None, "lb": None, "sd": [], "state": []})
            rec["t"].append(t_rel); rec["d"].append(d); rec["sl"].append(sl)
            rec["sd"].append(sdelta); rec["state"].append(state)
            rec["la"] = rec["la"] if rec["la"] is not None else la
            rec["lb"] = rec["lb"] if rec["lb"] is not None else lb

    out = {}
    for p, rec in pairs.items():
        sl_vals = [v for v in rec["sl"] if v is not None]
        tarr, distarr = np.array(rec["t"]), np.array(rec["d"])
        states = rec["state"]
        event_idx = [i for i, s in enumerate(states) if s in EVENT_STATES]
        out[p] = {
            "tarr": tarr, "distarr": distarr,
            "same_lane": (sum(sl_vals) / len(sl_vals)) >= 0.5 if sl_vals else None,
            "lane_a": rec["la"], "lane_b": rec["lb"],
            "seg_delta_arr": np.array([v if v is not None else float("nan") for v in rec["sd"]], dtype=float),
            "state_list": states,
            "event_t": [tarr[i] for i in event_idx], "event_v": [distarr[i] for i in event_idx],
        }
    return out

def waiting_durations(frames: List[dict], cid: str) -> List[float]:
    """Durations (s) of contiguous True-runs in the per-frame 'waiting' flag."""
    arrs = frames_to_arrays(frames, cid)
    w, t = arrs["waiting"].astype(bool), arrs["t"]
    durations, in_run, start_t = [], False, None
    for i in range(len(w)):
        if w[i] and not in_run:
            in_run, start_t = True, t[i]
        elif not w[i] and in_run:
            durations.append(float(t[i] - start_t)); in_run = False
    if in_run and start_t is not None and len(t):
        durations.append(float(t[-1] - start_t))
    return durations

# ── Summary metrics ──────────────────────────────────────────────────────

def compute_summary(frames: List[dict], meta: dict, cid: str) -> dict:
    arrs = frames_to_arrays(frames, cid)
    lat_errs = np.abs(arrs["lat"]); lat_errs = lat_errs[~np.isnan(lat_errs)]
    hdg_errs = np.abs(arrs["hdg"]); hdg_errs = hdg_errs[~np.isnan(hdg_errs)]
    px_per_cm = meta.get("px_per_cm") or None

    gap_px = [d for pair, rec in frames_to_iv_arrays(frames).items()
              if str(cid) in pair.split("-") and rec.get("same_lane")
              for d, s in zip(rec["distarr"], rec["state_list"])
              if s in ("hold_gap", "small_gap") and not np.isnan(d)]
    mean_gap_cm = (round(np.mean(gap_px) / px_per_cm, 4) if px_per_cm else round(float(np.mean(gap_px)), 4)) if gap_px else None

    n_col, n_near = arrs["safety"].count("collision"), arrs["safety"].count("near_miss")
    n_col_car, n_near_car, n_dec_car = (arrs["safety_car"].count(k) for k in ("collision", "near_miss", "decision"))
    n_col_obj, n_near_obj, n_dec_obj = (arrs["safety_object"].count(k) for k in ("collision", "near_miss", "decision"))
    taus = waiting_durations(frames, cid)
    n_frames = max(len(frames), 1)
    n_emstop = int(np.sum(arrs["emstop"]))
    obs_valid = arrs["obs_dist"][~np.isnan(arrs["obs_dist"])]

    return dict(
        mean_lateral_error_px=round(float(np.mean(lat_errs)), 4) if len(lat_errs) else None,
        mean_heading_error_deg=round(float(np.mean(hdg_errs)), 4) if len(hdg_errs) else None,
        mean_gap_cm=mean_gap_cm,
        mean_obs_dist_cm=(round(np.mean(obs_valid) / px_per_cm, 4) if px_per_cm else round(float(np.mean(obs_valid)), 4)) if len(obs_valid) else None,
        n_collision=n_col, n_near_miss=n_near,
        n_collision_car=n_col_car, n_near_miss_car=n_near_car, n_decision_car=n_dec_car,
        n_collision_object=n_col_obj, n_near_miss_object=n_near_obj, n_decision_object=n_dec_obj,
        n_emergency_stop=n_emstop,
        collision_rate=round(n_col / n_frames, 4), near_miss_rate=round(n_near / n_frames, 4),
        collision_rate_car=round(n_col_car / n_frames, 4), near_miss_rate_car=round(n_near_car / n_frames, 4),
        collision_rate_object=round(n_col_obj / n_frames, 4), near_miss_rate_object=round(n_near_obj / n_frames, 4),
        emergency_stop_rate=round(n_emstop / n_frames, 4),
        mean_waiting_time_s=round(float(np.mean(taus)), 4) if taus else None,
    )

# ── Chart helpers ────────────────────────────────────────────────────────

def new_fig(h: float = FIGH):
    fig, ax = plt.subplots(figsize=(FIGW, h), dpi=DPI)
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(DARK); ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    return fig, ax

def save_fig(fig, path: str) -> None:
    """Saves PNG + vector PDF sibling."""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    pdf_path = os.path.splitext(path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart -> {path} / {pdf_path}")

def title_suffix(meta: dict) -> str:
    return f"{meta.get('scenario', '')} {meta.get('policy', '')}"

# ── Per-run charts ───────────────────────────────────────────────────────

def plot_lateral_error(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    fig, ax = new_fig()
    for cid, arrs in arrs_by_car.items():
        lat = np.abs(arrs["lat"])
        ax.plot(arrs["t"], lat, lw=1.2, alpha=0.85, label=f"Car {cid}")
        ax.axhline(float(np.nanmean(lat)), lw=1.2, ls="--", label=f"Mean car {cid} ({np.nanmean(lat):.1f} px)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Lateral error [px]")
    ax.set_title(f"Pose Tracking - Lateral Error {title_suffix(meta)}")
    ax.legend(framealpha=0.7)
    save_fig(fig, os.path.join(outdir, "lateral_error_timeseries.png"))

def plot_heading_error(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    fig, ax = new_fig()
    for cid, arrs in arrs_by_car.items():
        hdg = np.abs(arrs["hdg"])
        ax.plot(arrs["t"], hdg, lw=1.2, alpha=0.85, label=f"Car {cid}")
        ax.axhline(float(np.nanmean(hdg)), lw=1.2, ls="--", label=f"Mean car {cid} ({np.nanmean(hdg):.1f} deg)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Heading error [deg]")
    ax.set_title(f"Pose Tracking - Heading Error {title_suffix(meta)}")
    ax.legend(framealpha=0.7)
    save_fig(fig, os.path.join(outdir, "heading_error_timeseries.png"))

def plot_car_object_dist(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Car-to-object distance with D_COL/D_WARN/D_SAFE bands and safety_events.object markers."""
    has_obs = any(not np.all(np.isnan(a["obs_dist"])) for a in arrs_by_car.values())
    fig, ax = new_fig()
    out_path = os.path.join(outdir, "car_object_dist_timeseries.png")
    if not has_obs:
        ax.text(0.5, 0.5, "No obstacle data in this scenario\n(obstacles not present - e.g. S1)",
                transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=10)
        ax.set_title(f"Car-to-Object Distance {title_suffix(meta)}")
        save_fig(fig, out_path)
        return

    for cid, arrs in arrs_by_car.items():
        ax.plot(arrs["t"], arrs["obs_dist"], lw=1.2, alpha=0.85, color=PAL["obs_dist"], label=f"Obs dist car {cid}")
        obs_d, sobj = arrs["obs_dist"], arrs.get("safety_object", [])
        for state, marker, z in (("decision", "o", 3), ("near_miss", "^", 4), ("collision", "X", 5)):
            idxs = [i for i, s in enumerate(sobj) if s == state and i < len(obs_d) and not np.isnan(obs_d[i])]
            if idxs:
                ax.scatter(arrs["t"][idxs], obs_d[idxs], marker=marker, s=26, color=PAL[state],
                           edgecolor=DARK, linewidth=0.4, zorder=z, alpha=0.9, label=f"{state} event car {cid}")

    ax.axhline(D_COL, color=DARK, lw=1.2, label=f"D_COL {D_COL} px")
    ax.axhline(D_WARN, color=PAL["d_warn"], lw=1.5, ls="--", label=f"D_WARN {D_WARN} px")
    ax.axhline(D_SAFE, color=PAL["d_safe"], lw=1.5, ls=":", label=f"D_SAFE {D_SAFE} px")
    t_all = np.concatenate([a["t"] for a in arrs_by_car.values()])
    if len(t_all):
        ax.fill_between(t_all, 0, D_COL, alpha=0.12, color=PAL["collision"])
        ax.fill_between(t_all, D_COL, D_WARN, alpha=0.08, color=PAL["d_warn"])
        ax.fill_between(t_all, D_WARN, D_SAFE, alpha=0.06, color=PAL["d_safe"])
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Distance [px]")
    ax.set_title(f"Car-to-Object Distance {title_suffix(meta)}")
    ax.legend(framealpha=0.7, fontsize=7, loc="upper right")
    save_fig(fig, out_path)

def _safety_pie(all_safety: list, title: str, fname: str, meta: dict, outdir: str) -> None:
    counts = {k: all_safety.count(k) for k in ("collision", "near_miss", "decision", "safe")}
    labels = [k for k, v in counts.items() if v > 0]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    out_path = os.path.join(outdir, fname)
    if not labels:
        ax.text(0.5, 0.5, "No safety events recorded\n(all frames: safe)",
                transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=10)
    else:
        wedges, texts, autotexts = ax.pie(
            [counts[k] for k in labels], labels=labels, colors=[PAL[k] for k in labels],
            autopct="%.1f%%", startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor=BG, linewidth=1.5))
        for at in autotexts:
            at.set_color("white"); at.set_fontsize(9)
    ax.set_title(f"{title} {title_suffix(meta)}", color=DARK)
    save_fig(fig, out_path)

def plot_safety_pie(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Combined (legacy) safety event distribution across all cars/sources."""
    all_safety = [s for a in arrs_by_car.values() for s in a["safety"]]
    _safety_pie(all_safety, "Safety Event Distribution (combined)", "safety_event_pie.png", meta, outdir)

def plot_safety_pie_car(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    all_safety = [s for a in arrs_by_car.values() for s in a.get("safety_car", [])]
    _safety_pie(all_safety, "Safety Event Distribution - Car-to-Car", "safety_event_pie_car.png", meta, outdir)

def plot_safety_pie_object(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    all_safety = [s for a in arrs_by_car.values() for s in a.get("safety_object", [])]
    _safety_pie(all_safety, "Safety Event Distribution - Car-to-Object", "safety_event_pie_object.png", meta, outdir)

def plot_commands(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Servo + motor step commands, one subplot pair per car."""
    n = len(arrs_by_car)
    if n == 0:
        return
    fig, axes = plt.subplots(n * 2, 1, figsize=(FIGW, 3 * n), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    axes = list(np.array([axes, axes] if n * 2 == 1 else axes).flatten())
    for idx, (cid, arrs) in enumerate(arrs_by_car.items()):
        ax1, ax2 = axes[idx * 2], axes[idx * 2 + 1]
        for ax in (ax1, ax2):
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
            ax.tick_params(colors=MUTED)
        ax1.step(arrs["t"], arrs["servo"], color=PAL["servo"], lw=1.2, where="post")
        ax1.set_ylabel(f"Servo [rad]\n(car {cid})"); ax1.axhline(0, color=GRID, lw=0.8)
        ax2.step(arrs["t"], arrs["motor"], color=PAL["motor"], lw=1.2, where="post")
        ax2.set_ylabel(f"Motor [-]\n(car {cid})")
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(f"Commands - Servo & Motor {title_suffix(meta)}", color=DARK)
    save_fig(fig, os.path.join(outdir, "commands_timeseries.png"))

SAME_LANE_STYLE = {"hold_gap": ("decision", "o"), "small_gap": ("collision", "X")}
CROSS_LANE_STYLE = {"decision": ("decision", "o"), "near_miss": ("near_miss", "^"), "collision": ("collision", "X")}

def plot_car_car_dist(frames: List[dict], meta: dict, outdir: str) -> None:
    """Car-to-car pairwise distance: same-lane panel (gap-following scale) + cross-lane panel."""
    iv = frames_to_iv_arrays(frames)
    if not iv:
        return
    same_pairs = {p: v for p, v in iv.items() if v["same_lane"] is not False}
    cross_pairs = {p: v for p, v in iv.items() if v["same_lane"] is False}
    n_panels = bool(same_pairs) + bool(cross_pairs)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(FIGW, FIGH * n_panels), dpi=DPI)
    fig.patch.set_facecolor(BG)
    axes = [axes] if n_panels == 1 else list(axes)
    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED)

    panel, suffix = 0, title_suffix(meta)
    if same_pairs:
        ax = axes[panel]; panel += 1
        for pair, v in same_pairs.items():
            ax.plot(v["tarr"], v["distarr"], lw=1.4, label=f"d({pair}) L{v['lane_a']}->L{v['lane_b']}")
            for i, s in enumerate(v.get("state_list", [])):
                if s in SAME_LANE_STYLE:
                    pal_key, marker = SAME_LANE_STYLE[s]
                    ax.scatter(v["tarr"][i], v["distarr"][i], marker=marker, s=22, color=PAL[pal_key],
                               edgecolor=DARK, linewidth=0.3, alpha=0.85, zorder=4)
        all_t = np.concatenate([v["tarr"] for v in same_pairs.values()])
        ax.fill_between(all_t, 0, D_COL, alpha=0.10, color=PAL["collision"])
        ax.fill_between(all_t, D_COL, D_WARN, alpha=0.08, color=PAL["d_warn"])
        ax.fill_between(all_t, D_WARN, D_SAFE, alpha=0.06, color=PAL["d_safe"])
        ax.axhline(D_WARN, color=PAL["d_warn"], lw=1.4, ls="--", label=f"D_WARN {D_WARN} [px]")
        ax.axhline(D_SAFE, color=PAL["d_safe"], lw=1.2, ls=":", label=f"D_SAFE {D_SAFE} [px]")
        if any(not np.all(np.isnan(v["seg_delta_arr"])) for v in same_pairs.values()):
            ax2 = ax.twinx(); ax2.set_facecolor("none")
            for pair, v in same_pairs.items():
                if not np.all(np.isnan(v["seg_delta_arr"])):
                    ax2.plot(v["tarr"], v["seg_delta_arr"], lw=1.0, ls="--", alpha=0.55,
                             color="#c8c6c0", label=f"dseg({pair})")
            ax2.set_ylabel("Segment-index gap (samples)", color="#c8c6c0", fontsize=8)
            ax2.tick_params(colors="#c8c6c0")
            ax2.legend(loc="upper right", fontsize=7, framealpha=0.5)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Euclidean distance [px]")
        ax.set_title(f"Car-to-Car Distance - Same-Lane {suffix}")
        ax.legend(loc="upper left", framealpha=0.7, fontsize=8)

    if cross_pairs:
        ax = axes[panel]
        for pair, v in cross_pairs.items():
            ax.plot(v["tarr"], v["distarr"], lw=1.4, ls="-.", label=f"d({pair}) L{v['lane_a']}<->L{v['lane_b']}")
            for i, s in enumerate(v.get("state_list", [])):
                if s in CROSS_LANE_STYLE:
                    pal_key, marker = CROSS_LANE_STYLE[s]
                    ax.scatter(v["tarr"][i], v["distarr"][i], marker=marker, s=22, color=PAL[pal_key],
                               edgecolor=DARK, linewidth=0.3, alpha=0.85, zorder=4)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Euclidean distance [px]")
        ax.set_title(f"Car-to-Car Distance - Cross-Lane {suffix}")
        ax.legend(loc="upper left", framealpha=0.7, fontsize=8)

    save_fig(fig, os.path.join(outdir, "car_car_dist_timeseries.png"))

def plot_waiting_times(frames: List[dict], cids: List[str], meta: dict, outdir: str) -> None:
    """Per-car mean waiting time (+/-1 std), derived from the 'waiting' flag."""
    taus_by_car = {cid: t for cid in cids if (t := waiting_durations(frames, cid))}
    if not taus_by_car:
        return
    labels = list(taus_by_car.keys())
    means = [float(np.mean(taus_by_car[c])) for c in labels]
    stds = [float(np.std(taus_by_car[c])) if len(taus_by_car[c]) > 1 else 0.0 for c in labels]
    counts = [len(taus_by_car[c]) for c in labels]

    fig, ax = new_fig(4)
    colour = PAL.get(meta.get("policy", "cooperative"), PAL["default"])
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colour, alpha=0.85, edgecolor=BG, linewidth=1.2)
    overall_mean = float(np.mean([v for vals in taus_by_car.values() for v in vals]))
    ax.axhline(overall_mean, color=DARK, lw=1.5, ls="--", label=f"Overall mean {overall_mean:.2f} s")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.03,
                f"n={n}", ha="center", va="bottom", fontsize=8, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels([f"Car {c}" for c in labels])
    ax.set_xlabel("Car ID"); ax.set_ylabel("Waiting time [s] (mean +/- std)")
    ax.set_title(f"Per-Car Waiting Time {title_suffix(meta)}")
    ax.legend(framealpha=0.7)
    save_fig(fig, os.path.join(outdir, "waiting_time_bar.png"))

def plot_error_cdf(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    fig, ax = new_fig()
    plotted = False
    for cid, arrs in arrs_by_car.items():
        errs = np.sort(np.abs(arrs["lat"]))
        errs = errs[~np.isnan(errs)]
        if len(errs) == 0:
            continue
        ax.plot(errs, np.arange(1, len(errs) + 1) / len(errs), lw=1.8, label=f"Car {cid}")
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No lateral error data available", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=10)
    ax.set_xlabel("Lateral error [px]"); ax.set_ylabel("Cumulative probability")
    ax.set_title(f"CDF - Lateral Error {title_suffix(meta)}")
    ax.grid(True, alpha=0.3, color=GRID); ax.legend(framealpha=0.7)
    save_fig(fig, os.path.join(outdir, "error_cdf.png"))

def plot_lane_timeline(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    fig, ax = new_fig()
    for cid, arrs in arrs_by_car.items():
        lane = np.where(np.isnan(arrs["lane"].astype(float)), 1, arrs["lane"])
        ax.step(arrs["t"], lane, lw=1.4, where="post", label=f"Car {cid}")
    ax.set_yticks([1, 2]); ax.set_yticklabels(["Lane 1 (inner)", "Lane 2 (outer)"])
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Lane")
    ax.set_title(f"Lane Assignment Over Time {title_suffix(meta)}")
    ax.legend(framealpha=0.7)
    save_fig(fig, os.path.join(outdir, "lane_timeline.png"))

def plot_emergency_stop_timeline(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Skipped when no emergency stops occurred (e.g. nominal S1 runs)."""
    if not any(bool(np.any(a.get("emstop", np.zeros(1)) > 0)) for a in arrs_by_car.values()):
        return
    fig, ax = new_fig(3)
    for cid, arrs in arrs_by_car.items():
        em = arrs.get("emstop", np.zeros_like(arrs["t"]))
        ax.step(arrs["t"], em, lw=1.4, where="post", label=f"Car {cid}")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "EMERGENCY"])
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Emergency stop active")
    ax.set_title(f"Emergency Stop Active Frames {title_suffix(meta)}")
    ax.legend(framealpha=0.7)
    save_fig(fig, os.path.join(outdir, "emergency_stop_timeline.png"))

# ── Trajectory coverage ──────────────────────────────────────────────────

def extract_car_positions(frames: List[dict], cid: str, px_per_cm: Optional[float] = None, unit: str = "px") -> List[tuple]:
    scale = str(unit).lower() == "cm" and px_per_cm not in (None, 0)
    pts = []
    for f in frames:
        p = car_field(f, cid, "pose", default=None)
        if p and len(p) >= 2:
            x, y = float(p[0]), float(p[1])
            if scale:
                x, y = x / float(px_per_cm), y / float(px_per_cm)
            pts.append((x, y))
    return pts

GT_STYLE = {"lane1_ref": ("#6daa45", 2.0, "Lane 1 ideal path"), "lane2_ref": ("#5591c7", 2.0, "Lane 2 ideal path")}

def plot_trajectory(runs_data: List[dict], outdir: str, avg_mode: bool = False) -> None:
    """Track ground-truth curves + car position scatter, merged across runs in avg_mode."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=DPI)
    fig.patch.set_facecolor(BG); ax.set_facecolor("#171614")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED); ax.set_aspect("equal"); ax.invert_yaxis()

    gt_run = next((r for r in runs_data if r.get("track_ground_truth")), None)
    gt = gt_run.get("track_ground_truth") if gt_run else None
    gt_meta = gt_run.get("meta", {}) if gt_run else {}
    gt_scale = str(gt_meta.get("unit", "px")).lower() == "cm" and gt_meta.get("px_per_cm") not in (None, 0)

    if gt:
        for key, (col, lw, lbl) in GT_STYLE.items():
            pts = gt.get(key, [])
            if pts:
                arr = np.array(pts, dtype=float)
                if gt_scale:
                    arr = arr / float(gt_meta["px_per_cm"])
                arr_closed = np.vstack([arr, arr[[0]]])
                ax.plot(arr_closed[:, 0], arr_closed[:, 1], color=col, lw=lw, alpha=0.75, label=lbl)
    else:
        ax.text(0.5, 0.5, "No track_ground_truth in log\n(run at least one frame first)",
                transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)

    all_cids = sorted(set(cid for r in runs_data for f in r.get("frames", []) for cid in f.get("cars", {}).keys()))
    for i, cid in enumerate(all_cids):
        colour = RUN_COLOURS[i % len(RUN_COLOURS)]
        xs, ys = [], []
        for r in runs_data:
            rmeta = r.get("meta", {})
            for x, y in extract_car_positions(r.get("frames", []), cid, rmeta.get("px_per_cm"), rmeta.get("unit", "px")):
                xs.append(x); ys.append(y)
        if xs:
            ax.scatter(xs, ys, s=2, alpha=0.35, color=colour, label=f"Car {cid} positions", rasterized=True)

    meta0 = runs_data[0].get("meta", {}) if runs_data else {}
    suffix = f" ({len(runs_data)} runs averaged)" if avg_mode and len(runs_data) > 1 else ""
    dfov_lbl = f" dFOV={meta0.get('dfov')}deg" if meta0.get("dfov") else ""
    ax.set_title(f"Trajectory Coverage {title_suffix(meta0)}{dfov_lbl}{suffix}", color=DARK)
    unit = meta0.get("unit", "px")
    ax.set_xlabel(f"x [{unit}]", color=MUTED); ax.set_ylabel(f"y [{unit}]", color=MUTED)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6, facecolor="#1c1b19", labelcolor="white")
    save_fig(fig, os.path.join(outdir, "trajectory_coverage.png"))

# ── Multi-run comparison ─────────────────────────────────────────────────

COMPARISON_METRICS = [
    ("mean_lateral_error_px", "Mean lat. error px"), ("mean_heading_error_deg", "Mean hdg error deg"),
    ("mean_waiting_time_s", "Mean waiting time s"), ("collision_rate", "Collision rate"),
    ("near_miss_rate", "Near-miss rate"),
]

def plot_policy_comparison(runs: List[dict], outdir: str) -> None:
    labels = [f"{r['meta'].get('scenario', '')}\n{r['meta'].get('policy', '?')[:4].upper()} c{cid}"
              for r in runs for cid in r.get("meta", {}).get("car_ids", ["?"])]
    fig, axes = plt.subplots(1, len(COMPARISON_METRICS), figsize=(3.5 * len(COMPARISON_METRICS), 5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    axes = [axes] if len(COMPARISON_METRICS) == 1 else axes

    for ax, (key, ylabel) in zip(axes, COMPARISON_METRICS):
        vals, colors = [], []
        for r in runs:
            pol = r["meta"].get("policy", "cooperative")
            for cid in r.get("meta", {}).get("car_ids", ["?"]):
                summ = r.get("summary_by_car", {}).get(str(cid), r.get("summary", {}))
                vals.append(summ.get(key)); colors.append(PAL.get(pol, PAL["default"]))
        has_data = [v is not None for v in vals]
        bar_vals = [v if v is not None else 0 for v in vals]
        x = np.arange(len(labels))
        shown_x = [i for i in range(len(labels)) if has_data[i]]
        bars = ax.bar(shown_x, [bar_vals[i] for i in shown_x], width=0.65,
                       color=[colors[i] for i in shown_x], alpha=0.88, edgecolor=BG, linewidth=1.2)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED); ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED)
        for bar, i in zip(bars, shown_x):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(bar_vals + [1e-9]),
                    f"{bar_vals[i]:.3g}", ha="center", va="bottom", fontsize=7, color=DARK)

    fig.suptitle("Policy / Scenario Comparison - Summary Metrics", color=DARK, fontsize=11)
    save_fig(fig, os.path.join(outdir, "policy_comparison_bar.png"))

# ── Summary table ────────────────────────────────────────────────────────

SUMMARY_COLS = [
    "scenario", "policy", "calibration", "car_id", "n_frames",
    "mean_lateral_error_px", "mean_heading_error_deg", "mean_gap_cm", "mean_obs_dist_cm",
    "n_collision", "n_near_miss", "n_emergency_stop",
    "collision_rate", "near_miss_rate", "emergency_stop_rate",
    "n_collision_car", "n_near_miss_car", "n_decision_car", "collision_rate_car", "near_miss_rate_car",
    "n_collision_object", "n_near_miss_object", "n_decision_object", "collision_rate_object", "near_miss_rate_object",
    "mean_waiting_time_s",
]
RATE_COLS = {c for c in SUMMARY_COLS if c.startswith(("n_", "collision_rate", "near_miss_rate", "emergency_stop_rate"))}

def _fmt(v, col: str) -> str:
    if v is None:
        return "0" if col in RATE_COLS else "-"
    return str(v)

def write_summary_table(runs: List[dict], outdir: str) -> None:
    rows = [{c: {**r["meta"], "car_id": cid, **r.get("summary_by_car", {}).get(str(cid), r.get("summary", {}))}.get(c)
             for c in SUMMARY_COLS}
            for r in runs for cid in r.get("meta", {}).get("car_ids", ["?"])]

    csv_path = os.path.join(outdir, "summary_table.csv")
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

# ── Multi-file averaging ─────────────────────────────────────────────────

def resample(arr: np.ndarray, n: int = 1000) -> np.ndarray:
    if len(arr) == 0:
        return np.full(n, np.nan)
    return np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr)

NUMERIC_KEYS = ["lat", "hdg", "servo", "motor", "obs_dist", "waiting", "lane", "emstop"]
N_RESAMPLE = 1000

def process_averaged_files(paths: List[str]) -> dict:
    """Averages N JSON files (resampled to a normalised time axis) and charts the result."""
    print(f"\nAveraging {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")
    loaded = [load_json(p) for p in paths]

    meta = dict(loaded[0]["meta"])
    meta["n_frames"] = int(round(np.mean([len(d["frames"]) for d in loaded])))
    meta["averaged_runs"] = len(paths)
    meta["source_files"] = [Path(p).name for p in paths]
    all_cids = sorted(set(cid for d in loaded for f in d["frames"] for cid in f.get("cars", {}).keys()))
    meta["car_ids"] = all_cids

    def majority_vote(run_arrs, step_i, key):
        votes = []
        for ra in run_arrs:
            series = ra.get(key, [])
            n = len(series)
            votes.append(series[min(int(round(step_i * (n - 1) / (N_RESAMPLE - 1))), n - 1)] if n else "safe")
        return Counter(votes).most_common(1)[0][0]

    avg_arrs_by_car = {}
    for cid in all_cids:
        run_arrs = [frames_to_arrays(d["frames"], cid) for d in loaded]
        averaged = {k: np.nanmean(np.stack([resample(ra[k].astype(float), N_RESAMPLE) for ra in run_arrs], axis=0), axis=0)
                    for k in NUMERIC_KEYS}
        mean_dur = float(np.mean([ra["t"][-1] if len(ra["t"]) else 1.0 for ra in run_arrs]))
        averaged["t"] = np.linspace(0, mean_dur, N_RESAMPLE)
        for key in ("safety", "safety_car", "safety_object"):
            averaged[key] = [majority_vote(run_arrs, i, key) for i in range(N_RESAMPLE)]
        avg_arrs_by_car[cid] = averaged

    rundir = results_dir(meta)
    plot_lateral_error(avg_arrs_by_car, meta, rundir)
    plot_heading_error(avg_arrs_by_car, meta, rundir)
    plot_car_object_dist(avg_arrs_by_car, meta, rundir)
    plot_safety_pie(avg_arrs_by_car, meta, rundir)
    plot_safety_pie_car(avg_arrs_by_car, meta, rundir)
    plot_safety_pie_object(avg_arrs_by_car, meta, rundir)
    plot_commands(avg_arrs_by_car, meta, rundir)
    plot_car_car_dist(loaded[0]["frames"], meta, rundir)
    plot_waiting_times(loaded[0]["frames"], all_cids, meta, rundir)
    plot_error_cdf(avg_arrs_by_car, meta, rundir)
    plot_lane_timeline(avg_arrs_by_car, meta, rundir)
    plot_emergency_stop_timeline(avg_arrs_by_car, meta, rundir)
    plot_trajectory(loaded, rundir, avg_mode=True)

    summary_by_car = {}
    for cid in all_cids:
        summ_list = [compute_summary(d["frames"], meta, cid) for d in loaded]
        summary_by_car[cid] = {
            key: round(float(np.mean(vals)), 4) if (vals := [s[key] for s in summ_list if s[key] is not None]) else None
            for key in summ_list[0]
        }

    return {
        "meta": meta, "frames": loaded[0]["frames"], "summary_by_car": summary_by_car,
        "summary": summary_by_car.get(all_cids[0], {}) if all_cids else {},
        "track_ground_truth": next((d.get("track_ground_truth") for d in loaded if d.get("track_ground_truth")), None),
    }

# ── Entry points ─────────────────────────────────────────────────────────

def process_file(path: str) -> dict:
    print(f"\nProcessing {path}")
    data = load_json(path)
    meta, frames = data["meta"], data["frames"]
    rundir = results_dir(meta)
    cids = car_ids(frames)
    meta["car_ids"] = cids

    if not cids:
        print(f"  [warn] No car data found in frames -- skipping charts for {path}")
        return data

    arrs_by_car = {cid: frames_to_arrays(frames, cid) for cid in cids}
    summary_by_car = {cid: compute_summary(frames, meta, cid) for cid in cids}
    data["summary_by_car"] = summary_by_car
    data["summary"] = summary_by_car.get(cids[0], {})

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    plot_lateral_error(arrs_by_car, meta, rundir)
    plot_heading_error(arrs_by_car, meta, rundir)
    plot_car_object_dist(arrs_by_car, meta, rundir)
    plot_safety_pie(arrs_by_car, meta, rundir)
    plot_safety_pie_car(arrs_by_car, meta, rundir)
    plot_safety_pie_object(arrs_by_car, meta, rundir)
    plot_commands(arrs_by_car, meta, rundir)
    plot_car_car_dist(frames, meta, rundir)
    plot_waiting_times(frames, cids, meta, rundir)
    plot_error_cdf(arrs_by_car, meta, rundir)
    plot_lane_timeline(arrs_by_car, meta, rundir)
    plot_emergency_stop_timeline(arrs_by_car, meta, rundir)
    plot_trajectory([data], rundir, avg_mode=False)
    return data

def main():
    parser = argparse.ArgumentParser(description="Benchmark plotter for experiment JSON files")
    parser.add_argument("files", nargs="*", help="Paths to experiment .json files")
    parser.add_argument("-f", "--avg-files", type=int, default=None, metavar="N",
                         help="Average the first N files as repeated runs of the same scenario")
    args = parser.parse_args()
    if not args.files:
        parser.print_help(); sys.exit(0)

    os.makedirs(os.path.join(".", "exp", "results"), exist_ok=True)

    if args.avg_files and len(args.files) >= args.avg_files:
        avg_run = process_averaged_files(args.files[:args.avg_files])
        runs = [avg_run] + [process_file(p) for p in args.files[args.avg_files:]]
    else:
        runs = [process_file(p) for p in args.files]

    for r in runs:
        write_summary_table([r], results_dir(r.get("meta", {})))

    if len(runs) > 1:
        multi_dir = multi_run_dir(runs)
        write_summary_table(runs, multi_dir)
        plot_policy_comparison(runs, multi_dir)
        print(f"  Multi-run outputs -> {multi_dir}")

    print("\nAll per-run outputs written to ./exp/results/")

if __name__ == "__main__":
    main()
