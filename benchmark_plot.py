#!/usr/bin/env python3
"""
benchmark_plot.py  –  Analysis & visualisation tool for experiment JSON files.

Input : one or more .json files produced by auto_control.save_log
Output: PNG charts + a Markdown/CSV summary table

Usage
-----
    python benchmark_plot.py experiment_log.json               # single file
    python benchmark_plot.py S1_coop.json S1_noncoop.json      # compare runs
    python benchmark_plot.py *.json --out results              # batch

JSON schema (new nested per-car format)
----------------------------------------
Top-level keys
    meta       : {scenario, policy, car_ids, n_frames, d_col_px, d_safe_px,
                  d_warn_px, saved_at}
    frames     : list of frame dicts
    interaction_zones : {interactions: [{car_id, taus}]}
    summary    : populated here and written back

Per-frame dict
    t          : float   timestamp (s)
    k          : int     frame counter
    distances  : {"1-2": float, ...}   pairwise inter-car gaps (px)
    cars       : {
        "<car_id>": {
            policy        : str,
            pose          : [x_px, y_px, theta_deg],
            lane          : int,
            segment       : str,
            command       : {servo: float, motor: float},
            waiting       : bool,
            lateral_error : float,
            heading_error : float,
            obstacle      : {state: str, distance_px: float|null},
            events        : [str]
        }
    }

Charts produced (per file)
--------------------------
 1. lateral_error_timeseries.png   lateral error over time + mean
 2. heading_error_timeseries.png   heading error over time + mean
 3. obstacle_dist_timeseries.png   obstacle dist + D_SAFE / D_WARN bands
 4. safety_event_pie.png           collision / near-miss / safe proportions
 5. commands_timeseries.png        servo + motor step commands over time
 6. iv_distance_timeseries.png     estimated vs reference inter-vehicle dist
 7. waiting_time_bar.png           per-interaction waiting time τᵢ
 8. error_cdf.png                  empirical CDF of lateral error  (RQ1.2)
Multi-file
 9. policy_comparison_bar.png      side-by-side summary across runs
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Colour palette (Nexus design system) ──────────────────────────────────────
PAL = {
    "cooperative":    "#01696f",
    "non_cooperative":"#964219",
    "egocentric":     "#964219",   # alias kept for backward compat
    "safe":           "#437a22",
    "near_miss":      "#d19900",
    "collision":      "#a12c7b",
    "servo":          "#006494",
    "motor":          "#da7101",
    "lateral":        "#01696f",
    "heading":        "#7a39bb",
    "obs_dist":       "#a13544",
    "d_safe":         "#a12c7b",
    "d_warn":         "#d19900",
    "default":        "#28251d",
}

# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Schema helpers ────────────────────────────────────────────────────────────

def _car_ids(frames: List[dict]) -> List[str]:
    """Return sorted list of car_id strings present across all frames."""
    ids = set()
    for f in frames:
        ids.update(f.get("cars", {}).keys())
    return sorted(ids)


def _car_field(f: dict, car_id: str, *path, default=None):
    """Safely navigate frame → cars → car_id → nested keys."""
    node = f.get("cars", {}).get(str(car_id), {})
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is None:
            return default
    return node


# ── Array extraction (per car_id) ─────────────────────────────────────────────

def frames_to_arrays(frames: List[dict], car_id: str) -> dict:
    """
    Extract numpy arrays from the frame list for one car.

    All benchmark metrics – pose tracking error, obstacle distance, safety
    events, waiting time – are derived from these arrays.
    """
    t0 = frames[0]["t"] if frames else 0.0
    t       = np.array([f["t"] - t0 for f in frames])
    lat     = np.array([_car_field(f, car_id, "lateral_error", default=np.nan)
                        for f in frames])
    hdg     = np.array([_car_field(f, car_id, "heading_error", default=np.nan)
                        for f in frames])
    servo   = np.array([_car_field(f, car_id, "command", "servo", default=np.nan)
                        for f in frames])
    motor   = np.array([_car_field(f, car_id, "command", "motor", default=np.nan)
                        for f in frames])
    obs_dist = np.array([
        _car_field(f, car_id, "obstacle", "distance_px", default=np.nan)
        for f in frames
    ])
    # safety classification: any 'safety_stop' or same-lane emergency event
    def _safety(f):
        evts = _car_field(f, car_id, "events", default=[]) or []
        if "safety_stop" in evts:
            return "collision"
        if "same_lane_emergency" in evts:
            return "collision"
        if "same_lane_slow" in evts:
            return "near_miss"
        return "safe"
    safety  = [_safety(f) for f in frames]
    # waiting flag
    waiting = np.array([int(_car_field(f, car_id, "waiting", default=False) or False)
                        for f in frames])
    # lane per frame
    lane    = np.array([_car_field(f, car_id, "lane", default=1) for f in frames])

    return dict(t=t, lat=lat, hdg=hdg, servo=servo, motor=motor,
                obs_dist=obs_dist, safety=safety, waiting=waiting, lane=lane)


def frames_to_iv_arrays(frames: List[dict]) -> dict:
    """
    Extract inter-vehicle distance arrays from the frame-level 'distances' dict.

    The new schema stores pairwise gaps at frame level (not per-car), e.g.:
        frame["distances"] = {"1-2": 183.4}
    Returns dict keyed by pair string → (t_arr, dist_arr).
    """
    t0 = frames[0]["t"] if frames else 0.0
    pairs: Dict[str, list] = {}
    for f in frames:
        t_rel = f["t"] - t0
        for pair, d in (f.get("distances") or {}).items():
            pairs.setdefault(pair, []).append((t_rel, d))
    return {p: (np.array([x[0] for x in v]),
                np.array([x[1] for x in v]))
            for p, v in pairs.items()}


# ── Summary metrics ───────────────────────────────────────────────────────────

def compute_summary(frames: List[dict], zones: dict, meta: dict,
                    car_id: str) -> dict:
    """
    Compute all evaluation metrics defined in the methodology for one car.

    Metrics
    -------
    mean_lateral_error_px   mean |εₚ|          Pose tracking error
    mean_heading_error_deg  mean |εθ|
    mean_iv_dist_error_px   mean |ε_dij|       Distance estimation error
    n_collision             Safety events
    n_near_miss
    collision_rate, near_miss_rate
    mean_waiting_time_s     Mean waiting time τ̄
    """
    arrs = frames_to_arrays(frames, car_id)

    lat_errs = np.abs(arrs["lat"])
    lat_errs = lat_errs[~np.isnan(lat_errs)]
    hdg_errs = np.abs(arrs["hdg"])
    hdg_errs = hdg_errs[~np.isnan(hdg_errs)]

    # IV distance error: compare estimated distance to D_WARN as reference
    # (true reference is not available without ground truth; kept as None)
    iv_errors = []

    n_col  = arrs["safety"].count("collision")
    n_near = arrs["safety"].count("near_miss")

    n_exp = max(len(zones.get("interactions", [])), 1)
    taus  = [z["taus"] for z in zones.get("interactions", []) if "taus" in z]

    return dict(
        mean_lateral_error_px   = round(float(np.mean(lat_errs)), 4) if len(lat_errs) else None,
        mean_heading_error_deg  = round(float(np.mean(hdg_errs)), 4) if len(hdg_errs) else None,
        mean_iv_dist_error_px   = round(float(np.mean(iv_errors)), 4) if iv_errors else None,
        n_collision             = n_col,
        n_near_miss             = n_near,
        collision_rate          = round(n_col  / n_exp, 4),
        near_miss_rate          = round(n_near / n_exp, 4),
        mean_waiting_time_s     = round(float(np.mean(taus)), 4) if taus else None,
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
    print(f"  chart → {path}")


# ── Per-run charts ────────────────────────────────────────────────────────────

def plot_lateral_error(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 1 – lateral error time-series (all cars overlaid)."""
    fig, ax = _fig()
    for car_id, arrs in arrs_by_car.items():
        lat = np.abs(arrs["lat"])
        ax.plot(arrs["t"], lat, lw=1.2, alpha=0.85, label=f"Car {car_id}")
        ax.axhline(float(np.nanmean(lat)), lw=1.2, ls="--",
                   label=f"Mean car {car_id} ({np.nanmean(lat):.1f} px)")
    ax.set_xlabel("Time s")
    ax.set_ylabel("Lateral error px")
    ax.set_title(f"Pose Tracking – Lateral Error  {meta['scenario']}  {meta['policy']}")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "lateral_error_timeseries.png"))


def plot_heading_error(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 2 – heading error time-series (all cars overlaid)."""
    fig, ax = _fig()
    for car_id, arrs in arrs_by_car.items():
        hdg = np.abs(arrs["hdg"])
        ax.plot(arrs["t"], hdg, lw=1.2, alpha=0.85, label=f"Car {car_id}")
        ax.axhline(float(np.nanmean(hdg)), lw=1.2, ls="--",
                   label=f"Mean car {car_id} ({np.nanmean(hdg):.1f}°)")
    ax.set_xlabel("Time s")
    ax.set_ylabel("Heading error °")
    ax.set_title(f"Pose Tracking – Heading Error  {meta['scenario']}  {meta['policy']}")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "heading_error_timeseries.png"))


def plot_obstacle_dist(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 3 – obstacle distance with D_SAFE / D_WARN reference bands."""
    d_col  = meta.get("d_col_px",  30)
    d_safe = meta.get("d_safe_px", 57)
    d_warn = meta.get("d_warn_px", 115)
    fig, ax = _fig()
    for car_id, arrs in arrs_by_car.items():
        ax.plot(arrs["t"], arrs["obs_dist"], lw=1.2, alpha=0.85,
                color=PAL["obs_dist"], label=f"Obs dist car {car_id}")
    ax.axhline(d_col,  color="#28251d", lw=1.2, ls="-",  label=f"d_col {d_col} px")
    ax.axhline(d_safe, color=PAL["d_safe"], lw=1.5, ls="--", label=f"D_SAFE {d_safe} px")
    ax.axhline(d_warn, color=PAL["d_warn"], lw=1.5, ls=":",  label=f"D_WARN {d_warn} px")
    t_all = np.concatenate([a["t"] for a in arrs_by_car.values()])
    ax.fill_between(t_all, 0, d_col,  alpha=0.12, color=PAL["collision"])
    ax.fill_between(t_all, d_col, d_safe, alpha=0.08, color=PAL["d_safe"])
    ax.fill_between(t_all, d_safe, d_warn, alpha=0.06, color=PAL["d_warn"])
    ax.set_xlabel("Time s")
    ax.set_ylabel("Distance px")
    ax.set_title(f"Obstacle Distance  {meta['scenario']}  {meta['policy']}")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "obstacle_dist_timeseries.png"))


def plot_safety_pie(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 4 – safety event distribution (aggregated over all cars)."""
    all_safety = []
    for arrs in arrs_by_car.values():
        all_safety.extend(arrs["safety"])
    counts  = {k: all_safety.count(k) for k in ("collision", "near_miss", "safe")}
    labels  = [k for k, v in counts.items() if v > 0]
    vals    = [counts[k] for k in labels]
    colors  = [PAL[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    fig.patch.set_facecolor("#f7f6f2")
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors, autopct="%.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor="#f7f6f2", linewidth=1.5))
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(9)
    ax.set_title(f"Safety Event Distribution  {meta['scenario']}  {meta['policy']}",
                 color="#28251d")
    _save(fig, os.path.join(outdir, "safety_event_pie.png"))


def plot_commands(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 5 – quantised servo + motor commands (one subplot per car)."""
    n = len(arrs_by_car)
    fig, axes = plt.subplots(n * 2, 1, figsize=(FIGW, 3 * n), dpi=DPI, sharex=True)
    fig.patch.set_facecolor("#f7f6f2")
    if n * 2 == 2:
        axes = list(axes)   # ensure iterable
    for idx, (car_id, arrs) in enumerate(arrs_by_car.items()):
        ax1 = axes[idx * 2]
        ax2 = axes[idx * 2 + 1]
        for ax in (ax1, ax2):
            ax.set_facecolor("#f9f8f5")
            for sp in ax.spines.values():
                sp.set_edgecolor("#d4d1ca")
            ax.tick_params(colors="#7a7974")
        ax1.step(arrs["t"], arrs["servo"], color=PAL["servo"], lw=1.2, where="post")
        ax1.set_ylabel(f"Servo rad\n(car {car_id})")
        ax1.axhline(0, color="#d4d1ca", lw=0.8)
        ax2.step(arrs["t"], arrs["motor"], color=PAL["motor"], lw=1.2, where="post")
        ax2.set_ylabel(f"Motor\n(car {car_id})")
    axes[-1].set_xlabel("Time s")
    axes[0].set_title(f"Commands – Servo & Motor  {meta['scenario']}  {meta['policy']}",
                      color="#28251d")
    _save(fig, os.path.join(outdir, "commands_timeseries.png"))


def plot_iv_distance(frames: List[dict], meta: dict, outdir: str) -> None:
    """
    Chart 6 – inter-vehicle distance over time.

    Reads frame["distances"] {"1-2": float, ...} directly —
    no per-car nesting needed.
    """
    iv = frames_to_iv_arrays(frames)
    if not iv:
        return
    d_safe = meta.get("d_safe_px", 57)
    fig, ax = _fig()
    for pair, (t_arr, d_arr) in iv.items():
        ax.plot(t_arr, d_arr, lw=1.2, label=f"d({pair})")
    ax.axhline(d_safe, color=PAL["d_safe"], lw=1.2, ls="--", alpha=0.7,
               label=f"D_SAFE {d_safe} px")
    ax.set_xlabel("Time s")
    ax.set_ylabel("Inter-vehicle distance px")
    ax.set_title(f"Inter-Vehicle Distance  {meta['scenario']}  {meta['policy']}")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "iv_distance_timeseries.png"))


def plot_waiting_times(zones: dict, meta: dict, outdir: str) -> None:
    """Chart 7 – per-interaction waiting time τᵢ."""
    interactions = zones.get("interactions", [])
    if not interactions:
        return
    taus = [z["taus"] for z in interactions if "taus" in z]
    if not taus:
        return
    idxs = list(range(1, len(taus) + 1))
    fig, ax = _fig(4)
    pol_color = PAL.get(meta.get("policy", "cooperative"), PAL["default"])
    ax.bar(idxs, taus, color=pol_color, alpha=0.85,
           edgecolor="#f7f6f2", linewidth=1.2)
    mean_t = float(np.mean(taus))
    ax.axhline(mean_t, color="#28251d", lw=1.5, ls="--",
               label=f"Mean {mean_t:.2f} s")
    ax.set_xlabel("Interaction")
    ax.set_ylabel("Waiting time τᵢ s")
    ax.set_title(f"Per-Interaction Waiting Time  {meta['scenario']}  {meta['policy']}")
    ax.set_xticks(idxs)
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "waiting_time_bar.png"))


def plot_error_cdf(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 8 – empirical CDF of lateral error (supports RQ1.2)."""
    fig, ax = _fig()
    for car_id, arrs in arrs_by_car.items():
        errs = np.sort(np.abs(arrs["lat"]))
        errs = errs[~np.isnan(errs)]
        cdf  = np.arange(1, len(errs) + 1) / len(errs)
        ax.plot(errs, cdf, lw=1.8, label=f"Car {car_id}")
    ax.set_xlabel("Lateral error px")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"CDF – Lateral Error  {meta['scenario']}  {meta['policy']}")
    ax.grid(True, alpha=0.3, color="#d4d1ca")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "error_cdf.png"))


def plot_lane_timeline(arrs_by_car: dict, meta: dict, outdir: str) -> None:
    """Chart 10 (new) – lane assignment over time per car."""
    fig, ax = _fig()
    for car_id, arrs in arrs_by_car.items():
        ax.step(arrs["t"], arrs["lane"], lw=1.4, where="post",
                label=f"Car {car_id}")
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Lane 1 (inner)", "Lane 2 (outer)"])
    ax.set_xlabel("Time s")
    ax.set_ylabel("Lane")
    ax.set_title(f"Lane Assignment Over Time  {meta['scenario']}  {meta['policy']}")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(outdir, "lane_timeline.png"))


# ── Multi-run comparison chart ────────────────────────────────────────────────

def plot_policy_comparison(runs: List[dict], outdir: str) -> None:
    """Chart 9 – side-by-side summary metrics: cooperative vs non-cooperative."""
    metrics = [
        ("mean_lateral_error_px",  "Mean lat. error px"),
        ("mean_heading_error_deg", "Mean hdg error °"),
        ("mean_waiting_time_s",    "Mean waiting time s"),
        ("collision_rate",         "Collision rate"),
        ("near_miss_rate",         "Near-miss rate"),
    ]
    # one bar group per (run × car_id)
    labels = []
    for r in runs:
        for cid in sorted(r.get("meta", {}).get("car_ids", ["?"])):
            labels.append(f"{r['meta']['scenario']}\n{r['meta']['policy'][:4].upper()} c{cid}")

    n_met = len(metrics)
    fig, axes = plt.subplots(1, n_met, figsize=(3.5 * n_met, 5), dpi=DPI)
    fig.patch.set_facecolor("#f7f6f2")
    if n_met == 1:
        axes = [axes]

    for ax, (key, ylabel) in zip(axes, metrics):
        vals, colors = [], []
        for r in runs:
            pol = r["meta"].get("policy", "cooperative")
            for cid in sorted(r.get("meta", {}).get("car_ids", ["?"])):
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
        for sp in ax.spines.values():
            sp.set_edgecolor("#d4d1ca")
        ax.tick_params(colors="#7a7974")
        for bar, val in zip(bars, [bar_vals[i] for i in range(len(labels)) if has_data[i]]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(bar_vals + [1e-9]),
                    f"{val:.3g}", ha="center", va="bottom",
                    fontsize=7, color="#28251d")

    fig.suptitle("Policy / Scenario Comparison – Summary Metrics",
                 color="#28251d", fontsize=11)
    _save(fig, os.path.join(outdir, "policy_comparison_bar.png"))


# ── Summary table ─────────────────────────────────────────────────────────────

SUMMARY_COLS = [
    "scenario", "policy", "car_id", "n_frames",
    "mean_lateral_error_px", "mean_heading_error_deg", "mean_iv_dist_error_px",
    "n_collision", "n_near_miss", "collision_rate", "near_miss_rate",
    "mean_waiting_time_s",
]

def write_summary_table(runs: List[dict], outdir: str) -> None:
    rows = []
    for r in runs:
        for cid in sorted(r.get("meta", {}).get("car_ids", ["?"])):
            summ = r.get("summary_by_car", {}).get(str(cid), r.get("summary", {}))
            row  = {**r["meta"], "car_id": cid, **summ}
            rows.append({c: row.get(c) for c in SUMMARY_COLS})

    csv_path = os.path.join(outdir, "summary_table.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(SUMMARY_COLS) + "\n")
        for row in rows:
            fh.write(",".join(str(row[c]) if row[c] is not None else ""
                              for c in SUMMARY_COLS) + "\n")
    print(f"  table → {csv_path}")

    md_path = os.path.join(outdir, "summary_table.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(" | ".join(SUMMARY_COLS) + "\n")
        fh.write(" | ".join(["---"] * len(SUMMARY_COLS)) + "\n")
        for row in rows:
            fh.write(" | ".join(str(row[c]) if row[c] is not None else ""
                                for c in SUMMARY_COLS) + "\n")
    print(f"  table → {md_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def process_file(path: str, outdir: str) -> dict:
    print(f"\nProcessing {path}")
    data   = load_json(path)
    meta   = data["meta"]
    frames = data["frames"]
    zones  = data.get("interaction_zones", {})

    stem   = Path(path).stem
    rundir = os.path.join(outdir, stem)
    os.makedirs(rundir, exist_ok=True)

    # Discover all car IDs present in this log
    car_ids = _car_ids(frames)
    meta.setdefault("car_ids", car_ids)

    # Build per-car arrays and summaries
    arrs_by_car    = {cid: frames_to_arrays(frames, cid) for cid in car_ids}
    summary_by_car = {cid: compute_summary(frames, zones, meta, cid)
                      for cid in car_ids}
    data["summary_by_car"] = summary_by_car
    # Backward-compat: keep flat "summary" as the first/only car's summary
    data["summary"] = summary_by_car.get(car_ids[0], {}) if car_ids else {}

    # Write updated JSON with summaries filled in
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    # Charts
    plot_lateral_error(arrs_by_car, meta, rundir)
    plot_heading_error(arrs_by_car, meta, rundir)
    plot_obstacle_dist(arrs_by_car, meta, rundir)
    plot_safety_pie(arrs_by_car, meta, rundir)
    plot_commands(arrs_by_car, meta, rundir)
    plot_iv_distance(frames, meta, rundir)
    plot_waiting_times(zones, meta, rundir)
    plot_error_cdf(arrs_by_car, meta, rundir)
    plot_lane_timeline(arrs_by_car, meta, rundir)

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark plotter for experiment JSON files")
    parser.add_argument("files", nargs="+",
                        help="Paths to experiment .json files")
    parser.add_argument("--out", default="benchmark_results",
                        help="Output directory (default: benchmark_results)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = [process_file(p, args.out) for p in args.files]
    write_summary_table(runs, args.out)
    if len(runs) > 1:
        plot_policy_comparison(runs, args.out)
    print(f"\nAll outputs written to {args.out}")


if __name__ == "__main__":
    main()
