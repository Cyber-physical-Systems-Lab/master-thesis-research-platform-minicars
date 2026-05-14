#!/usr/bin/env python3
"""
benchmark_plot.py — Analysis & visualisation tool for experiment JSON files
============================================================================
Input :  one or more *.json files produced by logger.ExperimentLogger.save()
Output:  PNG charts  +  a Markdown / CSV summary table

Usage
-----
    python benchmark_plot.py experiment_output.json           # single file
    python benchmark_plot.py S1_coop.json S1_egoc.json        # compare runs
    python benchmark_plot.py *.json --out results/            # batch

Charts produced (per file)
--------------------------
  1.  lateral_error_timeseries.png  — lateral tracking error over time
  2.  heading_error_timeseries.png  — heading error over time
  3.  obstacle_dist_timeseries.png  — obstacle distance + D_SAFE / D_WARN bands
  4.  safety_event_pie.png          — collision / near-miss / safe proportions
  5.  commands_timeseries.png       — servo & motor commands over time
  6.  iv_distance_timeseries.png    — inter-vehicle distance (if present)
  7.  waiting_time_bar.png          — per-interaction waiting times
  8.  error_cdf.png                 — CDF of lateral error (good for RQ1.2)

  (Multi-file)
  9.  policy_comparison_bar.png     — side-by-side summary metrics across runs
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Colour palette ─────────────────────────────────────────────────────────────
PAL = {
    "cooperative":   "#01696f",   # Hydra Teal
    "egocentric":    "#964219",   # Terra Brown
    "safe":          "#437a22",   # Gridania Green
    "near_miss":     "#d19900",   # Altana Gold
    "collision":     "#a12c7b",   # Jenova Maroon
    "servo":         "#006494",   # Limsa Blue
    "motor":         "#da7101",   # Costa Orange
    "lateral":       "#01696f",
    "heading":       "#7a39bb",   # Kuja Purple
    "obs_dist":      "#a13544",   # Rosa Red
    "d_safe":        "#a12c7b",
    "d_warn":        "#d19900",
    "default":       "#28251d",
}


# ── Load helpers ───────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

def frames_to_arrays(frames: List[dict]):
    """Extract numpy arrays from frame list for vectorised plotting."""
    t        = np.array([f["t_s"] for f in frames])
    t        = t - t[0]                              # relative time
    lat      = np.array([f["lateral_error_px"] for f in frames])
    hdg      = np.array([f["heading_error_deg"] for f in frames])
    servo    = np.array([f["servo_cmd"] for f in frames])
    motor    = np.array([f["motor_cmd"] for f in frames])
    obs_dist = np.array([f["obs_dist_px"] if f["obs_dist_px"] is not None else np.nan
                         for f in frames])
    safety   = [f["safety_event"] for f in frames]
    in_zone  = np.array([int(f.get("in_interaction_zone", False)) for f in frames])
    return dict(t=t, lat=lat, hdg=hdg, servo=servo, motor=motor,
                obs_dist=obs_dist, safety=safety, in_zone=in_zone)


# ── Chart helpers ──────────────────────────────────────────────────────────────

FIG_W, FIG_H = 10, 4
DPI = 150

def _fig(h=FIG_H):
    fig, ax = plt.subplots(figsize=(FIG_W, h), dpi=DPI)
    fig.patch.set_facecolor("#f7f6f2")
    ax.set_facecolor("#f9f8f5")
    for spine in ax.spines.values():
        spine.set_edgecolor("#d4d1ca")
    ax.tick_params(colors="#7a7974")
    ax.title.set_color("#28251d")
    ax.xaxis.label.set_color("#7a7974")
    ax.yaxis.label.set_color("#7a7974")
    return fig, ax

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path}")


# ── Per-run charts ─────────────────────────────────────────────────────────────

def plot_lateral_error(arrs, meta, out_dir):
    fig, ax = _fig()
    ax.plot(arrs["t"], np.abs(arrs["lat"]), color=PAL["lateral"], lw=1.2, alpha=0.85)
    ax.axhline(np.mean(np.abs(arrs["lat"])), color=PAL["lateral"],
               lw=1.5, ls="--", label=f'Mean = {np.mean(np.abs(arrs["lat"])):.1f} px')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Lateral error| (px)")
    ax.set_title(f"Pose Tracking — Lateral Error  [{meta['scenario']} / {meta['policy']}]")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(out_dir, "lateral_error_timeseries.png"))

def plot_heading_error(arrs, meta, out_dir):
    fig, ax = _fig()
    ax.plot(arrs["t"], np.abs(arrs["hdg"]), color=PAL["heading"], lw=1.2, alpha=0.85)
    ax.axhline(np.mean(np.abs(arrs["hdg"])), color=PAL["heading"],
               lw=1.5, ls="--", label=f'Mean = {np.mean(np.abs(arrs["hdg"])):.1f}°')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Heading error| (°)")
    ax.set_title(f"Pose Tracking — Heading Error  [{meta['scenario']} / {meta['policy']}]")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(out_dir, "heading_error_timeseries.png"))

def plot_obstacle_dist(arrs, meta, out_dir):
    d_safe = meta.get("d_safe_px", 40)
    d_warn = meta.get("d_warn_px", 90)
    fig, ax = _fig()
    ax.plot(arrs["t"], arrs["obs_dist"], color=PAL["obs_dist"], lw=1.2, alpha=0.85,
            label="Obstacle distance")
    ax.axhline(d_safe, color=PAL["d_safe"], lw=1.5, ls="--", label=f"d_safe = {d_safe} px")
    ax.axhline(d_warn, color=PAL["d_warn"], lw=1.5, ls=":", label=f"d_warn = {d_warn} px")
    ax.fill_between(arrs["t"], 0, d_safe, alpha=0.08, color=PAL["d_safe"])
    ax.fill_between(arrs["t"], d_safe, d_warn, alpha=0.06, color=PAL["d_warn"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (px)")
    ax.set_title(f"Obstacle Distance  [{meta['scenario']} / {meta['policy']}]")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(out_dir, "obstacle_dist_timeseries.png"))

def plot_safety_pie(arrs, meta, out_dir):
    counts = {k: arrs["safety"].count(k) for k in ("collision", "near_miss", "safe")}
    labels = [k for k, v in counts.items() if v > 0]
    vals   = [counts[k] for k in labels]
    colors = [PAL[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    fig.patch.set_facecolor("#f7f6f2")
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor="#f7f6f2", linewidth=1.5))
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(9)
    ax.set_title(f"Safety Event Distribution  [{meta['scenario']} / {meta['policy']}]",
                 color="#28251d")
    _save(fig, os.path.join(out_dir, "safety_event_pie.png"))

def plot_commands(arrs, meta, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 6), dpi=DPI, sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("#f9f8f5")
        for sp in ax.spines.values(): sp.set_edgecolor("#d4d1ca")
        ax.tick_params(colors="#7a7974")
    fig.patch.set_facecolor("#f7f6f2")
    ax1.step(arrs["t"], arrs["servo"], color=PAL["servo"], lw=1.2, where="post")
    ax1.set_ylabel("Servo δ (rad)")
    ax1.axhline(0, color="#d4d1ca", lw=0.8)
    ax2.step(arrs["t"], arrs["motor"], color=PAL["motor"], lw=1.2, where="post")
    ax2.set_ylabel("Motor v")
    ax2.set_xlabel("Time (s)")
    ax1.set_title(f"Commands (servo & motor)  [{meta['scenario']} / {meta['policy']}]",
                  color="#28251d")
    _save(fig, os.path.join(out_dir, "commands_timeseries.png"))

def plot_iv_distance(frames, meta, out_dir):
    """Inter-vehicle distance — estimated vs reference (if available)."""
    t_vals, est_vals, ref_vals = [], [], []
    for f in frames:
        for iv in f.get("iv_distances", []):
            t_vals.append(f["t_s"])
            est_vals.append(iv.get("iv_dist_est_px"))
            ref_vals.append(f.get("iv_dist_ref_px"))
    if not t_vals:
        return
    t0 = t_vals[0]
    t_rel = [t - t0 for t in t_vals]
    fig, ax = _fig()
    ax.plot(t_rel, est_vals, color=PAL["cooperative"], lw=1.2, label="Estimated d_ij")
    has_ref = any(r is not None for r in ref_vals)
    if has_ref:
        ax.plot(t_rel, ref_vals, color="#28251d", lw=1.2, ls="--", label="Reference d*_ij")
    ax.axhline(meta.get("d_safe_px", 40), color=PAL["d_safe"], lw=1.2, ls="--",
               alpha=0.7, label=f"d_safe = {meta.get('d_safe_px',40)} px")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Inter-vehicle distance (px)")
    ax.set_title(f"Inter-Vehicle Distance  [{meta['scenario']} / {meta['policy']}]")
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(out_dir, "iv_distance_timeseries.png"))

def plot_waiting_times(zones, meta, out_dir):
    interactions = zones.get("interactions", [])
    if not interactions:
        return
    taus = [r["tau_s"] for r in interactions]
    idxs = list(range(1, len(taus) + 1))
    fig, ax = _fig(h=4)
    ax.bar(idxs, taus, color=PAL[meta.get("policy", "cooperative")], alpha=0.85,
           edgecolor="#f7f6f2", linewidth=1.2)
    mean_t = sum(taus) / len(taus)
    ax.axhline(mean_t, color="#28251d", lw=1.5, ls="--",
               label=f"Mean τ̄ = {mean_t:.2f} s")
    ax.set_xlabel("Interaction #")
    ax.set_ylabel("Waiting time τ_i (s)")
    ax.set_title(f"Per-Interaction Waiting Time  [{meta['scenario']} / {meta['policy']}]")
    ax.set_xticks(idxs)
    ax.legend(framealpha=0.7)
    _save(fig, os.path.join(out_dir, "waiting_time_bar.png"))

def plot_error_cdf(arrs, meta, out_dir):
    """Empirical CDF of |lateral error| — useful for RQ1.2 height comparison."""
    errs = np.sort(np.abs(arrs["lat"]))
    cdf  = np.arange(1, len(errs) + 1) / len(errs)
    fig, ax = _fig()
    ax.plot(errs, cdf, color=PAL["lateral"], lw=1.8)
    ax.set_xlabel("|Lateral error| (px)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"CDF — Lateral Error  [{meta['scenario']} / {meta['policy']}]")
    ax.grid(True, alpha=0.3, color="#d4d1ca")
    _save(fig, os.path.join(out_dir, "error_cdf.png"))


# ── Multi-run comparison chart ─────────────────────────────────────────────────

def plot_policy_comparison(runs: List[dict], out_dir: str):
    """Side-by-side bar chart of summary metrics across all loaded runs."""
    metrics = [
        ("mean_lateral_error_px",  "Mean lat. error (px)",    False),
        ("mean_heading_error_deg", "Mean hdg error (°)",       False),
        ("mean_waiting_time_s",    "Mean waiting time (s)",    False),
        ("collision_rate",         "Collision rate",           False),
        ("near_miss_rate",         "Near-miss rate",           False),
    ]
    labels = [f"{r['meta']['scenario']}\n{r['meta']['policy'][:4].upper()}" for r in runs]
    n_met = len(metrics)
    x = np.arange(len(runs))
    bar_w = 0.65

    fig, axes = plt.subplots(1, n_met, figsize=(3.5 * n_met, 5), dpi=DPI)
    fig.patch.set_facecolor("#f7f6f2")
    if n_met == 1:
        axes = [axes]

    for ax, (key, ylabel, _) in zip(axes, metrics):
        vals = [r["summary"].get(key) for r in runs]
        colors = [PAL.get(r["meta"]["policy"], PAL["default"]) for r in runs]
        has_data = [v is not None for v in vals]
        bar_vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x[has_data], [bar_vals[i] for i in range(len(runs)) if has_data[i]],
                      width=bar_w,
                      color=[colors[i] for i in range(len(runs)) if has_data[i]],
                      alpha=0.88, edgecolor="#f7f6f2", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8, color="#7a7974")
        ax.set_facecolor("#f9f8f5")
        for sp in ax.spines.values(): sp.set_edgecolor("#d4d1ca")
        ax.tick_params(colors="#7a7974")
        for bar, val in zip(bars, [bar_vals[i] for i in range(len(runs)) if has_data[i]]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(bar_vals + [1e-9]),
                    f"{val:.3g}", ha="center", va="bottom", fontsize=7, color="#28251d")

    fig.suptitle("Policy / Scenario Comparison — Summary Metrics", color="#28251d", fontsize=11)
    _save(fig, os.path.join(out_dir, "policy_comparison_bar.png"))


# ── Summary table (CSV + Markdown) ────────────────────────────────────────────

SUMMARY_COLS = [
    "scenario", "policy", "car_id", "n_frames",
    "mean_lateral_error_px", "mean_heading_error_deg",
    "mean_iv_dist_error_px", "n_collision", "n_near_miss",
    "collision_rate", "near_miss_rate", "mean_waiting_time_s",
]

def write_summary_table(runs: List[dict], out_dir: str):
    rows = []
    for r in runs:
        row = {**r["meta"], **r["summary"]}
        rows.append({c: row.get(c, None) for c in SUMMARY_COLS})

    # CSV
    csv_path = os.path.join(out_dir, "summary_table.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(SUMMARY_COLS) + "\n")
        for row in rows:
            fh.write(",".join(str(row[c]) if row[c] is not None else "" for c in SUMMARY_COLS) + "\n")
    print(f"  [table] {csv_path}")

    # Markdown
    md_path = os.path.join(out_dir, "summary_table.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(SUMMARY_COLS) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(SUMMARY_COLS)) + " |\n")
        for row in rows:
            fh.write("| " + " | ".join(
                str(row[c]) if row[c] is not None else "—"
                for c in SUMMARY_COLS) + " |\n")
    print(f"  [table] {md_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def process_file(path: str, out_dir: str):
    print(f"\n── Processing: {path}")
    data   = load_json(path)
    meta   = data["meta"]
    frames = data["frames"]
    zones  = data.get("interaction_zones", {})
    summ   = data.get("summary", {})

    stem    = Path(path).stem
    run_dir = os.path.join(out_dir, stem)
    os.makedirs(run_dir, exist_ok=True)

    arrs = frames_to_arrays(frames)

    plot_lateral_error(arrs, meta, run_dir)
    plot_heading_error(arrs, meta, run_dir)
    plot_obstacle_dist(arrs, meta, run_dir)
    plot_safety_pie(arrs, meta, run_dir)
    plot_commands(arrs, meta, run_dir)
    plot_iv_distance(frames, meta, run_dir)
    plot_waiting_times(zones, meta, run_dir)
    plot_error_cdf(arrs, meta, run_dir)

    return data


def main():
    parser = argparse.ArgumentParser(description="Benchmark plotter for experiment JSON files")
    parser.add_argument("files", nargs="+", help="Path(s) to experiment .json file(s)")
    parser.add_argument("--out", default="benchmark_results",
                        help="Output directory (default: benchmark_results/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = [process_file(p, args.out) for p in args.files]

    write_summary_table(runs, args.out)
    if len(runs) > 1:
        plot_policy_comparison(runs, args.out)

    print(f"\n✓  All outputs written to: {args.out}/")


if __name__ == "__main__":
    main()
