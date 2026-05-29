"""
auto_control.py
===============
Centralised vision-based controller for the minicar testbed.

Architecture (sense → plan → act)
──────────────────────────────────
Sensing : Overhead USB camera + ArUco marker detection.
Three marker dictionaries are used in parallel:
• 4×4_50 – car identification (front + rear markers)
• 5×5_50 – track boundary (inner / middle / outer rings)
• 6×6_250 – static obstacles
Planning : Pure-pursuit path tracking on a stadium-fitted (pill-shape) curve,
combined with rule-based distance-threshold coordination.
Acting : Quantised servo (steering) and motor (speed) commands sent
as UDP packets to each minicar's Raspberry Pi.

Entry point
───────────
run(frame, car_id) → (servo, motor, annotated_frame)
Called once per camera frame from the main loop.

Experiment logging
──────────────────
Built-in lightweight logger — no external script required.
Each frame appends one entry to an in-memory list; call save_log()
at shutdown to write a single JSON file for offline analysis.

Log schema (per entry)
──────────────────────
{
  "t":       float,            # wall-clock timestamp (s)
  "k":       int,              # frame counter
  "car_id":  int,
  "policy":  str,
  "pose":    [x_px, y_px, theta_deg],
  "lane":    int,
  "segment": str,              # "sharp_curve" | "light_curve" | "straight"
  "command": {"servo": float, "motor": float},
  "waiting": bool,
  "lateral_error":  float,     # px, signed
  "heading_error":  float,     # deg, signed
  "obstacle": {
      "state":       str,      # "clear" | "near" | "blocking"
      "distance_px": float
  },
  "distances": {"<a>-<b>": float},   # all pairwise inter-car px distances
  "events":    [str]                 # e.g. ["lane_change", "safety_stop"]
}
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from cv2 import aruco

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Tracking / speed thresholds ───────────────────────────────────────────────
SPEED_THRESHOLD = 15
STATUS_UPDATE_INTERVAL = 0.1
SCALING_FACTOR = 57.5
COMMAND_INTERVAL = 0.40

# ── ArUco dictionaries ────────────────────────────────────────────────────────
CAR_DICT = aruco.DICT_4X4_50
FRONT_MARKER_ID = 49
TRACK_DICT = aruco.DICT_5X5_50
OBSTACLE_DICT = aruco.DICT_6X6_250

# ── Track marker groupings (by ArUco ID) ─────────────────────────────────────
INNER_SET  = [0, 1, 2, 3]
MIDDLE_SET = [4, 5, 6, 7]
OUTER_SET  = [8, 9, 10, 11]

STADIUM_SAMPLES = 270  # total sample points on the fitted stadium curve

# ── Obstacle geometry ─────────────────────────────────────────────────────────
OBSTACLE_RADIUS           = 45
OBSTACLE_TRACK_HALF_WIDTH = 58
OBSTACLE_LOOKAHEAD        = 120

# ── Speed levels ──────────────────────────────────────────────────────────────
MAX_SPEED    = 0.60
CRUISE_SPEED = 0.45
SLOW_SPEED   = 0.35
STOP_SPEED   = 0.00

# ── Servo limits (radians, sent as normalised floats) ─────────────────────────
MAX_SERVO       = 0.50   # maximum steering angle magnitude
MAX_MID_SERVO   = 0.35
MED_SERVO       = 0.22   # medium steering step
MED_SMALL_SERVO = 0.12
SMALL_SERVO     = 0.05   # small steering step

# ── Discrete output sets ──────────────────────────────────────────────────────
SERVO_STEPS = [-MAX_SERVO, -MAX_MID_SERVO, -MED_SERVO, -MED_SMALL_SERVO, -SMALL_SERVO,
               0.0,
               MAX_SERVO, MAX_MID_SERVO, MED_SERVO, MED_SMALL_SERVO, SMALL_SERVO]
"""Allowed quantised servo values. Negative = steer right, positive = steer left.
Continuous pure-pursuit output is snapped to the nearest step."""

MOTOR_STEPS = [STOP_SPEED, SLOW_SPEED, CRUISE_SPEED, MAX_SPEED]
"""Allowed quantised motor values. Continuous speed is snapped to nearest step."""

# ── Pure-pursuit (bicycle model) parameters ───────────────────────────────────
WHEELBASE_PX    = 120          # rear-axle to front-axle distance in pixels
DELTA_MAX       = MAX_SERVO    # 0.50 → maps to ±45°

K_DD            = 150          # lookahead gain (pixels) — tune this single value
                               # larger = smoother but slower response
                               # smaller = tighter tracking but more oscillation

K_BOUNDARY_PUSH = 0.012        # rad/px — corrective push when outside outer boundary

# Servo mapping:  servo = delta_rad × (2/π)
#   ±π/4 rad (±45°) → ±0.5 servo  |  0 rad → 0.0 servo
#   Scale = MAX_SERVO / (π/4) = 0.5 / 0.785 = 2/π ≈ 0.6366
_RAD_TO_SERVO = 0.5 / (math.pi / 4.0)  # = 2/π ≈ 0.6366

# ── Lane-change / coordination ────────────────────────────────────────────────
LANE_CHANGE_HOLD      = 2.5
D_SAFE                = 57
D_WARN                = 115
COOP_MIN_GAP          = 290
COOP_MAX_CLOSING_SPEED = 30

# ── Driving policies ──────────────────────────────────────────────────────────
DRIVING_POLICY: dict = {}
DEFAULT_POLICY = "cooperative"

# ── Visualisation colours (BGR) ───────────────────────────────────────────────
LANE1_FILL_COLOR  = (0, 160, 60)
LANE2_FILL_COLOR  = (0, 100, 180)
INNER_LINE_COLOR  = (0, 230, 120)
MIDDLE_LINE_COLOR = (255, 255, 0)
OUTER_LINE_COLOR  = (80, 200, 255)
REF_LINE_COLOR    = (200, 200, 200)
# ── Lane classification ───────────────────────────────────────────────────────
# Signed distance from the MIDDLE curve: negative = toward outer ring (lane 2).
# LANE_HALF_WIDTH is auto-calibrated from marker geometry in detect_lanes();
# this constant is the minimum floor in case detection is marginal.
LANE_HALF_WIDTH  = 20   # px — tune to half the actual lane width on your track
LANE_HYSTERESIS  = 10   # px — dead-band to prevent chattering at the lane boundary

# ── Experiment metadata (edit before each run) ────────────────────────────────
LOG_SCENARIO = "S1"
LOG_POLICY   = "cooperative"

# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE
# ══════════════════════════════════════════════════════════════════════════════

tracker:             dict = {}
obstacle_positions:  list = []
lane_state:          dict = {}
last_command_time:   dict = {}
coop_slowdown_until: dict = {}
_pp_waiting:         dict = {}
# _winding_fixed:      dict = {}   # key: (car_id, curve_name) -> bool
_TRACK_FORWARD = None   # will be set once

# ── Marker persistence (occlusion recovery) ──────────────────────────────────
# Track / obstacle markers: keyed by ArUco ID → (corner_array, timestamp)
_last_track_markers: dict = {}   # {id: (corner_array, t)}
_last_obs_positions: dict = {}   # {id: ((x,y), t)}

# Car marker persistence: per car_id
_car_marker_cache: dict = {}
# Structure per car_id:
#   'front_corner':         last known corner array for the front marker
#   'rear_corner':          last known corner array for the rear marker
#   'front_time':           wall-clock time of last front detection
#   'rear_time':            wall-clock time of last rear detection
#   'using_front_fallback': True when rear is occluded and front drives control

# Max seconds a cached marker stays valid. None = keep forever (fully static track).
_MARKER_MAX_AGE: float = 2.5

# ── Lightweight logger state ──────────────────────────────────────────────────
_log_entries:  list = []   # one dict per frame; flushed to JSON at shutdown
_frame_cars_data: dict = {}   # {frame_k: {car_id: per-car dict}}
_cycle_counter: int = 0

# ══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def _build_log_entry(
    t: float,
    k: int,
    cars_data: dict,   # {car_id: {pose, lane, segment, servo, motor, waiting,
                       #           lateral_error, heading_error, policy,
                       #           obstacle_info, events}}
    cars: dict,        # raw identify_cars() output for pairwise distances
) -> dict:
    """
    One log entry per camera frame.

    Schema
    ------
    {
      "t": float,
      "k": int,
      "distances": {"1-2": float, ...},   # pairwise inter-car gaps (px)
      "cars": {
        "<car_id>": {
          "policy":        str,
          "pose":          [x_px, y_px, theta_deg],
          "lane":          int,
          "segment":       str,
          "command":       {"servo": float, "motor": float},
          "waiting":       bool,
          "lateral_error": float,
          "heading_error": float,
          "obstacle":      {"state": str, "distance_px": float | null},
          "events":        [str]
        },
        ...
      }
    }
    """
    # Pairwise distances (frame-level — same for all cars)
    ids = sorted(cars.keys())
    distances = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pa = np.array(cars[a]["midpoint"], dtype=float)
            pb = np.array(cars[b]["midpoint"], dtype=float)
            distances[f"{a}-{b}"] = round(float(np.linalg.norm(pa - pb)), 2)

    # Per-car nested block
    cars_block = {}
    for cid, cd in cars_data.items():
        obs_d = cd["obstacle_info"]["distance"]
        cars_block[str(cid)] = {
            "policy":        cd["policy"],
            "pose":          [round(cd["pose"][0], 2),
                              round(cd["pose"][1], 2),
                              round(cd["pose"][2], 2)],
            "lane":          cd["lane"],
            "segment":       cd["segment"],
            "command":       {"servo": round(cd["servo"], 3),
                              "motor": round(cd["motor"], 3)},
            "waiting":       cd["waiting"],
            "lateral_error": round(cd["lateral_error"], 2),
            "heading_error": round(cd["heading_error"], 2),
            "obstacle":      {"state":       cd["obstacle_info"]["state"],
                              "distance_px": round(obs_d, 2)
                                             if obs_d < float("inf") else None},
            "events":        cd["events"],
        }

    return {"t": round(t, 5), "k": k, "distances": distances, "cars": cars_block}


def save_log(path: str = "experiment_log.json") -> None:
    """
    Write all accumulated log entries to a JSON file.

    The file is structured as:
    {
      "meta":    {"scenario": str, "policy": str, "saved_at": str},
      "frames":  [ <entry>, ... ]
    }

    Call once at shutdown (e.g. in the finally block of the main loop).
    The resulting file can be loaded directly into benchmark_plot.py or
    any script via: data = json.load(open("experiment_log.json")).
    """
    payload = {
        "meta": {
            "scenario":  LOG_SCENARIO,
            "policy":    LOG_POLICY,
            "saved_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_frames":  len(_log_entries),
        },
        "frames": _log_entries,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[log] {len(_log_entries)} frames saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def marker_center(corner) -> tuple:
    """Return the integer pixel centroid of an ArUco marker corner array.

    Parameters
    ----------
    corner : np.ndarray, shape (1, 4, 2)
        Corner array as returned by cv2.aruco.ArucoDetector.detectMarkers().

    Returns
    -------
    (x, y) : tuple[int, int]
    """
    return tuple(np.mean(corner[0], axis=0).astype(int))

def midpoint(p1: tuple, p2: tuple) -> tuple:
    """Return the integer pixel midpoint between two (x, y) points.

    Used to compute the vehicle reference position from the front and rear
    marker centres.
    """
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def wrap_angle_deg(a: float) -> float:
    """Wrap an angle in degrees to the half-open range (-180, 180].

    Used when computing heading error so that, e.g., the difference between
    350° and 10° is correctly reported as 20° rather than -340°.
    """
    return (a + 180.0) % 360.0 - 180.0

def heading_to_angle(heading_vec: tuple) -> float:
    """Convert a 2-D pixel heading vector to a compass angle in [0, 360).

    The heading vector is (front_x − rear_x, front_y − rear_y). The minus
    sign on the y-component converts from image coordinates (y down) to
    standard mathematical coordinates (y up) before calling atan2.
    """
    return float(np.degrees(np.arctan2(-heading_vec[1], heading_vec[0])) % 360)

def dominant_state(state_dict: dict) -> str:
    """Return the key with the highest value in a state membership dict.

    Used to convert a {'blocking': 0.8, 'near': 0.2, 'clear': 0.0} style
    dict (produced by the obstacle distance classifier) into a single label
    string such as 'blocking'.

    Returns '-' for an empty dict.
    """
    if not state_dict:
        return "-"
    return max(state_dict.items(), key=lambda kv: kv[1])[0]


# ══════════════════════════════════════════════════════════════════════════════
# TRACK GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def _sort_stadium_markers(pts: np.ndarray) -> np.ndarray:
    """Sort 4 corner points into consistent CCW angular order around their centroid.

    Guarantees that _fit_stadium_params() always receives points in the same
    spatial order regardless of which ArUco ID was physically placed where,
    eliminating the axis-vector ambiguity that causes curve winding to flip.
    """
    centre = pts.mean(axis=0)
    # Negate y to work in image coords (y-down), sort CW
    angles = np.arctan2(-(pts[:,1] - centre[1]), pts[:,0] - centre[0])
    return pts[np.argsort(angles)[::-1]]  # descending = CW in image space

def _fit_stadium_params(marker_positions: list):
    """Compute exact stadium parameters from four tangent-corner markers.

    The four ArUco markers sit at the exact tangent junctions where each
    straight section meets its semicircle::

        m0 ──── straight ──── m1
         )                    (     ← semicircles
        m3 ──── straight ──── m2

    |m0−m3| = diameter = 2·r.  The midpoints of (m0,m3) and (m1,m2) are the
    two semicircle centres; half their separation is the straight half-length a.

    Falls back to SVD estimate when fewer than 4 points are visible.

    Returns
    -------
    (cx, cy, a, r, angle)
    """
    pts = np.array(marker_positions, dtype=float)
    if len(pts) >= 4:
        pts = _sort_stadium_markers(pts[:4])  # ensure consistent spatial ordering
        left_centre  = (pts[0] + pts[3]) / 2.0

    if len(pts) >= 4:
        left_centre  = (pts[0] + pts[3]) / 2.0
        right_centre = (pts[1] + pts[2]) / 2.0
        cx       = float((left_centre[0] + right_centre[0]) / 2.0)
        cy       = float((left_centre[1] + right_centre[1]) / 2.0)
        axis_vec = right_centre - left_centre
        angle    = float(np.arctan2(axis_vec[1], axis_vec[0]))
        a        = float(np.linalg.norm(axis_vec) / 2.0)
        r_left   = float(np.linalg.norm(pts[0] - pts[3]) / 2.0)
        r_right  = float(np.linalg.norm(pts[1] - pts[2]) / 2.0)
        r        = max((r_left + r_right) / 2.0, 1.0)
    else:
        cx = float(pts[:, 0].mean()); cy = float(pts[:, 1].mean())
        centred    = pts - np.array([cx, cy])
        _, _, Vt   = np.linalg.svd(centred, full_matrices=False)
        angle      = float(np.arctan2(Vt[0, 1], Vt[0, 0]))
        proj_long  = centred @ Vt[0]; proj_short = centred @ Vt[1]
        r = max(float(np.mean(np.abs(proj_short))), 1.0)
        a = max(float(np.max(np.abs(proj_long))) - r, 0.0)
    return cx, cy, a, r, angle


def fit_stadium(marker_positions: list, n: int = STADIUM_SAMPLES,
                cx: float = None, cy: float = None,
                forward_hint: np.ndarray = None) -> list:
    """Fit a stadium-shaped closed curve through track marker positions.

    A *stadium* (https://en.wikipedia.org/wiki/Stadium_(geometry)) consists of
    two semicircles of radius *r* joined by two parallel straight segments of
    length *2a*.  The four corner markers are placed at the tangent junctions::

        m0 ──── straight ──── m1
         )                    (     ← semicircles
        m3 ──── straight ──── m2

    The output is **always exactly n points** (resampled at the end) so that
    all three rings produce arrays of identical length for element-wise ops.

    Parameters
    ----------
    marker_positions : list of (x, y)  [m0, m1, m2, m3]
    n   : total sample points (same value for every ring)
    cx, cy : external centroid override to keep rings concentric
    """
    scx, scy, a, r, angle = _fit_stadium_params(marker_positions)
    if cx is not None and cy is not None:
        scx, scy = float(cx), float(cy)
    centre = np.array([scx, scy])

    perim_semi  = math.pi * r
    perim_str   = 2.0 * a
    total_perim = 2.0 * perim_semi + 2.0 * perim_str

    n_semi = max(2, round(n * perim_semi / (total_perim + 1e-9)))
    n_str  = max(1, round(n * perim_str  / (total_perim + 1e-9)))
    n_semi = max(2, (n - 2 * n_str) // 2)

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    u = np.array([ cos_a,  sin_a])
    v = np.array([-sin_a,  cos_a])
    pts_out = []

    for i in range(n_semi):          # right semicircle: −π/2 → +π/2
        th = -math.pi / 2.0 + math.pi * i / max(n_semi - 1, 1)
        pts_out.append(centre + a * u + r * (math.cos(th) * u + math.sin(th) * v))
    for i in range(1, n_str + 1):    # top straight: right → left
        t = i / (n_str + 1)
        pts_out.append(centre + (1.0 - 2.0 * t) * a * u + r * v)
    for i in range(n_semi):          # left semicircle: +π/2 → +3π/2
        th = math.pi / 2.0 + math.pi * i / max(n_semi - 1, 1)
        pts_out.append(centre - a * u + r * (math.cos(th) * u + math.sin(th) * v))
    for i in range(1, n_str + 1):    # bottom straight: left → right
        t = i / (n_str + 1)
        pts_out.append(centre + (2.0 * t - 1.0) * a * u - r * v)

    # Enforce exactly n points to prevent shape mismatches in detect_lanes
    pts_arr = np.array(pts_out, dtype=float)
    m = len(pts_arr)
    if m != n:
        idx = np.round(np.linspace(0, m - 1, n)).astype(int)
        pts_arr = pts_arr[idx]
    return list(zip(pts_arr[:, 0].astype(int).tolist(),
                    pts_arr[:, 1].astype(int).tolist()))

def detect_lanes(track_markers: dict) -> dict:
    """Fit closed reference curves for each lane from the detected track markers.

    Reads the three ring marker sets (inner / middle / outer), fits a stadium
    (pill-shape) curve through each, and assigns lane reference paths directly
    to the boundary lines:
        lane1_ref = inner_curve   (car tracks the inner boundary)
        lane2_ref = middle_curve  (car tracks the lane separator)

    Parameters
    ----------
    track_markers : dict {int: corner_array}
        Output of detect_all_markers() for the TRACK_DICT dictionary.

    Returns
    -------
    dict with keys: 'inner_curve', 'middle_curve', 'outer_curve',
                    'lane1_ref', 'lane1_ready', 'lane2_ref', 'lane2_ready',
                    'n_inner', 'n_middle', 'n_outer'.
    Lane*_ready flags are True only when enough markers were detected to
    build that curve; callers must check these before using the curves.
    """
    global _TRACK_FORWARD
    result = {}
    inner_pts  = [marker_center(track_markers[i]) for i in INNER_SET  if i in track_markers]
    middle_pts = [marker_center(track_markers[i]) for i in MIDDLE_SET if i in track_markers]
    outer_pts  = [marker_center(track_markers[i]) for i in OUTER_SET  if i in track_markers]
    result.update({"n_inner": len(inner_pts), "n_middle": len(middle_pts), "n_outer": len(outer_pts)})
    if _TRACK_FORWARD is None and len(outer_pts) >= 4:
        _TRACK_FORWARD = get_track_direction(track_markers)

    all_pts = inner_pts + middle_pts + outer_pts
    if len(all_pts) < 3:
        return result

    all_arr = np.array(all_pts, dtype=float)
    cx, cy  = float(all_arr[:, 0].mean()), float(all_arr[:, 1].mean())

    # Stadium fitting: markers are tangent corners of the pill shape.
    # n= is explicit so all three rings always have identical array lengths.
    if len(inner_pts)  >= 2: 
        result["inner_curve"]  = fit_stadium(inner_pts, cx=cx, cy=cy,  forward_hint=_TRACK_FORWARD)
        # result['inner_curve'] = result['inner_curve'][::-1]  # flip to CW image-space order
    if len(middle_pts) >= 2: 
        result["middle_curve"] = fit_stadium(middle_pts, cx=cx, cy=cy,  forward_hint=_TRACK_FORWARD)
        # result['middle_curve'] = result['middle_curve'][::-1]  # flip to CW image-space order
    if len(outer_pts)  >= 2: 
        result["outer_curve"]  = fit_stadium(outer_pts, cx=cx, cy=cy,  forward_hint=_TRACK_FORWARD)
        # result['outer_curve'] = result['outer_curve'][::-1]  # flip to CW image-space order

    # Lane 1 tracks the inner boundary; lane 2 tracks the middle boundary.
    # Using the boundary lines directly (rather than averaged centrelines)
    # gives a crisper reference with geometry consistent with the markers.
    if "inner_curve" in result and "middle_curve" in result:
        inner_arr  = np.array(result["inner_curve"],  dtype=float)
        middle_arr = np.array(result["middle_curve"], dtype=float)
        centre_arr = ((inner_arr + middle_arr) / 2.0).astype(int)
        result["lane1_centre"]    = list(zip(centre_arr[:,0].tolist(),
                                             centre_arr[:,1].tolist()))
        result["lane1_ref"]       = result["lane1_centre"]
        result["lane1_ready"]     = True
        # Auto-calibrated half-width for the containment test below.
        half_gap = float(np.mean(np.linalg.norm(middle_arr - inner_arr, axis=1))) / 2.0
        result["lane1_half_width"] = max(half_gap, LANE_HALF_WIDTH)

    if "middle_curve" in result and "outer_curve" in result:
        middle_arr2 = np.array(result["middle_curve"], dtype=float)
        outer_arr   = np.array(result["outer_curve"],  dtype=float)
        centre2_arr = ((middle_arr2 + outer_arr) / 2.0).astype(int)
        result["lane2_centre"]    = list(zip(centre2_arr[:,0].tolist(),
                                             centre2_arr[:,1].tolist()))
        result["lane2_ref"]       = result["lane2_centre"]
        result["lane2_ready"]     = True
        half_gap2 = float(np.mean(np.linalg.norm(outer_arr - middle_arr2, axis=1))) / 2.0
        result["lane2_half_width"] = max(half_gap2, LANE_HALF_WIDTH)

    return result

def _classify_lane_from_pos(front_pos, rear_pos, lanes: dict,
                             current_lane: int) -> int:
    """
    Classify the car's lane only when BOTH the front and rear marker centres
    are geometrically inside the same lane (same side of the middle curve).

    If the two markers straddle the middle boundary (one on each side) the
    function returns current_lane unchanged — the car is mid-transition and
    no lane switch should be triggered.

    A 10 px hysteresis band (LANE_HYSTERESIS) on each side prevents
    chattering when a marker grazes the boundary.
    """
    middle = lanes.get("middle_curve")
    if middle is None or not lanes.get("lane2_ready", False):
        return 1   # single-lane mode

    hw   = lanes.get("lane1_half_width", LANE_HALF_WIDTH)
    hyst = LANE_HYSTERESIS
    arr  = np.array(middle, dtype=float)

    def _signed_dist(pos):
        """Signed perpendicular distance from pos to the middle curve.
        Positive = inner side (lane 1).  Negative = outer side (lane 2).
        """
        p   = np.array(pos, dtype=float)
        idx = int(np.argmin(np.linalg.norm(arr - p, axis=1)))
        _, _, left_normal = local_path_frame(middle, idx)
        return float(np.dot(p - arr[idx], left_normal))

    d_front = _signed_dist(front_pos)
    d_rear  = _signed_dist(rear_pos)

    # Classify each marker independently with hysteresis
    def _side(d):
        if d < -(hw - hyst):
            return 2   # outer side → lane 2
        if d >  (hw - hyst):
            return 1   # inner side → lane 1
        return 0       # hysteresis band → uncertain

    side_front = _side(d_front)
    side_rear  = _side(d_rear)

    # Only commit to a new lane if BOTH markers agree on the same side
    if side_front == side_rear and side_front != 0:
        return side_front

    # One or both markers are in the hysteresis band, or they disagree
    # (car is straddling the middle line) → hold previous classification
    return current_lane

def get_track_direction(track_markers: dict) -> np.ndarray:
    """
    Compute a reference forward vector based on the spatial order of
    the outer or middle ring markers. Assumes markers are placed in
    clockwise (CW) order around the track.
    Returns a unit vector pointing in the intended travel direction.
    """
    # Get centre points of outer markers, sorted by angle around the centroid
    pts = [marker_center(track_markers[i]) for i in MIDDLE_SET if i in track_markers]
    if len(pts) < 4:
        return np.array([1.0, 0.0])  # fallback
    centre = np.mean(pts, axis=0)
    # Negate y for image coords, sort CW
    angles = np.arctan2(-(np.array([p[1] for p in pts]) - centre[1]),
                         np.array([p[0] for p in pts]) - centre[0])
    sortedpts = [p for _, p in sorted(zip(angles, pts), reverse=True)]
    forward = np.array(sortedpts[1]) - np.array(sortedpts[0])
    return forward / (np.linalg.norm(forward) + 1e-9)

# ══════════════════════════════════════════════════════════════════════════════
# MARKER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_all_markers(frame: np.ndarray):
    """Detect all three ArUco marker types in one camera frame.

    Marker persistence
    ------------------
    Track markers (5×5): if a previously seen marker is absent, its last
    valid corner array is silently substituted so stadium curves stay stable.

    Obstacle markers (6×6): same policy. A dimmed cyan circle + "OBS*" label
    is drawn at the cached position to distinguish cached from live.

    Car markers (4×4): returned live only; fallback is handled by
    identify_cars() via _car_marker_cache.

    Returns
    -------
    car_markers   : dict {int: corner_array} — live 4×4 detections
    track_markers : dict {int: corner_array} — live + cached 5×5 detections
    obs_markers   : dict {int: corner_array} — live + cached 6×6 detections
    """
    global _last_track_markers, _last_obs_positions
    now = time.time()
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = aruco.DetectorParameters()

    def _detect(dictionary, color=None):
        det = aruco.ArucoDetector(aruco.getPredefinedDictionary(dictionary), params)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            return {}
        if color is None:
            aruco.drawDetectedMarkers(frame, corners, ids)
        else:
            aruco.drawDetectedMarkers(frame, corners, ids, color)
        return {int(i): corners[k] for k, i in enumerate(ids.flatten())}

    car_markers = _detect(CAR_DICT)

    # ── Track markers: persist across frames ──────────────────────────────
    live_track = _detect(TRACK_DICT)
    for mid, corner in live_track.items():
        _last_track_markers[mid] = (corner, now)
    track_markers = dict(live_track)
    for mid, (corner, t) in list(_last_track_markers.items()):
        if mid not in track_markers:
            if _MARKER_MAX_AGE is None or (now - t) <= _MARKER_MAX_AGE:
                track_markers[mid] = corner  # substitute cached corner

    # ── Obstacle markers: persist across frames ───────────────────────────
    live_obs = _detect(OBSTACLE_DICT, (0, 0, 255))
    for mid, corner in live_obs.items():
        _last_obs_positions[mid] = (marker_center(corner), now)
    obs_markers = dict(live_obs)
    for mid, (pos, t) in list(_last_obs_positions.items()):
        if mid not in obs_markers:
            if _MARKER_MAX_AGE is None or (now - t) <= _MARKER_MAX_AGE:
                # Draw cached obstacle in dimmed cyan (visually distinct from live)
                cv2.circle(frame, pos, 6, (0, 160, 160), 2)
                cv2.putText(frame, "OBS*", (pos[0] + 8, pos[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 160, 160), 1)
                # Synthesise a minimal corner array so downstream code is unchanged
                px, py = pos
                fake = np.array([[[px - 6, py - 6],
                                   [px + 6, py - 6],
                                   [px + 6, py + 6],
                                   [px - 6, py + 6]]], dtype=np.float32)
                obs_markers[mid] = fake

    return car_markers, track_markers, obs_markers

# ══════════════════════════════════════════════════════════════════════════════
# CAR IDENTIFICATION & SPEED TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def identify_cars(car_markers: dict, car_id: int) -> dict:
    """Build a structured pose dict for every visible car from raw marker data.

    Car marker persistence & front-marker fallback
    -----------------------------------------------
    Normal (both visible): rear = axle reference, heading = front−rear.
    Rear occluded → FRONT FALLBACK: front marker becomes the axle reference
        ("rear" key set to front_center). Heading reconstructed from cached
        rear. 'using_front_fallback' = True. HUD shows [F!]. Event logged.
    Front occluded → REAR ONLY: rear used normally, heading = (0,0).
    Both expired → car_id absent from dict; run() issues safe stop.
    """
    global _car_marker_cache
    now = time.time()

    if car_id not in _car_marker_cache:
        _car_marker_cache[car_id] = {
            "front_corner": None, "rear_corner": None,
            "front_time":   None, "rear_time":   None,
            "using_front_fallback": False,
        }
    cache = _car_marker_cache[car_id]

    # Refresh cache with whatever is live this frame
    if FRONT_MARKER_ID in car_markers:
        cache["front_corner"] = car_markers[FRONT_MARKER_ID]
        cache["front_time"]   = now
    if car_id in car_markers:
        cache["rear_corner"] = car_markers[car_id]
        cache["rear_time"]   = now

    def _valid(t):
        if t is None:
            return False
        return _MARKER_MAX_AGE is None or (now - t) <= _MARKER_MAX_AGE

    front_ok = _valid(cache["front_time"])
    rear_ok  = _valid(cache["rear_time"])

    # Other cars — legacy rear-only
    cars = {}
    for mid, corner in car_markers.items():
        if mid == FRONT_MARKER_ID or mid == car_id:
            continue
        rc = marker_center(corner)
        cars[mid] = {"front": None, "rear": rc, "midpoint": rc,
                     "heading": (0, 0), "using_front_fallback": False}

    if rear_ok and front_ok:
        # ── NORMAL ───────────────────────────────────────────────────────
        fc = marker_center(cache["front_corner"])
        rc = marker_center(cache["rear_corner"])
        cache["using_front_fallback"] = False
        cars[car_id] = {
            "front":    fc,
            "rear":     rc,
            "midpoint": midpoint(fc, rc),
            "heading":  (fc[0] - rc[0], fc[1] - rc[1]),
            "using_front_fallback": False,
        }

    elif front_ok and not rear_ok:
        # ── FRONT FALLBACK ────────────────────────────────────────────────
        fc = marker_center(cache["front_corner"])
        if cache["rear_corner"] is not None:
            rc_cached = marker_center(cache["rear_corner"])
            dx, dy = fc[0] - rc_cached[0], fc[1] - rc_cached[1]
            norm_h = math.hypot(dx, dy)
            heading = (int(dx / norm_h * WHEELBASE_PX),
                       int(dy / norm_h * WHEELBASE_PX)) if norm_h > 1e-4 else (0, 0)
        else:
            heading = (0, 0)
        cache["using_front_fallback"] = True
        cars[car_id] = {
            "front":    fc,
            "rear":     fc,        # front marker acts as bicycle-model axle
            "midpoint": fc,
            "heading":  heading,
            "using_front_fallback": True,
        }

    elif rear_ok and not front_ok:
        # ── REAR ONLY (degraded) ──────────────────────────────────────────
        rc = marker_center(cache["rear_corner"])
        cache["using_front_fallback"] = False
        cars[car_id] = {
            "front":    None,
            "rear":     rc,
            "midpoint": rc,
            "heading":  (0, 0),
            "using_front_fallback": False,
        }
    # else: both expired → car_id absent → run() will safe-stop

    return cars


def estimate_speed(curr_pos, prev_pos, dt: float) -> float:
    """Estimate instantaneous speed in scaled units per second.

    Speed = pixel displacement / (dt × SCALING_FACTOR), giving a value in
    centimetres per second when SCALING_FACTOR is correctly calibrated.

    Returns 0.0 if prev_pos is None or dt ≤ 0.
    """
    if prev_pos is None or dt <= 0:
        return 0.0
    return float(np.linalg.norm(np.array(curr_pos) - np.array(prev_pos)) / (dt * SCALING_FACTOR))


def update_tracker(cars: dict, current_time: float) -> None:
    """Refresh the module-level tracker dict with current speed and status.

    Called every frame. Only updates entries that have been waiting at least
    STATUS_UPDATE_INTERVAL seconds since their last update, which avoids
    spurious speed spikes from sub-frame timing noise.

    Side effects: mutates the global `tracker` dict.
    """
    for rear_id, cd in cars.items():
        if rear_id not in tracker:
            tracker[rear_id] = {"status": None, "center": None,
                                 "last_time": None, "heading": None, "speed": 0.0}
        t         = tracker[rear_id]
        last_time = t["last_time"]
        if last_time is None or (current_time - last_time >= STATUS_UPDATE_INTERVAL):
            dt          = (current_time - last_time) if last_time else 0.0
            speed       = estimate_speed(cd["midpoint"], t["center"], dt)
            t["speed"]  = speed
            t["status"] = "moving" if speed > SPEED_THRESHOLD else "stopped"
            t["center"] = cd["midpoint"]
            t["last_time"] = current_time
            t["heading"]   = cd["heading"]

# ══════════════════════════════════════════════════════════════════════════════
# PATH & CURVE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def project_onto_curve(pos: tuple, curve_pts: list) -> int:
    """Return the index of the curve point nearest to pos (Euclidean).

    Used as the starting index for lookahead walks and path-frame computations.
    """
    arr = np.array(curve_pts, dtype=float)
    return int(np.argmin(np.linalg.norm(arr - np.array(pos, dtype=float), axis=1)))


def local_path_frame(curve_pts: list, idx: int):
    """Compute the local Frenet frame at one point on the closed curve.

    The tangent is estimated as the chord between the predecessor and
    successor points (central differences), giving a smooth direction
    estimate on the discretised curve.

    Parameters
    ----------
    curve_pts : list of (x, y) tuples — the closed reference polyline.
    idx       : int — index of the query point.

    Returns
    -------
    (p_curr, tangent, left_normal)
        p_curr      : np.ndarray, shape (2,) — the query point.
        tangent     : np.ndarray, shape (2,) — unit forward tangent.
        left_normal : np.ndarray, shape (2,) — unit left-pointing normal
                      (rotated 90° CCW from tangent).
    """
    n      = len(curve_pts)
    p_prev = np.array(curve_pts[(idx - 1) % n], dtype=float)
    p_curr = np.array(curve_pts[idx], dtype=float)
    p_next = np.array(curve_pts[(idx + 1) % n], dtype=float)
    tangent = p_next - p_prev
    norm_t  = np.linalg.norm(tangent)
    tangent = tangent / norm_t if norm_t >= 1e-6 else np.array([1.0, 0.0])
    left_normal = np.array([tangent[1], -tangent[0]])
    return p_curr, tangent, left_normal


def compute_lane_measurements(car_pos: tuple, heading_vec: tuple, curve_pts: list) -> dict:
    """Project the car onto the lane reference curve and compute tracking errors.

    Lateral error is the signed perpendicular distance from the nearest curve
    point, measured along the left normal (positive = car is left of centre).
    Heading error is the signed angular difference between the path tangent
    and the car's heading (positive = car needs to turn left).

    These errors are the primary inputs to the HUD and the experiment logger
    (methodology §Pose tracking error).

    Returns
    -------
    dict with keys:
        'idx'              – nearest curve index
        'path_point'       – (x, y) of nearest curve point
        'tangent'          – unit tangent at that point
        'left_normal'      – unit left normal at that point
        'lateral_error'    – signed lateral offset (px)
        'heading_error'    – signed heading difference (deg)
        'tangent_angle'    – compass angle of the path tangent (deg)
        'car_heading_angle'– compass angle of the car heading (deg)
    """
    idx               = project_onto_curve(car_pos, curve_pts)
    p_ref, tangent, left_normal = local_path_frame(curve_pts, idx)
    pos_vec           = np.array(car_pos, dtype=float) - p_ref
    lateral_error     = float(np.dot(pos_vec, left_normal))
    tangent_angle     = float(np.degrees(np.arctan2(-tangent[1], tangent[0])) % 360)
    if np.linalg.norm(np.array(heading_vec, dtype=float)) < 1e-6:
        car_heading_angle = tangent_angle
    else:
        car_heading_angle = heading_to_angle(heading_vec)
    heading_error = wrap_angle_deg(tangent_angle - car_heading_angle)
    return {
        "idx":              idx,
        "path_point":       tuple(p_ref.astype(int)),
        "tangent":          tangent,
        "left_normal":      left_normal,
        "lateral_error":    lateral_error,
        "heading_error":    heading_error,
        "tangent_angle":    tangent_angle,
        "car_heading_angle": car_heading_angle,
    }

# ══════════════════════════════════════════════════════════════════════════════
# OBSTACLE DETECTION & CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _classify_obstacle_distance(d: float) -> dict:
    """Classify an obstacle's proximity into three named states.

    Thresholds are chosen to give comfortable transition zones:
        blocking : d < 70 px  (definite stop)
        near     : 60–170 px  (slow down / yield)
        clear    : d > 130 px (no action needed)

    Zones overlap intentionally — dominant_state() picks the highest value.

    Parameters
    ----------
    d : float — Euclidean pixel distance to the obstacle.

    Returns
    -------
    dict {'blocking': float, 'near': float, 'clear': float}
        Membership values in [0, 1].
    """
    def _ls(x, a, b): return 1.0 if x<=a else 0.0 if x>=b else (b-x)/(b-a+1e-9)
    def _tri(x,a,b,c):
        if x<=a or x>=c: return 0.0
        return (x-a)/(b-a+1e-9) if x<b else (c-x)/(c-b+1e-9)
    def _rs(x, a, b): return 0.0 if x<=a else 1.0 if x>=b else (x-a)/(b-a+1e-9)
    return {"blocking": _ls(d,45,70), "near": _tri(d,60,110,170), "clear": _rs(d,130,180)}


def nearest_relevant_obstacle(car_pos, tangent, left_normal, obstacles) -> dict:
    """Find the nearest obstacle that lies ahead and within the lane corridor.

    Scans all detected obstacle positions and retains only those that are:
        1. Ahead of the car along the path tangent (along > 0).
        2. Within OBSTACLE_LOOKAHEAD pixels longitudinally.
        3. Within OBSTACLE_TRACK_HALF_WIDTH pixels laterally.

    The closest qualifying obstacle is classified via _classify_obstacle_distance()
    and returned with its distance, side offset, state label, and pixel position.
    If no relevant obstacle exists, a 'clear' sentinel is returned.

    Parameters
    ----------
    car_pos     : (x, y) pixel position of the car reference point.
    tangent     : unit forward tangent vector at the car's path projection.
    left_normal : unit left normal vector at the car's path projection.
    obstacles   : list of (x, y) pixel positions from obstacle markers.

    Returns
    -------
    dict with keys: 'distance', 'side_offset', 'state', 'point'.
    """
    _CLEAR = {"distance": float("inf"), "side_offset": 0.0, "state": "clear", "point": None}
    if not obstacles:
        return _CLEAR
    t  = tangent     / (np.linalg.norm(tangent)     + 1e-9)
    n  = left_normal / (np.linalg.norm(left_normal) + 1e-9)
    p0 = np.array(car_pos, dtype=float)
    best_dist, best = float("inf"), None
    for obs in obstacles:
        vec  = np.array(obs, dtype=float) - p0
        along = float(np.dot(vec, t))
        side  = float(np.dot(vec, n))
        eucl  = float(np.linalg.norm(vec))
        if along < 0 or along > OBSTACLE_LOOKAHEAD: continue
        if abs(side) > OBSTACLE_TRACK_HALF_WIDTH:    continue
        if eucl < best_dist:
            best_dist, best = eucl, (obs, side)
    if best is None:
        return _CLEAR
    return {"distance": best_dist, "side_offset": best[1],
            "state": dominant_state(_classify_obstacle_distance(best_dist)), "point": best[0]}

# ══════════════════════════════════════════════════════════════════════════════
# LANE-CHANGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _lane_obstacle_free(car_pos, lanes, lane_num) -> bool:
    """Return True if no relevant obstacle is detected ahead in the given lane.

    Projects car_pos onto the target lane's reference curve to obtain the
    local tangent frame, then queries nearest_relevant_obstacle() for that lane.
    """
    ref = lanes.get(f"lane{lane_num}_ref", [])
    if not ref: return False
    idx = project_onto_curve(car_pos, ref)
    _, tangent, ln = local_path_frame(ref, idx)
    return nearest_relevant_obstacle(car_pos, tangent, ln, obstacle_positions)["state"] == "clear"


def adjacent_lane_cars(cars, lane_num) -> list:
    """Return IDs of all cars currently assigned to lane_num.

    Reads from the module-level lane_state dict; cars not yet registered
    default to lane 1.
    """
    return [cid for cid in cars if lane_state.get(cid, {}).get("lane", 1) == lane_num]


def cooperative_gap_ok(car_pos, adj_cars, cars) -> bool:
    """Check whether a cooperative lane change is safe given adjacent traffic.

    Returns True only if every car in adj_cars is:
        • at least COOP_MIN_GAP pixels away, AND
        • not closing faster than COOP_MAX_CLOSING_SPEED while inside
          COOP_MIN_GAP × 1.5 pixels.
    """
    p0 = np.array(car_pos, dtype=float)
    for cid in adj_cars:
        if cid not in cars: continue
        gap     = float(np.linalg.norm(np.array(cars[cid]["midpoint"], dtype=float) - p0))
        closing = tracker.get(cid, {}).get("speed", 0.0)
        if gap < COOP_MIN_GAP: return False
        if closing > COOP_MAX_CLOSING_SPEED and gap < COOP_MIN_GAP * 1.5: return False
    return True


def apply_implicit_coop(adj_cars, duration: float = 1.5) -> None:
    """Set a timed slow-down hint for adjacent-lane cars.

    When a car performs a cooperative merge, this function signals the cars
    it is merging in front of by setting their coop_slowdown_until expiry.
    Those cars will then apply a speed reduction in the next control cycle.

    Parameters
    ----------
    adj_cars : list of car_ids to signal.
    duration : float — how many seconds the slow-down hint lasts.
    """
    expiry = time.time() + duration
    for cid in adj_cars:
        coop_slowdown_until[cid] = expiry

# ══════════════════════════════════════════════════════════════════════════════
# DRIVING POLICY BRANCHES
# ══════════════════════════════════════════════════════════════════════════════

def egocentric_decide_lane(car_id, car_pos, lanes, obstacle_info, now) -> int:
    """Egocentric lane-change policy: switch immediately when blocked.

    No safety gap check is performed — the car switches to the adjacent lane
    the moment its current lane is blocked ('blocking' or 'near' state).
    Returns to lane 1 after LANE_CHANGE_HOLD seconds once the obstacle clears.

    This policy is used for the non-cooperative scenario (RQ2 baseline).
    """
    if car_id not in lane_state:
        lane_state[car_id] = {"lane": 1, "timer": now, "overtaking": False}
    st      = lane_state[car_id]
    current = st["lane"]
    obs_bad = obstacle_info["state"] in ("blocking", "near")
    if obs_bad:
        adjacent = 2 if current == 1 else 1
        if lanes.get(f"lane{adjacent}_ready", False) and adjacent != current:
            st["lane"] = adjacent; st["timer"] = now; st["overtaking"] = True
    if st["overtaking"] and st["lane"] == 2:
        if (now - st["timer"] >= LANE_CHANGE_HOLD) and not obs_bad:
            if _lane_obstacle_free(car_pos, lanes, 1):
                st["lane"] = 1; st["timer"] = now; st["overtaking"] = False
    return st["lane"]


def cooperative_decide_lane(car_id, car_pos, lanes, obstacle_info, cars, now) -> int:
    """Cooperative lane-change policy: switch only when gap and timing are safe.

    Before switching, checks that:
        • the adjacent lane is obstacle-free (_lane_obstacle_free), AND
        • the spacing and closing speed to adjacent cars are acceptable
          (cooperative_gap_ok).
    If both conditions hold, signals adjacent cars to slow down
    (apply_implicit_coop) and performs the lane change.
    Returns to lane 1 after LANE_CHANGE_HOLD seconds once the obstacle clears.

    This policy is used for the cooperative scenario (RQ2 evaluation).
    """
    if car_id not in lane_state:
        lane_state[car_id] = {"lane": 1, "timer": now, "overtaking": False}
    st      = lane_state[car_id]
    current = st["lane"]
    obs_bad = obstacle_info["state"] in ("blocking", "near")
    if obs_bad:
        adjacent  = 2 if current == 1 else 1
        if lanes.get(f"lane{adjacent}_ready", False) and adjacent != current:
            adj_cars = adjacent_lane_cars(cars, adjacent)
            if _lane_obstacle_free(car_pos, lanes, adjacent) and cooperative_gap_ok(car_pos, adj_cars, cars):
                apply_implicit_coop(adj_cars)
                st["lane"] = adjacent; st["timer"] = now; st["overtaking"] = True
    if st["overtaking"] and st["lane"] == 2:
        if (now - st["timer"] >= LANE_CHANGE_HOLD) and not obs_bad:
            if _lane_obstacle_free(car_pos, lanes, 1):
                st["lane"] = 1; st["timer"] = now; st["overtaking"] = False
    return st["lane"]


def decide_lane(car_id, car_pos, lanes, obstacle_info, cars, now) -> int:
    """Top-level lane-decision dispatcher.

    Reads DRIVING_POLICY[car_id] (defaulting to DEFAULT_POLICY) and routes
    to the appropriate policy branch. In single-lane mode (lane2 not detected),
    the car is always kept in lane 1 and no overtaking is attempted; the caller
    is responsible for stopping the motor when an obstacle is blocking.
    """
    policy      = DRIVING_POLICY.get(car_id, DEFAULT_POLICY)
    single_lane = not lanes.get("lane2_ready", False)
    if single_lane:
        if car_id not in lane_state:
            lane_state[car_id] = {"lane": 1, "timer": now, "overtaking": False}
        lane_state[car_id]["lane"]      = 1
        lane_state[car_id]["overtaking"] = False
        return 1
    if policy == "egocentric":
        return egocentric_decide_lane(car_id, car_pos, lanes, obstacle_info, now)
    return cooperative_decide_lane(car_id, car_pos, lanes, obstacle_info, cars, now)

# ══════════════════════════════════════════════════════════════════════════════
# PURE-PURSUIT STEERING
# ══════════════════════════════════════════════════════════════════════════════

def _compute_lookahead(car_midpoint, heading_vec, lane_ref,
                       outer_ref=None):
    """
    Pure-Pursuit steering — textbook implementation (Coulter 1992 / Ding Yan 2019).

    Bicycle-model geometry
    ──────────────────────
    Reference point  : car_midpoint  — midpoint of front and rear ArUco markers,
                       used as the rear-axle equivalent (as in the rest of the
                       codebase).  Labelled 'p' in the formulae below.
    Lookahead circle : radius K_DD (pixels), centred on p.
    Lookahead target : the first intersection of the lookahead circle with the
                       upcoming reference path that lies ahead of the vehicle.
    α (alpha)        : signed angle between the vehicle heading and the chord
                       p → target.  Positive = target is to the left.
    Steering angle   : δ = atan2(2 · L · sin α,  l_d)
                       where L = WHEELBASE_PX, l_d = chord length p → target.
    Servo output     : δ_rad × _RAD_TO_SERVO  →  clipped to ±π/4 rad → ±0.5.

    How the forward target is found (robust to CW / CCW winding, large lateral
    offsets, and degraded heading estimates)
    ──────────────────────────────────────────
    Step 1  Find ni — the index of the path point nearest to p.

    Step 2  Determine the path's winding direction.
            The stored path may run CW or CCW.  We need to know which step
            direction (+1 or −1 in the index array) corresponds to "forward".
            We measure this by computing the path tangent at ni and comparing
            it to the vehicle heading.  If the tangent opposes the heading we
            flip the step direction.  When heading is unknown we fall back to
            _TRACK_FORWARD (set once at startup from marker geometry).

    Step 3  Forward-snap ni along the PATH tangent.
            Starting from the nearest point, advance ni one sample at a time
            (in the forward step direction) until the candidate point is no
            longer "behind" the vehicle — tested as:
                dot(arr[ni] − p, path_tangent_at_ni) ≥ 0
            We use the PATH tangent (not the vehicle heading) so the test is
            independent of large heading errors or lateral offsets.
            The advance is capped at K_DD arc-length to prevent jumping to
            the far side of a stadium-shaped track.

    Step 4  Arc-walk K_DD from the snapped ni.
            Walk forward along the path, accumulating arc length, until the
            total exceeds K_DD.  Interpolate the exact point on the segment
            where the arc length equals K_DD.  This is the lookahead target T.

    Step 5  Compute α and apply the pure-pursuit formula.

    Returns
    ───────
    (raw_delta_rad, cross_track_error_px, path_curvature, lookahead_point_px)

    raw_delta_rad        : steering angle in radians, clipped to [−π/4, +π/4].
    cross_track_error_px : signed lateral offset of p from the path (px).
    path_curvature       : dimensionless curvature κ at ni (for segment tagging).
    lookahead_point_px   : (x, y) pixel coords of the lookahead target T.
    """
    arr = np.array(lane_ref, dtype=float)       # (N, 2) path array
    pos = np.array(car_midpoint, dtype=float)   # reference point p
    n   = len(lane_ref)

    # ── Step 1: nearest path index ────────────────────────────────────────────
    ni = int(np.argmin(np.linalg.norm(arr - pos, axis=1)))

    # ── Path curvature at ni (returned for downstream segment classification) ─
    p0 = arr[(ni - 2) % n]
    p1 = arr[ni]
    p2 = arr[(ni + 2) % n]
    v1, v2        = p1 - p0, p2 - p1
    cross_prod    = abs(float(np.cross(v1, v2)))
    chord_len     = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + 1e-9
    path_curvature = cross_prod / (chord_len ** 2)
    print(f"curvature: {path_curvature:.4f}\n")

    # ── Step 2: determine forward step direction (+1 or −1) ───────────────────
    # Primary source: vehicle heading.
    # Fallback: module-level _TRACK_FORWARD set once from marker geometry.
    hx, hy       = float(heading_vec[0]), float(heading_vec[1])
    heading_norm = math.hypot(hx, hy)
    heading_known = heading_norm > 1e-4

    def _path_tangent(idx, step):
        """Unit tangent at arr[idx] in the given step direction (+1 or -1)."""
        t = (arr[(idx + step) % n] - arr[(idx - step) % n]).astype(float)
        nt = np.linalg.norm(t)
        return t / nt if nt > 1e-6 else np.array([1.0, 0.0])

    # Try the +1 direction first; compare its tangent to the heading / TRACK_FORWARD
    fwd_step = 1
    tangent_fwd = _path_tangent(ni, +1)

    if heading_known:
        ref_dir = np.array([hx / heading_norm, hy / heading_norm])
    elif _TRACK_FORWARD is not None:
        ref_dir = np.array(_TRACK_FORWARD, dtype=float)
        norm_tf  = np.linalg.norm(ref_dir)
        ref_dir  = ref_dir / norm_tf if norm_tf > 1e-6 else np.array([1.0, 0.0])
    else:
        ref_dir = tangent_fwd   # no information — assume +1 is forward

    if np.dot(tangent_fwd, ref_dir) < 0.0:
        fwd_step = -1   # path is stored backwards relative to travel direction

    # Adaptive lookahead: shorten on tight curves, use full K_DD on straights.
    # path_curvature is computed at ni (the forward-snapped index) just above.
    # Clamp to a minimum of 70 px so the target never collapses onto the car.
    lookahead_dist = max(70.0, K_DD / (1.0 + 8.0 * path_curvature))

    # ── Step 3: forward-snap ni along the path tangent ────────────────────────
    # Advance ni until the candidate point is "ahead" of p along the path.
    # Test: project (arr[ni] − p) onto the PATH tangent at ni.
    # Cap the advance at K_DD arc-length to avoid jumping to the far side
    # of a closed track (e.g. the opposite straight of a stadium).
    budget = lookahead_dist
    for _ in range(n):
        if budget <= 0:
            break
        t_ni   = _path_tangent(ni, fwd_step)
        to_ni  = arr[ni] - pos                  # vector: p → candidate
        if np.dot(to_ni, t_ni) >= 0.0:         # candidate is ahead → done
            break
        seg_len = float(np.linalg.norm(arr[(ni + fwd_step) % n] - arr[ni]))
        budget -= seg_len
        ni = (ni + fwd_step) % n

    # ── Step 4: arc-walk K_DD pixels to find the lookahead target T ──────────
    accum  = 0.0
    target = arr[ni].copy()
    for step in range(1, n):
        i0  = (ni + (step - 1) * fwd_step) % n
        i1  = (ni + step       * fwd_step) % n
        seg = float(np.linalg.norm(arr[i1] - arr[i0]))
        if accum + seg >= lookahead_dist:
            frac   = (lookahead_dist - accum) / (seg + 1e-9)
            target = arr[i0] + frac * (arr[i1] - arr[i0])
            break
        accum += seg


    # ── Cross-track error (signed lateral distance at snapped ni) ────────────
    _, _, left_normal_ni = local_path_frame(lane_ref, ni)
    cte = float(np.dot(pos - arr[ni], left_normal_ni))

    # ── Step 5: heading fallback → pure-CTE correction ───────────────────────
    if not heading_known:
        k_fb  = 2.0 * cte / (lookahead_dist ** 2 + 1e-9)
        delta = math.atan(k_fb * WHEELBASE_PX)
        delta = float(np.clip(delta, -(math.pi / 4), +(math.pi / 4)))
        return delta, cte, path_curvature, tuple(target.astype(int))

    # ── Compute α: signed angle from heading to chord p → T ──────────────────
    # Image-coord convention: x right, y down.
    # We negate y before atan2 to convert to standard maths coords (y up),
    # then wrap the difference to (−π, +π].
    # Positive α = target is to the LEFT of the heading = steer left = +δ.
    theta   = math.atan2(hy, hx)                       # heading angle (math coords)
    dx, dy  = float(target[0] - pos[0]), float(target[1] - pos[1])
    bearing = math.atan2(dy, dx)                        # chord bearing (math coords)
    alpha   = (bearing - theta + math.pi) % (2.0 * math.pi) - math.pi  # ∈ (−π, +π]

    # ── Pure-pursuit formula ──────────────────────────────────────────────────
    #   δ = atan2(2 · L · sin α,  l_d)
    l_d   = math.hypot(dx, dy) + 1e-9
    delta = math.atan2(2.0 * WHEELBASE_PX * math.sin(alpha), l_d)

    # ── Optional outer-boundary push correction ───────────────────────────────
    if outer_ref is not None:
        outer_arr = np.array(outer_ref, dtype=float)
        oi        = int(np.argmin(np.linalg.norm(outer_arr - pos, axis=1)))
        o_tan     = (outer_arr[(oi + 1) % len(outer_ref)]
                     - outer_arr[(oi - 1) % len(outer_ref)]).astype(float)
        o_tan    /= np.linalg.norm(o_tan) + 1e-9
        o_normal  = np.array([o_tan[1], -o_tan[0]])
        overshoot = float(np.dot(pos - outer_arr[oi], o_normal))
        if overshoot > 0:
            delta -= K_BOUNDARY_PUSH * overshoot
    # print(f"alpha: {math.degrees(alpha):.2f} deg, delta: {math.degrees(delta):.2f} deg\n")
    delta = float(np.clip(delta, -(math.pi / 4), +(math.pi / 4)))
    # print(f"delta (post-clip): {math.degrees(delta):.2f} deg\n")
    return delta, cte, path_curvature, tuple(target.astype(int))

# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED COORDINATION
# ══════════════════════════════════════════════════════════════════════════════

def _apply_coordination(cars, base_speed) -> dict:
    """Apply pairwise distance-threshold coordination to all detected cars.

    For every pair of cars, measures the Euclidean pixel distance between their
    midpoints. If one car is within D_SAFE or D_WARN of another, the higher-ID
    car (the 'yielder') has its speed reduced. Cars with policy 'non_cooperative'
    are excluded from yielding.

    Same lane  (both cars in identical lane number):
        d < D_SAFE  → STOP  (emergency — they are about to collide)
        d < D_WARN  → SLOW_SPEED

    Different lanes:
        d < D_SAFE  → SLOW_SPEED only (proximity warning, no full stop)
        d >= D_SAFE → no action

    Parameters
    ----------
    cars       : dict — output of identify_cars().
    base_speed : float — speed before coordination (from _apply_curve_slowdown).

    Returns
    -------
    dict {car_id: speed} — per-car speed after coordination rules are applied.
    """
    speeds = {cid: base_speed for cid in cars}
    ids = sorted(cars.keys())
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1:]:
            pa = np.array(cars[id_a]["midpoint"], dtype=float)
            pb = np.array(cars[id_b]["midpoint"], dtype=float)
            d  = float(np.linalg.norm(pa - pb))

            lane_a = lane_state.get(id_a, {}).get("lane", 1)
            lane_b = lane_state.get(id_b, {}).get("lane", 1)
            same_lane = (lane_a == lane_b)

            yielder = max(id_a, id_b)   # higher ID yields
            if DRIVING_POLICY.get(yielder, DEFAULT_POLICY) == "non_cooperative":
                continue

            if same_lane:
                if d < D_SAFE:
                    speeds[yielder] = STOP_SPEED
                    _pp_waiting[yielder] = True
                    # Tag as emergency in events — picked up by logger next frame
                elif d < D_WARN:
                    speeds[yielder] = min(speeds[yielder], SLOW_SPEED)
                    _pp_waiting[yielder] = True
                else:
                    _pp_waiting[yielder] = False
            else:
                # Different lanes — proximity warning only, no hard stop
                if d < D_SAFE:
                    speeds[yielder] = min(speeds[yielder], SLOW_SPEED)
                else:
                    _pp_waiting.setdefault(yielder, False)

    return speeds

# ══════════════════════════════════════════════════════════════════════════════
# CURVE SPEED REDUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _classify_segment(car_pos, lanes, lane: int = 1) -> tuple:
    """Classify the current track section as 'straight', 'light_curve', or 'sharp_curve'.

    Uses the **local curvature of the lane's own reference path** at the point
    nearest to the car.  This is more reliable than centroid-distance heuristics
    because it directly measures how sharply the path bends where the car is,
    independent of the track's overall size or aspect ratio.

    Curvature is estimated via the cross-product formula on the three-point
    stencil (p_{i-2}, p_i, p_{i+2}) at the nearest curve index:

        κ = |v1 × v2| / ||chord||²

    where v1 = p_i − p_{i-2}, v2 = p_{i+2} − p_i.

    Thresholds (empirically tuned for the minicar track at ~10 px/cm):
        κ > 0.015  → sharp_curve
        κ > 0.007  → light_curve
        otherwise  → straight

    Parameters
    ----------
    car_pos : (x, y) pixel position of the car reference point.
    lanes   : output of detect_lanes().
    lane    : active lane number (1 or 2) — selects the reference path.
    """
    ref = lanes.get(f"lane{lane}_ref", [])
    if not ref or len(ref) < 5:
        return "straight"
    n   = len(ref)
    idx = project_onto_curve(car_pos, ref)
    p0  = np.array(ref[(idx - 2) % n], dtype=float)
    p1  = np.array(ref[idx],            dtype=float)
    p2  = np.array(ref[(idx + 2) % n], dtype=float)
    v1, v2   = p1 - p0, p2 - p1

    def _cross2d(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    
    cross    = abs(float(_cross2d(v1, v2)))
    seg_len  = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + 1e-9
    curvature = cross / (seg_len ** 2)
    
    if curvature > 0.3:
        return "sharp_curve", round(curvature,3)
    if curvature > 0.07:
        return "light_curve", round(curvature,3)
    return "straight", round(curvature,3)


# def _apply_curve_slowdown(car_pos: tuple, lanes: dict, lane: int = 1) -> float:
#     """Return the target speed based on local curvature of the active lane's path.

#     Uses the same three-point curvature stencil as _classify_segment() but on
#     the reference path for *lane* (1 = inner boundary, 2 = middle boundary),
#     so the speed decision matches the actual path the car is following.

#     Parameters
#     ----------
#     car_pos : (x, y) pixel reference position.
#     lanes   : output of detect_lanes().
#     lane    : active lane number (1 or 2).
#     """
#     ref = lanes.get(f"lane{lane}_ref", [])
#     if not ref or len(ref) < 5:
#         return CRUISE_SPEED
#     n   = len(ref)
#     idx = project_onto_curve(car_pos, ref)
#     p0  = np.array(ref[(idx - 2) % n], dtype=float)
#     p1  = np.array(ref[idx],            dtype=float)
#     p2  = np.array(ref[(idx + 2) % n], dtype=float)
#     v1, v2    = p1 - p0, p2 - p1
#     cross     = abs(float(np.cross(v1, v2)))
#     seg_len   = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + 1e-9
#     curvature = cross / (seg_len ** 2)
#     if curvature > 0.3:
#         return SLOW_SPEED
#     if curvature > 0.08:
#         return CRUISE_SPEED * 0.85
#     return CRUISE_SPEED 

# ══════════════════════════════════════════════════════════════════════════════
# DRAWING / VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def draw_lanes(frame, lanes):
    """Render lane fills, boundary polylines, and dashed reference centrelines.

    Lane 1 (inner–middle) is tinted green; lane 2 (middle–outer) is tinted
    blue. Both fills use a low-opacity blend (α = 0.20) so marker overlays
    remain visible. Reference centrelines are drawn as dashed polylines every
    8 curve samples.
    """
    overlay = frame.copy()
    if "inner_curve" in lanes and "middle_curve" in lanes:
        cv2.fillPoly(overlay, [np.vstack([np.array(lanes["inner_curve"], np.int32),
                                          np.array(lanes["middle_curve"], np.int32)[::-1]])], LANE1_FILL_COLOR)
    if "middle_curve" in lanes and "outer_curve" in lanes:
        cv2.fillPoly(overlay, [np.vstack([np.array(lanes["middle_curve"], np.int32),
                                          np.array(lanes["outer_curve"], np.int32)[::-1]])], LANE2_FILL_COLOR)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    for key, color in [("inner_curve", INNER_LINE_COLOR), ("middle_curve", MIDDLE_LINE_COLOR),
                       ("outer_curve", OUTER_LINE_COLOR)]:
        if key in lanes:
            cv2.polylines(frame, [np.array(lanes[key], np.int32)], True, color, 2)
    vis_centres = []
    if "inner_curve" in lanes and "middle_curve" in lanes:
        mid = ((np.array(lanes["inner_curve"], float)
                + np.array(lanes["middle_curve"], float)) / 2).astype(int)
        vis_centres.append([tuple(p) for p in mid.tolist()])
    if "middle_curve" in lanes and "outer_curve" in lanes:
        mid = ((np.array(lanes["middle_curve"], float)
                + np.array(lanes["outer_curve"], float)) / 2).astype(int)
        vis_centres.append([tuple(p) for p in mid.tolist()])
    for pts in vis_centres:
        nn = len(pts)
        for i in range(0, nn, 8):
            cv2.line(frame, pts[i], pts[(i + 4) % nn], REF_LINE_COLOR, 1)
    return frame


def draw_guide_line(frame, ref_pose, target, obstacle_close):
    """Draw the pure-pursuit guide line from the front marker to the lookahead point.

    The line is shown in red when an obstacle is close ('blocking' or 'near')
    and white otherwise, giving an immediate visual cue of the threat state.
    """
    color = (0, 0, 220) if obstacle_close else (255, 255, 255)
    cv2.circle(frame, target, 9, color, 2)
    cv2.circle(frame, target, 3, color, -1)
    cv2.line(frame, ref_pose, target, color, 2)
    return frame


def draw_obstacles(frame, obstacles):
    """Draw each detected obstacle as a translucent filled circle with an 'OBS' label."""
    for obs in obstacles:
        ox, oy  = obs
        overlay = frame.copy()
        cv2.circle(overlay, (ox, oy), OBSTACLE_RADIUS, (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.circle(frame, (ox, oy), OBSTACLE_RADIUS, (0, 0, 255), 2)
        cv2.circle(frame, (ox, oy), 6, (0, 0, 255), -1)
        cv2.putText(frame, "OBS", (ox + 8, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return frame


def draw_cars(frame, cars):
    """Overlay car marker positions on the frame.

    Colour convention:
        Red dot          – rear marker (normal mode)
        Cyan dot         – rear marker when front-fallback is active
        Orange dot       – midpoint (vehicle reference position)
        Green dot        – front marker
        Magenta line     – rear-to-front axis (heading vector)
        Cyan ring + tag  – shown around front marker in front-fallback mode
    """
    for _, cd in cars.items():
        fallback   = cd.get("using_front_fallback", False)
        rear_color = (0, 200, 200) if fallback else (0, 0, 255)
        cv2.circle(frame, cd["rear"],     8, rear_color,    -1)
        cv2.circle(frame, cd["midpoint"], 6, (0, 165, 255), -1)
        if cd["front"] is not None:
            cv2.circle(frame, cd["front"], 8, (0, 255, 0), -1)
            cv2.line(frame, cd["rear"], cd["front"], (255, 0, 255), 2)
            if fallback:
                cv2.circle(frame, cd["front"], 14, (0, 255, 255), 2)
                cv2.putText(frame, "FRONT CTRL",
                            (cd["front"][0] + 10, cd["front"][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return frame


def draw_hud(frame, car_id, servo, motor, lane_info, obstacle_info,
             steer_state, speed_state, current_lane, lc_state, policy):
    """Render a compact semi-transparent telemetry panel in the top-left corner.

    Displays six lines of real-time diagnostic data:
        Line 1 : Car ID, lane, overtake tag, policy, waiting flag
        Line 2 : Servo and motor command values, cooperative hint flag
        Line 3 : Lateral error (px) and dominant lateral state label
        Line 4 : Heading error (°) and dominant heading state label
        Line 5 : Obstacle distance (px) and obstacle state label
        Line 6 : Speed zone and obstacle steer state

    Also draws a yellow dot at the nearest path point and a red line to the
    nearest obstacle when one is present.
    """
    FONT, SCALE, THICKNESS, LINE_H, PAD = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1, 14, 6
    lc_tag   = "OVT" if lc_state.get("overtaking") else "NRM"
    coop_tag = " C!" if speed_state.get("coop_hint") else ""
    obs_d    = obstacle_info["distance"]
    obs_str  = f"{obs_d:.0f}px" if obs_d < float("inf") else "---"
    wait_tag     = " [W]"  if _pp_waiting.get(car_id, False) else ""
    fallback_tag = " [F!]" if _car_marker_cache.get(car_id, {}).get("using_front_fallback", False) else ""
    hud_lines = [
        f"Car {car_id} | Ln{current_lane} {lc_tag} | {policy[:4].upper()}{wait_tag}{fallback_tag}",
        f"Srv {servo:+.2f} Mtr {motor:.2f}{coop_tag}",
        f"Lat {lane_info['lateral_error']:+.0f}px {steer_state.get('lateral', '-')[:4]}",
        f"Hd {lane_info['heading_error']:+.0f}° {steer_state.get('heading', '-')[:4]}",
        f"Obs {obs_str} {obstacle_info['state'][:4]}",
        f"Spd {speed_state.get('zone', '-')} {steer_state.get('obstacle', '-')[:4]}",
    ]
    max_w   = max(cv2.getTextSize(l, FONT, SCALE, THICKNESS)[0][0] for l in hud_lines)
    panel_w = max_w + PAD * 2
    panel_h = LINE_H * len(hud_lines) + PAD * 2
    x0, y0  = 8, 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, txt in enumerate(hud_lines):
        cv2.putText(frame, txt, (x0 + PAD, y0 + PAD + (i + 1) * LINE_H - 2),
                    FONT, SCALE, (230, 230, 230), THICKNESS, cv2.LINE_AA)
    p = lane_info["path_point"]
    cv2.circle(frame, p, 4, (255, 255, 0), -1)
    if obstacle_info["point"] is not None:
        cv2.line(frame, p, obstacle_info["point"], (0, 0, 255), 1)
    return frame

# ══════════════════════════════════════════════════════════════════════════════
# HUD LABEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _label_lateral(e: float) -> str:
    if e < -25: return "far_right"
    if e < -8:  return "slgt_right"
    if e >  25: return "far_left"
    if e >   8: return "slgt_left"
    return "aligned"

def _label_heading(e: float) -> str:
    if e < -18 or e < -6: return "need_right"
    if e >  18 or e >  6: return "need_left"
    return "hdg_aligned"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTROL ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(frame: np.ndarray, car_id: int):
    """
    Execute one full sense–plan–act cycle for the given car.
    Returns (servo, motor, annotated_frame).
    Appends one entry to _log_entries; call save_log() at shutdown.
    """
    global obstacle_positions, _cycle_counter, _car_marker_cache, _last_track_markers, _last_obs_positions
    _cycle_counter += 1
    car_frame = frame.copy()
    now       = time.time()

    # 1. Detect markers
    car_markers, track_markers, obs_markers = detect_all_markers(car_frame)
    obstacle_positions = [marker_center(c) for c in obs_markers.values()]

    # 2. Scene state
    cars  = identify_cars(car_markers, car_id)
    lanes = detect_lanes(track_markers)
    update_tracker(cars, now)

    # Safe defaults
    servo         = 0.0
    motor         = 0.0
    steer_state   = {"lateral": "-", "heading": "-", "obstacle": "-"}
    speed_state   = {"zone": "-", "coop_hint": False}
    lane_info     = {"path_point": (0, 0), "lateral_error": 0.0, "heading_error": 0.0,
                     "tangent": np.array([1.0, 0.0]), "left_normal": np.array([0.0, -1.0])}
    obstacle_info = {"distance": float("inf"), "side_offset": 0.0, "point": None, "state": "clear"}
    current_lane  = lane_state.get(car_id, {}).get("lane", 1)
    target_pt     = None
    policy        = DRIVING_POLICY.get(car_id, DEFAULT_POLICY)
    ref_curve     = lanes.get(f"lane{current_lane}_ref", [])
    if not ref_curve:
    # Hold last known servo, cut speed, wait for re-detection
        return round(servo, 2), round(motor, 2), car_frame
    events: list  = []
    # Log front-fallback mode as a named event
    if _car_marker_cache.get(car_id, {}).get("using_front_fallback", False):
        events.append("front_fallback")

    # 3. Control
    if car_id in cars and ref_curve:
        car_ref = cars[car_id]["midpoint"]

        lane_info     = compute_lane_measurements(car_ref, cars[car_id]["heading"], ref_curve)
        obstacle_info = nearest_relevant_obstacle(car_ref, lane_info["tangent"],
                                                  lane_info["left_normal"], obstacle_positions)

        prev_lane = lane_state.get(car_id, {}).get("lane", 1)

        # A: Where is the car physically right now? (geometry-only, every frame)
        front_pt = cars[car_id]["front"] or car_ref   # fallback to midpoint if frontpoint occluded
        rear_pt  = cars[car_id]["rear"] or car_ref   # fallback to midpoint if rearpoint occluded
        phys_lane = _classify_lane_from_pos(front_pt, rear_pt, lanes, prev_lane)

        # B: Seed lane_state with the physical lane *unless* an obstacle-triggered
        #    overtake is currently active.  This ensures decide_lane() always
        #    reasons from the correct starting position.
        if car_id in lane_state:
            if not lane_state[car_id].get("overtaking", False):
                lane_state[car_id]["lane"] = phys_lane

        # C: Obstacle / coordination logic may override to the adjacent lane.
        current_lane = decide_lane(car_id, car_ref, lanes, obstacle_info, cars, now)

        if current_lane != prev_lane:
            events.append("lane_change")

        # D: Re-route to the correct reference curve for this lane.
        new_ref = lanes.get(f"lane{current_lane}_ref", ref_curve)
        if new_ref is not ref_curve:
            ref_curve     = new_ref
            lane_info     = compute_lane_measurements(
                                car_ref, cars[car_id]["heading"], ref_curve)
            obstacle_info = nearest_relevant_obstacle(
                                car_ref, lane_info["tangent"],
                                lane_info["left_normal"], obstacle_positions)
        if current_lane != prev_lane:
            events.append("lane_change")

        new_ref = lanes.get(f"lane{current_lane}_ref", ref_curve)
        if new_ref is not ref_curve:
            ref_curve     = new_ref
            lane_info     = compute_lane_measurements(car_ref, cars[car_id]["heading"], ref_curve)
            obstacle_info = nearest_relevant_obstacle(car_ref, lane_info["tangent"],
                                                      lane_info["left_normal"], obstacle_positions)

        outer_boundary = lanes.get("middle_curve") if current_lane == 1 else lanes.get("outer_curve")
        raw_delta, _, path_curvature, target_pt = _compute_lookahead(
            cars[car_id]['midpoint'],        # ← midpoint, not the rear marker
            cars[car_id]['heading'],
            ref_curve,
            outer_ref=outer_boundary
        )

        # ── Map radians → servo value [-0.5, 0.5] ────────────────────────────────────
        # δ=0 rad → servo=0.00   |   δ=+π/4 rad (+45°) → servo=+0.50
        # servo_continuous = capped * _RAD_TO_SERVO   # _RAD_TO_SERVO = 4/π ≈ 1.2732
        servo_continuous = raw_delta * _RAD_TO_SERVO   # _RAD_TO_SERVO = 4/π ≈ 1.2732
        servo = float(min(SERVO_STEPS, key=lambda s: abs(s - servo_continuous)))

        steer_state = {
            "lateral":  _label_lateral(lane_info["lateral_error"]),
            "heading":  _label_heading(lane_info["heading_error"]),
            "obstacle": obstacle_info["state"],
        }

        # AFTER — use curvature already computed by _compute_lookahead:
        if path_curvature > 0.3:
            base_speed = SLOW_SPEED
        elif path_curvature > 0.07:
            base_speed = CRUISE_SPEED * 0.85
        else:
            base_speed = CRUISE_SPEED
        coord_speeds = _apply_coordination(cars, base_speed)
        raw_motor    = coord_speeds.get(car_id, base_speed)

        single_lane   = not lanes.get("lane2_ready", False)
        not_overtaking = not lane_state.get(car_id, {}).get("overtaking", False)
        safety_stop    = (obstacle_info["state"] in ("blocking", "near")
                          and not_overtaking
                          and (single_lane or policy == "cooperative"))

        if safety_stop:
            motor = STOP_SPEED
            events.append("safety_stop")
        else:
            motor = float(np.clip(raw_motor, 0.0, MAX_SPEED))

        coop_active = time.time() < coop_slowdown_until.get(car_id, 0.0)
        speed_state = {
            "zone":      "stop" if motor == STOP_SPEED else ("slow" if motor <= SLOW_SPEED else "go"),
            "coop_hint": coop_active,
        }

        last_command_time[car_id] = now

    # 4. Drawing
    car_frame = draw_lanes(car_frame, lanes)
    car_frame = draw_obstacles(car_frame, obstacle_positions)
    car_frame = draw_cars(car_frame, cars)
    # ── Heading arrow: cyan arrow from rear marker in the car's heading direction ──
    if car_id in cars and cars[car_id]['rear'] is not None:
        hv = np.array(cars[car_id]['heading'], dtype=float)
        if np.linalg.norm(hv) > 1e-6:
            hv /= np.linalg.norm(hv)
            origin = cars[car_id]['rear']          # rear axle — same ref as pure pursuit
            arrow_end = (int(origin[0] + hv[0] * 60),
                        int(origin[1] + hv[1] * 60))
            cv2.arrowedLine(car_frame, tuple(origin), arrow_end,
                            (0, 255, 255), 2, tipLength=0.3)
    if car_id in cars and cars[car_id]["midpoint"] is not None and target_pt is not None:
        car_frame = draw_guide_line(car_frame, cars[car_id]["midpoint"], target_pt,
                                    obstacle_info["state"] in ("blocking", "near"))
    lc_st = lane_state.get(car_id, {})
    car_frame = draw_hud(car_frame, car_id, servo, motor, lane_info, obstacle_info,
                         steer_state, speed_state, current_lane, lc_st, policy)

    # 5. Log
    car_data  = cars.get(car_id, {})
    mid_      = car_data.get("midpoint", (0, 0))
    hv        = car_data.get("heading",  (1, 0))
    theta_est = (float(np.degrees(np.arctan2(-hv[1], hv[0])) % 360)
                 if np.linalg.norm(hv) > 1e-6 else 0.0)

    # Accumulate this car's data for the frame-level log entry
    _frame_cars_data.setdefault(_cycle_counter, {})[car_id] = {
        "policy":        policy,
        "pose":          [float(mid_[0]), float(mid_[1]), theta_est],
        "lane":          current_lane,
        "segment":       _classify_segment(mid_, lanes, lane=current_lane)[0],
        "servo":         servo,
        "motor":         motor,
        "waiting":       _pp_waiting.get(car_id, False),
        "lateral_error": lane_info["lateral_error"],
        "heading_error": lane_info["heading_error"],
        "obstacle_info": obstacle_info,
        "events":        events,
    }

    # Flush completed frames (any frame key older than the current one)
    for fk in [k for k in _frame_cars_data if k < _cycle_counter]:
        entry = _build_log_entry(now, fk, _frame_cars_data.pop(fk), cars)
        _log_entries.append(entry)

    return round(servo, 2), round(motor, 2), car_frame


# ─────────────────────────────────────────────────────────────────────────────
# Optional standalone camera test
# ─────────────────────────────────────────────────────────────────────────────
import os, sys

def init_camera(cam_index: int = 0) -> tuple[cv2.VideoCapture, cv2.VideoWriter]:
    """Open the overhead camera (1280×720 @ 60 fps if supported)."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(cam_index, backend)
    if not cap.isOpened():
        import sys; sys.exit(f"Camera {cam_index} not found.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,           60)

    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frameRate = cap.get(cv2.CAP_PROP_FPS)
    print(f"Width : {frameWidth}")
    print(f"Height: {frameHeight}")
    print(f"FPS   : {frameRate}")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.mp4', fourcc, frameRate, (int(frameWidth), int(frameHeight)))
    return cap, out


def boot():
        """Launches car.py in the Pi board"""
        os.system('putty -ssh {}@{} -pw {} -m "./player/launch.txt"'.format(
                username, ip, password))

if __name__ == "__main__":
    import argparse; parser = argparse.ArgumentParser(description="Start the control of the minicar(s)")
    parser.add_argument('-n', '--cars', nargs='+', type=int, default=None,
                        help='Manual cars: input the ID of each one')
    args = parser.parse_args()

    if args.cars is None:
        print('No minicar selected!')
        sys.exit(0)

    ip = f"192.168.0.201"
    username = 'cpslab1'
    password = 'cpslab1'
    print(ip)

    remote_port = 6789
    response = os.system('ping -n 1 -w 200 {} | find "Reply"'.format(ip))
    if response == 0:
        import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        import struct, keyboard
        from threading import Thread
        
        th = Thread(target=boot)
        th.start()
        
        car_idx = 1
        cap, recorder     = init_camera(0)
        running = True
        clean = False

        while running:
            ret, frame = cap.read()
            if ret and frame is not None:
                servo, motor, vis = run(frame, car_idx)
                if keyboard.is_pressed("esc"):
                    running = False
                    clean = True
                # Send actuation commands at 100 Hz
                buffer = bytearray(
                    struct.pack('<fff?',
                                motor,
                                servo,
                                0,
                                clean))
                s.sendto(buffer, (ip, remote_port))
                print(f"Actual send values for motor speed {motor} and servo angle {servo}\r\n")
                # time.sleep(1. / 100.)
                cv2.imshow("Lane Controller", vis)
                # recorder.write(vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cap.release()
        # recorder.release()
        s.close()
        save_log()
        cv2.destroyAllWindows()
    else:
        print("No connection")
    sys.exit(0)
