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
CRUISE_SPEED = 0.55
SLOW_SPEED   = 0.45
STOP_SPEED   = 0.00

# ── Servo limits (radians, sent as normalised floats) ─────────────────────────
MAX_SERVO       = 0.50   # maximum steering angle magnitude
MAX_MID_SERVO   = 0.35
MED_SERVO       = 0.22   # medium steering step
MED_SMALL_SERVO = 0.12
SMALL_SERVO     = 0.05   # small steering step

# ── Discrete output sets ──────────────────────────────────────────────────────
SERVO_STEPS = [MAX_SERVO, MAX_MID_SERVO, MED_SERVO, MED_SMALL_SERVO, SMALL_SERVO,
               0.0,
               -MAX_SERVO, -MAX_MID_SERVO, -MED_SERVO, -MED_SMALL_SERVO, -SMALL_SERVO]
"""Allowed quantised servo values. Negative = steer right, positive = steer left.
Continuous pure-pursuit output is snapped to the nearest step."""

MOTOR_STEPS = [STOP_SPEED, SLOW_SPEED, CRUISE_SPEED, MAX_SPEED]
"""Allowed quantised motor values. Continuous speed is snapped to nearest step."""

# ── Pure-pursuit (bicycle model) parameters ───────────────────────────────────
WHEELBASE_PX = 120
DELTA_MAX    = MAX_SERVO
LOOKAHEAD_DIST   = 150        # reduced: ~1.3× wheelbase keeps car inside lane
MIN_LOOKAHEAD    = 95         # floor: never shorter than this (px)
K_CURVATURE_LD   = 300.0      # how much to shrink ld per unit curvature (px per 1/px)
K_BOUNDARY_PUSH  = 0.012      # rad/px: steering correction per pixel of boundary overshoot

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

# ── Lightweight logger state ──────────────────────────────────────────────────
_log_entries:  list = []   # one dict per frame; flushed to JSON at shutdown
_cycle_counter: int = 0

# ══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def _build_log_entry(
    t:             float,
    k:             int,
    car_id:        int,
    policy:        str,
    pose:          list,
    lane:          int,
    segment:       str,
    curvature:     float,
    servo:         float,
    motor:         float,
    waiting:       bool,
    lateral_error: float,
    heading_error: float,
    obstacle_info: dict,
    cars:          dict,
    events:        list,
) -> dict:
    """
    Structured JSON-compatible log entry.

    Schema
    ------
    {
      "t":       float,
      "k":       int,
      "car_id":  int,
      "policy":  str,
      "pose":    [x_px, y_px, theta_deg],
      "lane":    int,
      "segment": str,
      "curvature": float,
      "command": {"servo": float, "motor": float},
      "waiting": bool,
      "lateral_error":  float,
      "heading_error":  float,
      "obstacle": {"state": str, "distance_px": float},
      "distances": {"<a>-<b>": float},
      "events":    [str]
    }
    """
    # Pairwise distances between all visible cars
    ids = sorted(cars.keys())
    distances = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pa = np.array(cars[a]["midpoint"], dtype=float)
            pb = np.array(cars[b]["midpoint"], dtype=float)
            distances[f"{a}-{b}"] = round(float(np.linalg.norm(pa - pb)), 2)

    obs_d = obstacle_info["distance"]
    return {
        "t":      round(t, 5),
        "k":      k,
        "car_id": car_id,
        "policy": policy,
        "pose":   [round(pose[0], 2), round(pose[1], 2), round(pose[2], 2)],
        "lane":   lane,
        "segment": segment,
        "curvature": curvature,
        "command": {
            "servo": round(servo, 3),
            "motor": round(motor, 3),
        },
        "waiting": waiting,
        "lateral_error": round(lateral_error, 2),
        "heading_error": round(heading_error, 2),
        "obstacle": {
            "state":       obstacle_info["state"],
            "distance_px": round(obs_d, 2) if obs_d < float("inf") else None,
        },
        "distances": distances,
        "events":    events,
    }


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
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    return pts[np.argsort(angles)]

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
                cx: float = None, cy: float = None) -> list:
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
    result = {}
    inner_pts  = [marker_center(track_markers[i]) for i in INNER_SET  if i in track_markers]
    middle_pts = [marker_center(track_markers[i]) for i in MIDDLE_SET if i in track_markers]
    outer_pts  = [marker_center(track_markers[i]) for i in OUTER_SET  if i in track_markers]
    result.update({"n_inner": len(inner_pts), "n_middle": len(middle_pts), "n_outer": len(outer_pts)})

    all_pts = inner_pts + middle_pts + outer_pts
    if len(all_pts) < 3:
        return result

    all_arr = np.array(all_pts, dtype=float)
    cx, cy  = float(all_arr[:, 0].mean()), float(all_arr[:, 1].mean())

    # Stadium fitting: markers are tangent corners of the pill shape.
    # n= is explicit so all three rings always have identical array lengths.
    if len(inner_pts)  >= 2: result["inner_curve"]  = fit_stadium(inner_pts,  n=STADIUM_SAMPLES, cx=cx, cy=cy)
    if len(middle_pts) >= 2: result["middle_curve"] = fit_stadium(middle_pts, n=STADIUM_SAMPLES, cx=cx, cy=cy)
    if len(outer_pts)  >= 2: result["outer_curve"]  = fit_stadium(outer_pts,  n=STADIUM_SAMPLES, cx=cx, cy=cy)

    # Lane 1 tracks the inner boundary; lane 2 tracks the middle boundary.
    # Using the boundary lines directly (rather than averaged centrelines)
    # gives a crisper reference with geometry consistent with the markers.
    if "inner_curve" in result and "middle_curve" in result:
        # Build the true lane-1 centreline: element-wise midpoint of inner
        # and middle boundary curves.  Both curves have exactly STADIUM_SAMPLES
        # points so the zip is always aligned.
        inner_arr  = np.array(result["inner_curve"],  dtype=float)
        middle_arr = np.array(result["middle_curve"], dtype=float)
        centre_arr = ((inner_arr + middle_arr) / 2.0).astype(int)
        result["lane1_centre"] = list(zip(centre_arr[:, 0].tolist(),
                                          centre_arr[:, 1].tolist()))
        # Use the centreline as the driving reference so the car stays
        # equidistant from both the inner and the middle boundaries.
        result["lane1_ref"]   = result["lane1_centre"]
        result["lane1_ready"] = True

    if "middle_curve" in result and "outer_curve" in result:
        middle_arr2 = np.array(result["middle_curve"], dtype=float)
        outer_arr   = np.array(result["outer_curve"],  dtype=float)
        centre2_arr = ((middle_arr2 + outer_arr) / 2.0).astype(int)
        result["lane2_centre"] = list(zip(centre2_arr[:, 0].tolist(),
                                          centre2_arr[:, 1].tolist()))
        result["lane2_ref"]   = result["lane2_centre"]
        result["lane2_ready"] = True

    return result

def _ensure_winding(curve: list, heading_vec: tuple, car_pos: tuple) -> list:
    """Reverse the curve array if it winds against the car's travel direction.

    If fit_stadium() produces an array whose index order runs opposite to the
    car's heading, the lookahead walk steps backwards and the steering sign
    flips — causing the observed opposite-direction correction near the
    semicircle entries (markers 0/4 coming from 3/7).
    """
    if not curve or np.linalg.norm(heading_vec) < 1e-4:
        return curve
    ni = project_onto_curve(car_pos, curve)
    n  = len(curve)
    t_vec = (np.array(curve[(ni + 1) % n], dtype=float)
             - np.array(curve[(ni - 1) % n], dtype=float))
    h_vec = np.array(heading_vec, dtype=float)
    if np.dot(t_vec, h_vec) < 0:
        return list(reversed(curve))
    return curve

# ══════════════════════════════════════════════════════════════════════════════
# MARKER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_all_markers(frame: np.ndarray):
    """Detect all three ArUco marker types in one camera frame.

    Runs three independent ArucoDetector passes on the same greyscale image,
    one per dictionary. Car and track markers are drawn in the default colour;
    obstacle markers are drawn in red so they are visually distinct.

    Parameters
    ----------
    frame : np.ndarray (BGR)
        Camera frame. Marker outlines are drawn onto this frame in-place.

    Returns
    -------
    car_markers   : dict {int: corner_array}  — 4×4 dictionary detections
    track_markers : dict {int: corner_array}  — 5×5 dictionary detections
    obs_markers   : dict {int: corner_array}  — 6×6 dictionary detections
    """
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

    car_markers   = _detect(CAR_DICT)
    track_markers = _detect(TRACK_DICT)
    obs_markers   = _detect(OBSTACLE_DICT, (0, 0, 255))
    return car_markers, track_markers, obs_markers

# ══════════════════════════════════════════════════════════════════════════════
# CAR IDENTIFICATION & SPEED TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def identify_cars(car_markers: dict, car_id: int) -> dict:
    """Build a structured pose dict for every visible car from raw marker data.

    The front marker (ID = FRONT_MARKER_ID = 49) is shared by all cars and
    always belongs to *the currently tracked car* (car_id). All other detected
    4×4 IDs are treated as rear markers, where the ID itself is the car index.

    For the tracked car the heading vector, front position, and midpoint are
    computed. For other cars only the rear position is known (no front marker
    distinction), so heading is (0, 0) and front is None.

    Parameters
    ----------
    car_markers : dict {int: corner_array}
    car_id      : int — the ID of the car being actively controlled.

    Returns
    -------
    dict {car_id: {'front', 'rear', 'midpoint', 'heading'}}
    """
    front_center = (marker_center(car_markers[FRONT_MARKER_ID])
                    if FRONT_MARKER_ID in car_markers else None)
    cars = {}
    for mid, corner in car_markers.items():
        if mid == FRONT_MARKER_ID:
            continue
        rear_center = marker_center(corner)
        if mid == car_id and front_center is not None:
            mid_pt  = midpoint(front_center, rear_center)
            heading = (front_center[0] - rear_center[0], front_center[1] - rear_center[1])
            front   = front_center
        else:
            mid_pt  = rear_center
            heading = (0, 0)
            front   = None
        cars[mid] = {"front": front, "rear": rear_center, "midpoint": mid_pt, "heading": heading}
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
    left_normal = np.array([-tangent[1], tangent[0]])
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

def _compute_lookahead(car_pos, heading_vec, lane_ref, outer_ref=None) -> tuple:
    """Compute the pure-pursuit steering angle using the bicycle kinematic model.

    Algorithm
    ─────────
    1. Find the nearest point on lane_ref to car_pos.
    2. Walk forward along the curve until LOOKAHEAD_DIST pixels of arc length
       have been accumulated. Interpolate the exact lookahead point linearly
       within the last segment.
    3. Compute the bearing α from the car to the lookahead point, relative to
       the car's current heading θ.
    4. Apply the pure-pursuit formula:
           δ = atan2(2 · WHEELBASE_PX · sin(α), l_d)
       where l_d = Euclidean distance to the lookahead point.
    5. Clip δ to ±DELTA_MAX and return alongside the lookahead pixel position.

    Parameters
    ----------
    car_pos     : (x, y) pixel reference position (rear marker centre = rear axle).
    heading_vec : (dx, dy) direction vector (front − rear marker positions).
    lane_ref    : list of (x, y) — active lane reference polyline.

    Returns
    -------
    (delta_rad : float, lookahead_point : tuple)
    """
    arr = np.array(lane_ref, dtype=float)
    pos = np.array(car_pos, dtype=float)
    ni  = int(np.argmin(np.linalg.norm(arr - pos, axis=1)))
    n   = len(lane_ref)

       # ── Option A: adaptive lookahead ─────────────────────────────────────────
    # Estimate local curvature at the nearest curve point (3-point stencil).
    p_prev = arr[(ni - 2) % n]
    p_curr = arr[ni]
    p_next = arr[(ni + 2) % n]
    v1, v2 = p_curr - p_prev, p_next - p_curr
    cross   = abs(float(np.cross(v1, v2)))
    seg_len = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + 1e-9
    curvature = cross / (seg_len ** 2)
    # Shrink lookahead on tighter curves so the car can react in time.
    ld_dist = max(MIN_LOOKAHEAD, LOOKAHEAD_DIST - K_CURVATURE_LD * curvature)

    # ── Walk the curve to find the raw lookahead target ───────────────────────
    accum, target = 0.0, arr[ni]
    for i in range(ni, ni + n - 1):
        i0  = i % n
        i1  = (i + 1) % n
        seg = np.linalg.norm(arr[i1] - arr[i0])
        if accum + seg >= ld_dist:
            frac   = (ld_dist - accum) / (seg + 1e-9)
            target = arr[i0] + frac * (arr[i1] - arr[i0])
            break
        accum += seg

    # ── Pure-pursuit formula ──────────────────────────────────────────────────
    # (Option B lateral-correction removed: lane_ref is now the true lane
    #  centreline so no perpendicular shift is needed or safe.)
    hx, hy = heading_vec
    theta  = math.atan2(hy, hx) if math.hypot(hx, hy) > 1e-4 else 0.0
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    alpha  = math.atan2(math.sin(math.atan2(dy, dx) - theta),
                        math.cos(math.atan2(dy, dx) - theta))
    ld     = math.hypot(dx, dy) + 1e-9
    delta  = math.atan2(2.0 * WHEELBASE_PX * math.sin(alpha), ld)

    # ── Boundary-push: if car drifts past the outer ref (middle curve),
    #    add a proportional corrective steering term to push it back in.
    if outer_ref is not None:
        outer_arr = np.array(outer_ref, dtype=float)
        oi = int(np.argmin(np.linalg.norm(outer_arr - pos, axis=1)))
        op = outer_arr[oi]
        ot = outer_arr[(oi + 1) % len(outer_arr)] - outer_arr[(oi - 1) % len(outer_arr)]
        ot = ot / (np.linalg.norm(ot) + 1e-9)
        on = np.array([-ot[1], ot[0]])             # outward normal of middle boundary
        # Signed distance: positive = car is outside the boundary (bad side)
        overshoot = float(np.dot(pos - op, on))
        if overshoot > 0:                          # car crossed the outer boundary
            # Steer back: add correction proportional to overshoot distance
            delta -= K_BOUNDARY_PUSH * overshoot

    return float(np.clip(delta, -DELTA_MAX, DELTA_MAX)), tuple(target.astype(int))

# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED COORDINATION
# ══════════════════════════════════════════════════════════════════════════════

def _apply_coordination(cars, base_speed) -> dict:
    """Apply pairwise distance-threshold coordination to all detected cars.

    For every pair of cars, measures the Euclidean pixel distance between their
    midpoints. If one car is within D_SAFE or D_WARN of another, the higher-ID
    car (the 'yielder') has its speed reduced. Cars with policy 'non_cooperative'
    are excluded from yielding.

    Thresholds (from methodology §Coordination Logic):
        d ≤ D_SAFE : yielder stops entirely.
        D_SAFE < d ≤ D_WARN : yielder slows to SLOW_SPEED.
        d > D_WARN : yielder continues at base_speed.

    Parameters
    ----------
    cars       : dict — output of identify_cars().
    base_speed : float — speed before coordination (from _apply_curve_slowdown).

    Returns
    -------
    dict {car_id: speed} — per-car speed after coordination rules are applied.
    """
    speeds = {cid: base_speed for cid in cars}
    ids    = sorted(cars.keys())
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1:]:
            pa = np.array(cars[id_a]["midpoint"], dtype=float)
            pb = np.array(cars[id_b]["midpoint"], dtype=float)
            d  = float(np.linalg.norm(pa - pb))
            yielder_id = max(id_a, id_b)
            if DRIVING_POLICY.get(yielder_id, DEFAULT_POLICY) == "non_cooperative":
                continue
            if d <= D_SAFE:
                speeds[yielder_id]        = STOP_SPEED
                _pp_waiting[yielder_id]   = True
            elif d <= D_WARN:
                speeds[yielder_id]        = min(speeds[yielder_id], SLOW_SPEED)
                _pp_waiting[yielder_id]   = True
            else:
                _pp_waiting[yielder_id]   = False
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
    cross    = abs(float(np.cross(v1, v2)))
    seg_len  = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + 1e-9
    curvature = cross / (seg_len ** 2)
    print(f"curvature: {curvature}\n")
    if curvature > 0.3:
        return "sharp_curve", round(curvature,3)
    if curvature > 0.08:
        return "light_curve", round(curvature,3)
    return "straight", round(curvature,3)


def _apply_curve_slowdown(car_pos: tuple, lanes: dict, lane: int = 1) -> float:
    """Return the target speed based on local curvature of the active lane's path.

    Uses the same three-point curvature stencil as _classify_segment() but on
    the reference path for *lane* (1 = inner boundary, 2 = middle boundary),
    so the speed decision matches the actual path the car is following.

    Parameters
    ----------
    car_pos : (x, y) pixel reference position.
    lanes   : output of detect_lanes().
    lane    : active lane number (1 or 2).
    """
    ref = lanes.get(f"lane{lane}_ref", [])
    if not ref or len(ref) < 5:
        return CRUISE_SPEED
    n   = len(ref)
    idx = project_onto_curve(car_pos, ref)
    p0  = np.array(ref[(idx - 2) % n], dtype=float)
    p1  = np.array(ref[idx],            dtype=float)
    p2  = np.array(ref[(idx + 2) % n], dtype=float)
    v1, v2    = p1 - p0, p2 - p1
    cross     = abs(float(np.cross(v1, v2)))
    seg_len   = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + 1e-9
    curvature = cross / (seg_len ** 2)
    if curvature > 0.3:
        return SLOW_SPEED
    if curvature > 0.08:
        return CRUISE_SPEED * 0.85
    return CRUISE_SPEED 

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
    cv2.line(frame, ref_pose, target, color, 2)
    cv2.circle(frame, target, 9, color, 2)
    cv2.circle(frame, target, 3, color, -1)
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
        Red    – rear marker
        Orange – midpoint (vehicle reference position)
        Green  – front marker
        Magenta line – rear-to-front axis (shows heading)
    """
    for _, cd in cars.items():
        cv2.circle(frame, cd["rear"], 8, (0, 0, 255), -1)
        cv2.circle(frame, cd["midpoint"], 6, (0, 165, 255), -1)
        if cd["front"] is not None:
            cv2.circle(frame, cd["front"], 8, (0, 255, 0), -1)
            cv2.line(frame, cd["rear"], cd["front"], (255, 0, 255), 2)
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
    wait_tag = " [W]" if _pp_waiting.get(car_id, False) else ""
    hud_lines = [
        f"Car {car_id} | Ln{current_lane} {lc_tag} | {policy[:4].upper()}{wait_tag}",
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
    global obstacle_positions, _cycle_counter
    _cycle_counter += 1
    car_frame = frame.copy()
    now       = time.time()

    # 1. Detect markers
    car_markers, track_markers, obs_markers = detect_all_markers(car_frame)
    obstacle_positions = [marker_center(c) for c in obs_markers.values()]

    # 2. Scene state
    cars  = identify_cars(car_markers, car_id)
    lanes = detect_lanes(track_markers)
    # Reverse any curve whose winding opposes the car's heading.
    # Prevents the lookahead walk from stepping backwards near markers 0/4.
    if car_id in cars and cars[car_id]["heading"] != (0, 0):
        _hv = cars[car_id]["heading"]
        _cr = cars[car_id]["rear"]
        for _key in ("lane1_ref", "lane2_ref", "inner_curve",
                     "middle_curve", "outer_curve",
                     "lane1_centre", "lane2_centre"):
            if _key in lanes:
                lanes[_key] = _ensure_winding(lanes[_key], _hv, _cr)
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

    # 3. Control
    if car_id in cars and ref_curve:
        car_ref = cars[car_id]["rear"]

        lane_info     = compute_lane_measurements(car_ref, cars[car_id]["heading"], ref_curve)
        obstacle_info = nearest_relevant_obstacle(car_ref, lane_info["tangent"],
                                                  lane_info["left_normal"], obstacle_positions)

        prev_lane    = lane_state.get(car_id, {}).get("lane", 1)
        current_lane = decide_lane(car_id, car_ref, lanes, obstacle_info, cars, now)
        if current_lane != prev_lane:
            events.append("lane_change")

        new_ref = lanes.get(f"lane{current_lane}_ref", ref_curve)
        if new_ref is not ref_curve:
            ref_curve     = new_ref
            lane_info     = compute_lane_measurements(car_ref, cars[car_id]["heading"], ref_curve)
            obstacle_info = nearest_relevant_obstacle(car_ref, lane_info["tangent"],
                                                      lane_info["left_normal"], obstacle_positions)

        outer_boundary = lanes.get("middle_curve") if current_lane == 1 else lanes.get("outer_curve")
        raw_delta, target_pt = _compute_lookahead(car_ref, cars[car_id]["heading"],
                                                   ref_curve, outer_ref=outer_boundary)
        
        # Segment-aware servo cap + snap to nearest SERVO_STEPS value
        seg_result = _classify_segment(car_ref, lanes, lane=current_lane)
        seg_name   = seg_result[0] if isinstance(seg_result, tuple) else seg_result
        if seg_name == "straight":
            delta_cap = MED_SERVO         # ±0.22 — raised: MED_SMALL_SERVO snapped to 0.0/0.05 and couldn't recover lateral drift
        elif seg_name == "light_curve":
            delta_cap = MED_SERVO         # ±0.22 — gentle arc
        else:                             # sharp_curve
            delta_cap = MAX_MID_SERVO     # ±0.35 — tight semicircle
        capped = float(np.clip(raw_delta, -delta_cap, delta_cap))
        servo  = float(min(SERVO_STEPS, key=lambda s: abs(s - capped)))

        steer_state = {
            "lateral":  _label_lateral(lane_info["lateral_error"]),
            "heading":  _label_heading(lane_info["heading_error"]),
            "obstacle": obstacle_info["state"],
        }

        base_speed   = _apply_curve_slowdown(car_ref, lanes, lane=current_lane)
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
    if car_id in cars and cars[car_id]["rear"] is not None and target_pt is not None:
        car_frame = draw_guide_line(car_frame, cars[car_id]["rear"], target_pt,
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

    entry = _build_log_entry(
        t             = now,
        k             = _cycle_counter,
        car_id        = car_id,
        policy        = policy,
        pose          = [float(mid_[0]), float(mid_[1]), theta_est],
        lane          = current_lane,
        segment       = _classify_segment(mid_, lanes, lane=current_lane)[0],
        curvature     = _classify_segment(mid_, lanes, lane=current_lane)[1],
        servo         = servo,
        motor         = motor,
        waiting       = _pp_waiting.get(car_id, False),
        lateral_error = lane_info["lateral_error"],
        heading_error = lane_info["heading_error"],
        obstacle_info = obstacle_info,
        cars          = cars,
        events        = events,
    )
    _log_entries.append(entry)

    return round(servo, 2), round(motor, 2), car_frame


# ─────────────────────────────────────────────────────────────────────────────
# Optional standalone camera test
# ─────────────────────────────────────────────────────────────────────────────
import os, sys

def init_camera(cam_index: int = 0) -> cv2.VideoCapture:
    """Open the overhead camera (1280×720 @ 60 fps if supported)."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(cam_index, backend)
    if not cap.isOpened():
        import sys; sys.exit(f"Camera {cam_index} not found.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,           60)
    print(f"Width : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS   : {cap.get(cv2.CAP_PROP_FPS)}")
    return cap

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
        cap     = init_camera(0)
        running = True
        clean = False
        # last_motor = 0.0
        # last_servo = 0.0

        while running:
            ret, frame = cap.read()
            if ret and frame is not None:
                servo, motor, vis = run(frame, car_idx)
                # # Only update cache when run() issued a real command
                # if motor != 0.0 or servo != 0.0:
                #     last_motor = motor
                #     last_servo = servo
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
                time.sleep(1. / 100.)
                cv2.imshow("Lane Controller", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cap.release()
        s.close()
        save_log()
        cv2.destroyAllWindows()
    else:
        print("No connection")
    sys.exit(0)
