### Lane-following in a rule-based fuzzy-like logic - version used ###

import cv2
import cv2.aruco as aruco
import numpy as np
import time

# ── Constants ──────────────────────────────────────────────────────────────────
SPEED_THRESHOLD      = 15
STATUS_UPDATE_INTERVAL = 0.1
SCALING_FACTOR       = 10.3

# 4×4 dictionary: car markers
# ID 49 = front marker of the controlled car; other IDs = rear markers
CAR_DICT        = aruco.DICT_4X4_50
FRONT_MARKER_ID = 49

MAX_SPEED    = 0.70
CRUISE_SPEED = 0.62
SLOW_SPEED   = 0.55
STOP_SPEED   = 0.00

MAX_SERVO    = 0.50
MED_SERVO    = 0.25
SMALL_SERVO  = 0.10

SERVO_SMOOTHING = 0.35
MOTOR_SMOOTHING = 0.30

LEADER_ID = 0

# ── Track boundary markers (5×5 ArUco) ────────────────────────────────────────
# Three ordered sets define two drivable lanes:
#   Lane 1 = inner  ↔ middle   (reference line = midpoint between the two curves)
#   Lane 2 = middle ↔ outer    (reference line = midpoint between the two curves)
TRACK_DICT  = aruco.DICT_5X5_50
INNER_SET   = [0, 1, 2]
MIDDLE_SET  = [3, 4, 5]
OUTER_SET   = [6, 7, 8]
TRIOVAL_SAMPLES = 220

# ── Static obstacle markers (6×6 ArUco) ───────────────────────────────────────
OBSTACLE_DICT             = aruco.DICT_6X6_250
OBSTACLE_RADIUS           = 45
OBSTACLE_TRACK_HALF_WIDTH = 70
OBSTACLE_LOOKAHEAD        = 180

# ── Guide-line lookahead ───────────────────────────────────────────────────────
LOOKAHEAD_DIST = 90   # pixels ahead on the reference curve

# ── Lane-change hysteresis ─────────────────────────────────────────────────────
LANE_CHANGE_HOLD = 2.5   # minimum seconds to stay in the newly selected lane

# ── Platoon target gap ─────────────────────────────────────────────────────────
TARGET_GAP = 120
GAP_TOL    = 12
RED_GAP    = 85
FAR_GAP    = 165

# ── Fuzzy thresholds ───────────────────────────────────────────────────────────
LATERAL_ALIGNED = 8
LATERAL_SLIGHT  = 25
LATERAL_FAR     = 45

HEADING_ALIGNED = 6
HEADING_SLIGHT  = 18
HEADING_FAR     = 35

TREND_STEADY    = 4
TREND_SHRINKING = -10
TREND_OPENING   = 10

# ── Visualization colors (BGR) ─────────────────────────────────────────────────
LANE1_FILL_COLOR  = (0,  160,  60)    # green  – between inner & middle
LANE2_FILL_COLOR  = (0,  100, 180)    # blue   – between middle & outer
INNER_LINE_COLOR  = (0,  230, 120)
MIDDLE_LINE_COLOR = (255, 255,   0)
OUTER_LINE_COLOR  = (80,  200, 255)
REF_LINE_COLOR    = (200, 200, 200)   # light-gray dashes for reference

# ── Global state ───────────────────────────────────────────────────────────────
tracker            = {}
last_sent_angles   = {}
last_sent_motors   = {}
platoon_metrics    = {}
obstacle_positions = []
lane_state         = {}   # {car_id: {'lane': 1|2, 'timer': float, 'overtaking': bool}}


# ══════════════════════════════════════════════════════════════════════════════
#  Basic helpers
# ══════════════════════════════════════════════════════════════════════════════

def marker_center(corner) -> tuple:
    return tuple(np.mean(corner[0], axis=0).astype(int))

def midpoint(p1: tuple, p2: tuple) -> tuple:
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def wrap_angle_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def heading_to_angle(heading_vec: tuple) -> float:
    return float(np.degrees(np.arctan2(-heading_vec[1], heading_vec[0])) % 360)

def smooth_value(prev: float, new: float, alpha: float, safety_stop=False) -> float:
    if safety_stop and new <= 0.0:
        return 0.0
    return (1.0 - alpha) * prev + alpha * new

def dominant_state(state_dict: dict) -> str:
    if not state_dict:
        return "-"
    return max(state_dict.items(), key=lambda kv: kv[1])[0]

def weighted_action(contributions: list, default_value: float = 0.0) -> float:
    num, den = 0.0, 0.0
    for degree, value in contributions:
        if degree > 0:
            num += degree * value
            den += degree
    return default_value if den <= 1e-9 else num / den


# ══════════════════════════════════════════════════════════════════════════════
#  Fuzzy membership helpers
# ══════════════════════════════════════════════════════════════════════════════

def tri(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    return (x - a) / (b - a + 1e-9) if x < b else (c - x) / (c - b + 1e-9)

def left_shoulder(x, a, b):
    if x <= a: return 1.0
    if x >= b: return 0.0
    return (b - x) / (b - a + 1e-9)

def right_shoulder(x, a, b):
    if x <= a: return 0.0
    if x >= b: return 1.0
    return (x - a) / (b - a + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
#  Tri-oval polar fitting
# ══════════════════════════════════════════════════════════════════════════════

def fit_trioval_polar(marker_positions: list,
                      n: int = TRIOVAL_SAMPLES,
                      cx: float = None,
                      cy: float = None) -> list:
    """
    Fit a smooth closed curve through `marker_positions` using a polar
    Fourier basis (DC + 1st + 2nd harmonics).
    `cx`, `cy` is the reference centroid; pass a shared global centroid so
    that different sets are consistently parameterized by the same angle axis,
    enabling point-wise averaging for the reference (mid) curves.
    """
    pts = np.array(marker_positions, dtype=float)
    if cx is None or cy is None:
        cx, cy = pts.mean(axis=0)

    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    angles = np.arctan2(dy, dx)
    radii  = np.hypot(dx, dy)

    A = np.column_stack([
        np.ones_like(angles),
        np.cos(angles), np.sin(angles),
        np.cos(2 * angles), np.sin(2 * angles),
    ])
    coeffs, *_ = np.linalg.lstsq(A, radii, rcond=None)
    R0, a1, b1, a2, b2 = coeffs

    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.maximum(
        R0 + a1 * np.cos(th) + b1 * np.sin(th)
           + a2 * np.cos(2 * th) + b2 * np.sin(2 * th),
        1.0
    )
    xs = (cx + r * np.cos(th)).astype(int)
    ys = (cy + r * np.sin(th)).astype(int)
    return list(zip(xs.tolist(), ys.tolist()))


def detect_lanes(track_markers: dict) -> dict:
    """
    Fit closed curves for inner / middle / outer marker sets.
    Also build reference (mid) curves for each lane and report readiness.

    Returned keys:
      inner_curve, middle_curve, outer_curve  – boundary polylines
      lane1_ref, lane2_ref                    – drivable reference lines
      lane1_ready, lane2_ready                – booleans
    """
    result = {}

    inner_pts  = [marker_center(track_markers[i]) for i in INNER_SET  if i in track_markers]
    middle_pts = [marker_center(track_markers[i]) for i in MIDDLE_SET if i in track_markers]
    outer_pts  = [marker_center(track_markers[i]) for i in OUTER_SET  if i in track_markers]

    result.update({'n_inner': len(inner_pts),
                   'n_middle': len(middle_pts),
                   'n_outer': len(outer_pts)})

    all_pts = inner_pts + middle_pts + outer_pts
    if len(all_pts) < 3:
        return result

    # Single global centroid so curves share the same angular parameterization
    all_arr = np.array(all_pts, dtype=float)
    cx, cy  = float(all_arr[:, 0].mean()), float(all_arr[:, 1].mean())

    if len(inner_pts) >= 2:
        result['inner_curve']  = fit_trioval_polar(inner_pts,  cx=cx, cy=cy)
    if len(middle_pts) >= 2:
        result['middle_curve'] = fit_trioval_polar(middle_pts, cx=cx, cy=cy)
    if len(outer_pts) >= 2:
        result['outer_curve']  = fit_trioval_polar(outer_pts,  cx=cx, cy=cy)

    # Lane 1 reference: midpoint between inner and middle curves
    if 'inner_curve' in result and 'middle_curve' in result:
        inner_arr  = np.array(result['inner_curve'],  dtype=float)
        middle_arr = np.array(result['middle_curve'], dtype=float)
        ref1 = ((inner_arr + middle_arr) / 2.0).astype(int)
        result['lane1_ref']   = [tuple(p) for p in ref1.tolist()]
        result['lane1_ready'] = True

    # Lane 2 reference: midpoint between middle and outer curves
    if 'middle_curve' in result and 'outer_curve' in result:
        middle_arr = np.array(result['middle_curve'], dtype=float)
        outer_arr  = np.array(result['outer_curve'],  dtype=float)
        ref2 = ((middle_arr + outer_arr) / 2.0).astype(int)
        result['lane2_ref']   = [tuple(p) for p in ref2.tolist()]
        result['lane2_ready'] = True

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Marker detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_all_markers(frame: np.ndarray):
    """
    Returns:
      car_markers   – 4×4 dict  (ID 49 = front; other IDs = rear markers)
      track_markers – 5×5 dict  (IDs 0-8: inner/middle/outer boundary)
      obs_markers   – 6×6 dict  (static obstacles)
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
#  Car geometry
# ══════════════════════════════════════════════════════════════════════════════

def identify_cars(car_markers: dict, car_id: int) -> dict:
    """
    ID FRONT_MARKER_ID (49) → front marker of the controlled car.
    All other IDs             → rear marker of the car with that ID.
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
            heading = (front_center[0] - rear_center[0],
                       front_center[1] - rear_center[1])
            front   = front_center
        else:
            mid_pt  = rear_center
            heading = (0, 0)
            front   = None

        cars[mid] = {
            'front':    front,
            'rear':     rear_center,
            'midpoint': mid_pt,
            'heading':  heading,
        }
    return cars


def estimate_speed(curr_pos, prev_pos, dt: float) -> float:
    if prev_pos is None or dt <= 0:
        return 0.0
    return float(np.linalg.norm(
        np.array(curr_pos) - np.array(prev_pos)) / (dt * SCALING_FACTOR))


def update_tracker(cars: dict, current_time: float) -> None:
    for rear_id, cd in cars.items():
        if rear_id not in tracker:
            tracker[rear_id] = {
                'status': None, 'center': None,
                'last_time': None, 'heading': None, 'speed': 0.0,
            }
        t = tracker[rear_id]
        last_time = t['last_time']
        if last_time is None or (current_time - last_time >= STATUS_UPDATE_INTERVAL):
            dt    = (current_time - last_time) if last_time else 0.0
            speed = estimate_speed(cd['midpoint'], t['center'], dt)
            t['speed']     = speed
            t['status']    = "moving" if speed > SPEED_THRESHOLD else "stopped"
            t['center']    = cd['midpoint']
            t['last_time'] = current_time
            t['heading']   = cd['heading']


# ══════════════════════════════════════════════════════════════════════════════
#  Path-relative measurements
# ══════════════════════════════════════════════════════════════════════════════

def project_onto_curve(pos: tuple, curve_pts: list) -> int:
    arr = np.array(curve_pts, dtype=float)
    return int(np.argmin(np.linalg.norm(arr - np.array(pos, dtype=float), axis=1)))


def local_path_frame(curve_pts: list, idx: int):
    n      = len(curve_pts)
    p_prev = np.array(curve_pts[(idx - 1) % n], dtype=float)
    p_curr = np.array(curve_pts[idx],            dtype=float)
    p_next = np.array(curve_pts[(idx + 1) % n],  dtype=float)

    tangent = p_next - p_prev
    norm_t  = np.linalg.norm(tangent)
    if norm_t < 1e-6:
        tangent = np.array([1.0, 0.0])
        norm_t  = 1.0
    tangent     /= norm_t
    left_normal  = np.array([-tangent[1], tangent[0]])
    return p_curr, tangent, left_normal


def compute_lane_measurements(car_pos: tuple, heading_vec: tuple, curve_pts: list):
    idx           = project_onto_curve(car_pos, curve_pts)
    p_ref, tangent, left_normal = local_path_frame(curve_pts, idx)

    pos_vec       = np.array(car_pos, dtype=float) - p_ref
    lateral_error = float(np.dot(pos_vec, left_normal))

    tangent_angle = float(np.degrees(np.arctan2(-tangent[1], tangent[0])) % 360)
    car_heading_angle = (tangent_angle
                         if np.linalg.norm(np.array(heading_vec, dtype=float)) < 1e-6
                         else heading_to_angle(heading_vec))
    heading_error = wrap_angle_deg(tangent_angle - car_heading_angle)

    return {
        'idx':              idx,
        'path_point':       tuple(p_ref.astype(int)),
        'tangent':          tangent,
        'left_normal':      left_normal,
        'lateral_error':    lateral_error,
        'heading_error':    heading_error,
        'tangent_angle':    tangent_angle,
        'car_heading_angle': car_heading_angle,
    }


def find_lookahead_point(curve_pts: list, start_idx: int, dist: float) -> tuple:
    """Walk along the curve from start_idx until cumulative arc length >= dist."""
    n     = len(curve_pts)
    accum = 0.0
    idx   = start_idx
    for _ in range(n):
        nxt   = (idx + 1) % n
        step  = float(np.linalg.norm(
            np.array(curve_pts[nxt], dtype=float) -
            np.array(curve_pts[idx], dtype=float)))
        accum += step
        idx    = nxt
        if accum >= dist:
            break
    return curve_pts[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  Fuzzy state extraction
# ══════════════════════════════════════════════════════════════════════════════

def fuzzify_lateral(e: float) -> dict:
    return {
        'far_right':     left_shoulder(e, -LATERAL_FAR, -LATERAL_SLIGHT),
        'slightly_right': tri(e, -LATERAL_FAR, -LATERAL_SLIGHT, -LATERAL_ALIGNED),
        'aligned':        tri(e, -LATERAL_ALIGNED, 0, LATERAL_ALIGNED),
        'slightly_left':  tri(e, LATERAL_ALIGNED, LATERAL_SLIGHT, LATERAL_FAR),
        'far_left':       right_shoulder(e, LATERAL_SLIGHT, LATERAL_FAR),
    }

def fuzzify_heading(e: float) -> dict:
    return {
        'need_right_hard': left_shoulder(e, -HEADING_FAR, -HEADING_SLIGHT),
        'need_right':       tri(e, -HEADING_FAR, -HEADING_SLIGHT, -HEADING_ALIGNED),
        'heading_aligned':  tri(e, -HEADING_ALIGNED, 0, HEADING_ALIGNED),
        'need_left':        tri(e, HEADING_ALIGNED, HEADING_SLIGHT, HEADING_FAR),
        'need_left_hard':   right_shoulder(e, HEADING_SLIGHT, HEADING_FAR),
    }

def fuzzify_gap(gap: float) -> dict:
    return {
        'too_close': left_shoulder(gap, RED_GAP - 10, RED_GAP + 10),
        'close':     tri(gap, RED_GAP, TARGET_GAP - GAP_TOL, TARGET_GAP),
        'ideal':     tri(gap, TARGET_GAP - GAP_TOL - 8, TARGET_GAP, TARGET_GAP + GAP_TOL + 8),
        'far':       right_shoulder(gap, TARGET_GAP + GAP_TOL, FAR_GAP),
    }

def fuzzify_gap_trend(gap_rate: float) -> dict:
    return {
        'shrinking': left_shoulder(gap_rate, TREND_SHRINKING, -TREND_STEADY),
        'steady':    tri(gap_rate, -TREND_STEADY, 0, TREND_STEADY),
        'opening':   right_shoulder(gap_rate, TREND_STEADY, TREND_OPENING),
    }

def fuzzify_obstacle_distance(d: float) -> dict:
    return {
        'blocking': left_shoulder(d, 45, 70),
        'near':     tri(d, 60, 110, 170),
        'clear':    right_shoulder(d, 130, 180),
    }

def fuzzify_turn_demand(abs_servo: float) -> dict:
    return {
        'straight': tri(abs_servo, 0.0, 0.0, 0.12),
        'gentle':   tri(abs_servo, 0.05, 0.18, 0.30),
        'sharp':    right_shoulder(abs_servo, 0.24, 0.40),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Obstacle measurements
# ══════════════════════════════════════════════════════════════════════════════

def nearest_relevant_obstacle(car_pos: tuple,
                               tangent: np.ndarray,
                               left_normal: np.ndarray,
                               obstacles: list) -> dict:
    """
    Scan obstacles that are:
      • ahead of the car (along tangent)
      • within OBSTACLE_LOOKAHEAD px
      • within OBSTACLE_TRACK_HALF_WIDTH lateral offset
    Returns the nearest such obstacle with state label.
    """
    _CLEAR = {'distance': float('inf'), 'side_offset': 0.0,
               'state': 'clear', 'point': None}
    if not obstacles:
        return _CLEAR

    t  = tangent     / (np.linalg.norm(tangent)     + 1e-9)
    n  = left_normal / (np.linalg.norm(left_normal) + 1e-9)
    p0 = np.array(car_pos, dtype=float)

    best_dist = float('inf')
    best      = None

    for obs in obstacles:
        vec  = np.array(obs, dtype=float) - p0
        along = float(np.dot(vec, t))
        side  = float(np.dot(vec, n))
        eucl  = float(np.linalg.norm(vec))

        if along < 0 or along > OBSTACLE_LOOKAHEAD:
            continue
        if abs(side) > OBSTACLE_TRACK_HALF_WIDTH:
            continue
        if eucl < best_dist:
            best_dist = eucl
            best      = (obs, side)

    if best is None:
        return _CLEAR

    obs_state = fuzzify_obstacle_distance(best_dist)
    return {
        'distance':    best_dist,
        'side_offset': best[1],
        'state':       dominant_state(obs_state),
        'point':       best[0],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Lane-change decision (rule-based)
# ══════════════════════════════════════════════════════════════════════════════

def _lane_obstacle_free(car_pos: tuple, lanes: dict, lane_num: int) -> bool:
    """True if the given lane has no relevant obstacle ahead of car_pos."""
    ref = lanes.get(f'lane{lane_num}_ref', [])
    if not ref:
        return False
    idx            = project_onto_curve(car_pos, ref)
    _, tangent, ln = local_path_frame(ref, idx)
    info           = nearest_relevant_obstacle(car_pos, tangent, ln, obstacle_positions)
    return info['state'] == 'clear'


def decide_lane(car_id: int, car_pos: tuple, lanes: dict,
                obstacle_info: dict, now: float) -> int:
    """
    Rule-based / fuzzy-style lane decision:
      • Default: stay in current lane.
      • Obstacle blocking/near → check adjacent lane.
          – Adjacent free  → change lane immediately.
          – Adjacent blocked → stay, speed controller will slow/stop.
      • After LANE_CHANGE_HOLD seconds in lane 2 with no obstacle → return to lane 1.
    """
    if car_id not in lane_state:
        lane_state[car_id] = {'lane': 1, 'timer': now, 'overtaking': False}

    st      = lane_state[car_id]
    current = st['lane']
    obs_bad = obstacle_info['state'] in ('blocking', 'near')

    if obs_bad:
        adjacent = 2 if current == 1 else 1
        if lanes.get(f'lane{adjacent}_ready', False):
            if _lane_obstacle_free(car_pos, lanes, adjacent):
                if adjacent != current:
                    st['lane']       = adjacent
                    st['timer']      = now
                    st['overtaking'] = True

    # Return to lane 1 after hysteresis hold, if no obstacle in lane 1
    if st['overtaking'] and st['lane'] == 2:
        if (now - st['timer'] >= LANE_CHANGE_HOLD) and not obs_bad:
            if _lane_obstacle_free(car_pos, lanes, 1):
                st['lane']       = 1
                st['timer']      = now
                st['overtaking'] = False

    return st['lane']


# ══════════════════════════════════════════════════════════════════════════════
#  Platoon helpers (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def get_curve_direction(curve_pts: list, idx: int, heading: tuple) -> int:
    h = np.array(heading, dtype=float)
    if np.linalg.norm(h) < 1e-6:
        return 1
    n    = len(curve_pts)
    curr = np.array(curve_pts[idx],           dtype=float)
    nxt  = np.array(curve_pts[(idx + 1) % n], dtype=float)
    prv  = np.array(curve_pts[(idx - 1) % n], dtype=float)
    return 1 if np.dot(nxt - curr, h) >= np.dot(prv - curr, h) else -1


def get_platoon_order(cars: dict, curve_pts: list) -> list:
    if LEADER_ID not in cars or not curve_pts:
        return ([LEADER_ID] if LEADER_ID in cars else []) + sorted(
            k for k in cars if k != LEADER_ID)

    direction = 1
    for cd in cars.values():
        if np.linalg.norm(np.array(cd['heading'], dtype=float)) > 1e-6:
            idx       = project_onto_curve(cd['midpoint'], curve_pts)
            direction = get_curve_direction(curve_pts, idx, cd['heading'])
            break

    n           = len(curve_pts)
    leader_idx  = project_onto_curve(cars[LEADER_ID]['midpoint'], curve_pts)
    followers   = []
    for cid, cd in cars.items():
        if cid == LEADER_ID:
            continue
        idx        = project_onto_curve(cd['midpoint'], curve_pts)
        arc_behind = (direction * (leader_idx - idx)) % n
        followers.append((arc_behind, cid))

    followers.sort()
    return [LEADER_ID] + [cid for _, cid in followers]


def compute_inter_vehicle_distance(car_ahead: dict, car_behind: dict) -> float:
    ref_ahead  = car_ahead.get('rear')
    ref_behind = car_behind.get('front') or car_behind.get('midpoint')
    if ref_ahead is None or ref_behind is None:
        return float('inf')
    return float(np.linalg.norm(
        np.array(ref_ahead, dtype=float) - np.array(ref_behind, dtype=float)))


def update_platoon_metrics(platoon_order: list, cars: dict, current_time: float) -> None:
    for rank, cid in enumerate(platoon_order):
        if cid not in platoon_metrics:
            platoon_metrics[cid] = {
                'rank': rank, 'gap': float('inf'), 'gap_speed': 0.0,
                'relative_speed': 0.0, 'gap_state': '-', 'trend_state': '-',
                'zone': '-', 'last_gap': None, 'last_time': None,
            }
        m       = platoon_metrics[cid]
        m['rank'] = rank

        if rank == 0:
            m.update({'gap': float('inf'), 'gap_speed': 0.0,
                      'relative_speed': 0.0, 'gap_state': 'leader',
                      'trend_state': '-', 'zone': '-'})
            continue

        prev_id = platoon_order[rank - 1]
        if cid not in cars or prev_id not in cars:
            continue

        gap = compute_inter_vehicle_distance(cars[prev_id], cars[cid])

        gap_speed = 0.0
        if m['last_gap'] is not None and m['last_time'] is not None:
            dt = current_time - m['last_time']
            if dt > 1e-6:
                gap_speed = (gap - m['last_gap']) / dt

        zone = ('red' if gap < RED_GAP
                else 'orange' if (gap < TARGET_GAP - GAP_TOL or gap_speed < -TREND_STEADY)
                else 'green')

        m.update({
            'gap':           gap,
            'gap_speed':     gap_speed,
            'relative_speed': tracker.get(cid, {}).get('speed', 0.0)
                               - tracker.get(prev_id, {}).get('speed', 0.0),
            'gap_state':     dominant_state(fuzzify_gap(gap)),
            'trend_state':   dominant_state(fuzzify_gap_trend(gap_speed)),
            'zone':          zone,
            'last_gap':      gap,
            'last_time':     current_time,
        })


# ══════════════════════════════════════════════════════════════════════════════
#  Fuzzy steering & speed
# ══════════════════════════════════════════════════════════════════════════════

def fuzzy_steering(lateral_error: float, heading_error: float, obstacle_info: dict):
    lat = fuzzify_lateral(lateral_error)
    hdg = fuzzify_heading(heading_error)

    contributions = [
        # Lateral corrections
        (lat['far_left'],       +MAX_SERVO),
        (lat['slightly_left'],  +MED_SERVO),
        (lat['aligned'],         0.0),
        (lat['slightly_right'], -MED_SERVO),
        (lat['far_right'],      -MAX_SERVO),
        # Heading corrections
        (hdg['need_left_hard'],  -MAX_SERVO),
        (hdg['need_left'],       -MED_SERVO),
        (hdg['heading_aligned'],  0.0),
        (hdg['need_right'],      +MED_SERVO),
        (hdg['need_right_hard'], +MAX_SERVO),
    ]

    # Obstacle lateral avoidance bias
    obs_dist = fuzzify_obstacle_distance(obstacle_info['distance'])
    side     = obstacle_info['side_offset']
    if obstacle_info['point'] is not None:
        if side > 0:   # obstacle on the left  → nudge right
            contributions += [(obs_dist['near'],     +SMALL_SERVO),
                               (obs_dist['blocking'], +MED_SERVO)]
        elif side < 0: # obstacle on the right → nudge left
            contributions += [(obs_dist['near'],     -SMALL_SERVO),
                               (obs_dist['blocking'], -MED_SERVO)]

    servo = float(np.clip(weighted_action(contributions), -MAX_SERVO, MAX_SERVO))
    state = {'lateral':  dominant_state(lat),
             'heading':  dominant_state(hdg),
             'obstacle': dominant_state(obs_dist)}
    return servo, state


def fuzzy_leader_speed(abs_servo: float, obstacle_info: dict):
    turn = fuzzify_turn_demand(abs_servo)
    obs  = fuzzify_obstacle_distance(obstacle_info['distance'])

    rules = [
        (obs['blocking'],                         STOP_SPEED),
        (obs['near'],                             SLOW_SPEED),
        (turn['sharp'],                           SLOW_SPEED),
        (turn['gentle'],                          CRUISE_SPEED),
        (min(turn['straight'], obs['clear']),     MAX_SPEED),
    ]
    speed = float(np.clip(weighted_action(rules, CRUISE_SPEED), 0.0, MAX_SPEED))

    zone = ('stop' if obs['blocking'] > 0.5
            else 'slow' if (obs['near'] > 0.3 or turn['sharp'] > 0.3)
            else 'go')
    state = {'turn': dominant_state(turn), 'obstacle': dominant_state(obs), 'zone': zone}
    return speed, state


def fuzzy_follower_speed(abs_servo: float, obstacle_info: dict,
                          gap: float, gap_rate: float):
    turn = fuzzify_turn_demand(abs_servo)
    obs  = fuzzify_obstacle_distance(obstacle_info['distance'])
    g    = fuzzify_gap(gap)
    tr   = fuzzify_gap_trend(gap_rate)

    orange = max(g['close'], tr['shrinking'], obs['near'], turn['sharp'])
    green  = min(max(g['ideal'], g['far']), max(tr['steady'], tr['opening']))

    rules = [
        (g['too_close'],                              STOP_SPEED),
        (obs['blocking'],                             STOP_SPEED),
        (orange,                                      SLOW_SPEED),
        (min(g['ideal'], tr['steady']),               CRUISE_SPEED),
        (min(g['far'],  max(tr['steady'],tr['opening'])), MAX_SPEED),
        (min(green, turn['gentle']),                  CRUISE_SPEED),
        (min(green, turn['straight'], obs['clear']),  MAX_SPEED),
    ]
    speed = float(np.clip(weighted_action(rules, CRUISE_SPEED), 0.0, MAX_SPEED))

    zone = ('red' if gap < RED_GAP
            else 'orange' if (gap < TARGET_GAP - GAP_TOL
                              or gap_rate < -TREND_STEADY
                              or obstacle_info['distance'] < 140)
            else 'green')
    state = {'gap': dominant_state(g), 'trend': dominant_state(tr),
             'obstacle': dominant_state(obs), 'turn': dominant_state(turn), 'zone': zone}
    return speed, state


# ══════════════════════════════════════════════════════════════════════════════
#  Drawing
# ══════════════════════════════════════════════════════════════════════════════

def draw_lanes(frame: np.ndarray, lanes: dict) -> np.ndarray:
    """
    1. fillPoly the lane regions (semi-transparent).
    2. Draw boundary polylines (inner / middle / outer).
    3. Draw reference lines as light dashes.
    """
    overlay = frame.copy()

    # Lane 1 fill: inner ↔ middle
    if 'inner_curve' in lanes and 'middle_curve' in lanes:
        inner_arr  = np.array(lanes['inner_curve'],  dtype=np.int32)
        middle_arr = np.array(lanes['middle_curve'], dtype=np.int32)
        poly1 = np.vstack([inner_arr, middle_arr[::-1]])
        cv2.fillPoly(overlay, [poly1], LANE1_FILL_COLOR)

    # Lane 2 fill: middle ↔ outer
    if 'middle_curve' in lanes and 'outer_curve' in lanes:
        middle_arr = np.array(lanes['middle_curve'], dtype=np.int32)
        outer_arr  = np.array(lanes['outer_curve'],  dtype=np.int32)
        poly2 = np.vstack([middle_arr, outer_arr[::-1]])
        cv2.fillPoly(overlay, [poly2], LANE2_FILL_COLOR)

    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    # Boundary polylines
    for key, color in [('inner_curve',  INNER_LINE_COLOR),
                        ('middle_curve', MIDDLE_LINE_COLOR),
                        ('outer_curve',  OUTER_LINE_COLOR)]:
        if key in lanes:
            arr = np.array(lanes[key], dtype=np.int32)
            cv2.polylines(frame, [arr], True, color, 2)

    # Reference lines: subtle dashes every 8 points
    for key in ('lane1_ref', 'lane2_ref'):
        if key in lanes:
            pts = lanes[key]
            n   = len(pts)
            for i in range(0, n, 8):
                j = (i + 4) % n
                cv2.line(frame, pts[i], pts[j], REF_LINE_COLOR, 1)

    return frame


def draw_guide_line(frame: np.ndarray, front_pos: tuple,
                    target: tuple, obstacle_close: bool) -> np.ndarray:
    """
    Draw a projected guide line from the front marker midpoint to the lookahead
    target. Color: red when an obstacle is close, white otherwise.
    """
    color = (0, 0, 220) if obstacle_close else (255, 255, 255)
    cv2.line(frame, front_pos, target, color, 2)
    cv2.circle(frame, target, 9, color, 2)
    cv2.circle(frame, target, 3, color, -1)
    return frame


def draw_obstacles(frame: np.ndarray, obstacles: list) -> np.ndarray:
    for obs in obstacles:
        ox, oy = obs
        overlay = frame.copy()
        cv2.circle(overlay, (ox, oy), OBSTACLE_RADIUS, (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.circle(frame, (ox, oy), OBSTACLE_RADIUS, (0, 0, 255), 2)
        cv2.circle(frame, (ox, oy), 6, (0, 0, 255), -1)
        cv2.putText(frame, "OBS", (ox + 8, oy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return frame


def draw_cars(frame: np.ndarray, cars: dict) -> np.ndarray:
    for _, cd in cars.items():
        front = cd['front']
        rear  = cd['rear']
        mid   = cd['midpoint']
        cv2.circle(frame, rear, 8, (0, 0, 255),   -1)
        cv2.circle(frame, mid,  6, (0, 165, 255),  -1)
        if front is not None:
            cv2.circle(frame, front, 8, (0, 255, 0), -1)
            cv2.line(frame, rear, front, (255, 0, 255), 2)
    return frame


def draw_platoon_overlay(frame: np.ndarray, platoon_order: list, cars: dict) -> np.ndarray:
    for rank, cid in enumerate(platoon_order):
        if cid not in cars:
            continue
        label = "L" if rank == 0 else f"F{rank}"
        pos   = (cars[cid]['rear'][0] + 10, cars[cid]['rear'][1] - 10)
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 215, 255) if rank == 0 else (220, 220, 255), 2)

    for rank in range(1, len(platoon_order)):
        fid = platoon_order[rank]
        aid = platoon_order[rank - 1]
        if fid not in cars or aid not in cars:
            continue
        follower_ref = cars[fid].get('front') or cars[fid]['midpoint']
        ahead_ref    = cars[aid]['rear']
        gap  = platoon_metrics.get(fid, {}).get('gap',  float('inf'))
        zone = platoon_metrics.get(fid, {}).get('zone', '-')
        color = ((0, 200, 0) if zone == 'green'
                 else (0, 165, 255) if zone == 'orange'
                 else (0, 0, 220))
        cv2.line(frame, follower_ref, ahead_ref, color, 2)
        cv2.putText(frame, f"{gap:.0f}px {zone}",
                    midpoint(follower_ref, ahead_ref),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
    return frame


def draw_rule_overlay(frame: np.ndarray,
                      car_id: int, servo: float, motor: float,
                      lane_info: dict, obstacle_info: dict,
                      role: str, steer_state: dict, speed_state: dict,
                      platoon_order: list, n_obstacles: int,
                      current_lane: int, lc_state: dict) -> np.ndarray:
    # Determine lane-change status label
    lc_label = "overtaking" if lc_state.get('overtaking') else "normal"

    lines = [
        f"Car {car_id} [{role}]  Lane {current_lane} [{lc_label}]",
        f"Servo {servo:+.2f}  Motor {motor:.2f}",
        f"Lat {lane_info['lateral_error']:+.1f}px  -> {steer_state.get('lateral', '-')}",
        f"Head {lane_info['heading_error']:+.1f}deg -> {steer_state.get('heading', '-')}",
        f"Obs {obstacle_info['distance']:.1f}px  -> {steer_state.get('obstacle', '-')}",
        f"Speed zone -> {speed_state.get('zone', '-')}",
        f"Platoon {platoon_order}",
        f"Visible obstacles: {n_obstacles}",
    ]

    if role != "LEADER" and car_id in platoon_metrics:
        m = platoon_metrics[car_id]
        lines += [
            f"Gap {m['gap']:.1f}px -> {m['gap_state']}",
            f"dGap/dt {m['gap_speed']:+.1f}px/s -> {m['trend_state']}",
            f"Platoon zone -> {m['zone']}",
        ]

    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, 28 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    # Reference-point dot and obstacle line
    p = lane_info['path_point']
    cv2.circle(frame, p, 5, (255, 255, 0), -1)
    if obstacle_info['point'] is not None:
        cv2.line(frame, p, obstacle_info['point'], (0, 0, 255), 1)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(frame: np.ndarray, car_id: int):
    """
    Lane-based fuzzy controller with tri-oval track, dual lanes, obstacle
    detection, and rule-based lane-change decision.

    Marker conventions
    ──────────────────
    4×4 ArUco (CAR_DICT)     : ID 49 = front marker; other IDs = rear markers.
    5×5 ArUco (TRACK_DICT)   : IDs 0-2 = inner boundary, 3-5 = middle, 6-8 = outer.
    6×6 ArUco (OBSTACLE_DICT): static obstacles anywhere on the track.

    Lanes
    ─────
    Lane 1 (default) : between inner and middle curves.
    Lane 2           : between middle and outer curves.
    Reference line   : midpoint between the two bounding curves of the active lane.

    Lane-change logic
    ─────────────────
    Stay in lane 1 by default.
    If an obstacle is blocking/near in the current lane:
      • Check the adjacent lane.
      • If adjacent is free → change immediately.
      • If adjacent is also blocked → hold position (speed drops via fuzzy speed rule).
    After LANE_CHANGE_HOLD seconds in lane 2 with no obstacle → return to lane 1.

    Returns
    ───────
    (servo, motor, annotated_frame)
    """
    global obstacle_positions

    car_frame = frame.copy()
    now       = time.time()

    # ── 1. Detect markers ──────────────────────────────────────────────────────
    car_markers, track_markers, obs_markers = detect_all_markers(car_frame)
    obstacle_positions = [marker_center(c) for c in obs_markers.values()]

    # ── 2. Scene state ─────────────────────────────────────────────────────────
    cars  = identify_cars(car_markers, car_id)
    lanes = detect_lanes(track_markers)
    update_tracker(cars, now)

    # Defaults
    servo        = 0.0
    motor        = 0.0
    steer_state  = {}
    speed_state  = {}
    lane_info    = {'path_point': (0, 0), 'lateral_error': 0.0,
                    'heading_error': 0.0, 'tangent': np.array([1.0, 0.0]),
                    'left_normal': np.array([0.0, -1.0])}
    obstacle_info = {'distance': float('inf'), 'side_offset': 0.0,
                     'point': None, 'state': 'clear'}
    current_lane = lane_state.get(car_id, {}).get('lane', 1)
    target_pt    = None

    # ── 3. Platoon bookkeeping (uses active-lane reference) ────────────────────
    ref_curve     = lanes.get(f'lane{current_lane}_ref', [])
    platoon_order = []
    if ref_curve:
        platoon_order = get_platoon_order(cars, ref_curve)
        update_platoon_metrics(platoon_order, cars, now)

    role = "LEADER" if car_id == LEADER_ID else "FOLLOWER"

    # ── 4. Control ─────────────────────────────────────────────────────────────
    if car_id in cars and ref_curve:
        car_ref = cars[car_id]['front'] or cars[car_id]['midpoint']

        # Compute lane measurements and obstacle scan for the current lane
        lane_info     = compute_lane_measurements(car_ref, cars[car_id]['heading'], ref_curve)
        obstacle_info = nearest_relevant_obstacle(
            car_ref, lane_info['tangent'], lane_info['left_normal'], obstacle_positions)

        # Lane-change decision (may switch current_lane)
        current_lane = decide_lane(car_id, car_ref, lanes, obstacle_info, now)

        # Re-evaluate on the (possibly new) reference curve
        new_ref = lanes.get(f'lane{current_lane}_ref', ref_curve)
        if new_ref is not ref_curve:
            ref_curve     = new_ref
            lane_info     = compute_lane_measurements(
                car_ref, cars[car_id]['heading'], ref_curve)
            obstacle_info = nearest_relevant_obstacle(
                car_ref, lane_info['tangent'], lane_info['left_normal'], obstacle_positions)

        # Steering
        raw_servo, steer_state = fuzzy_steering(
            lane_info['lateral_error'], lane_info['heading_error'], obstacle_info)
        prev_servo = last_sent_angles.get(car_id, 0.0)
        servo = float(np.clip(
            smooth_value(prev_servo, raw_servo, SERVO_SMOOTHING), -MAX_SERVO, MAX_SERVO))
        last_sent_angles[car_id] = servo

        # Speed
        if car_id == LEADER_ID:
            raw_motor, speed_state = fuzzy_leader_speed(abs(servo), obstacle_info)
        else:
            gap      = platoon_metrics.get(car_id, {}).get('gap',       float('inf'))
            gap_rate = platoon_metrics.get(car_id, {}).get('gap_speed', 0.0)
            raw_motor, speed_state = fuzzy_follower_speed(
                abs(servo), obstacle_info, gap, gap_rate)

        safety_stop = (speed_state.get('zone') == 'red') or (obstacle_info['state'] == 'blocking')
        prev_motor = last_sent_motors.get(car_id, 0.0)
        motor = float(np.clip(
            smooth_value(prev_motor, raw_motor, MOTOR_SMOOTHING, safety_stop=safety_stop),
            0.0, MAX_SPEED))
        last_sent_motors[car_id] = motor

        # Guide-line lookahead target on the reference curve
        idx_front = project_onto_curve(car_ref, ref_curve)
        target_pt = find_lookahead_point(ref_curve, idx_front, LOOKAHEAD_DIST)

    # ── 5. Drawing ─────────────────────────────────────────────────────────────
    car_frame = draw_lanes(car_frame, lanes)
    car_frame = draw_obstacles(car_frame, obstacle_positions)
    car_frame = draw_cars(car_frame, cars)

    # Guide line from front-marker midpoint to lookahead target
    if (car_id in cars and cars[car_id]['front'] is not None and target_pt is not None):
        obs_close = obstacle_info['state'] in ('blocking', 'near')
        car_frame = draw_guide_line(
            car_frame, cars[car_id]['front'], target_pt, obs_close)

    if platoon_order:
        car_frame = draw_platoon_overlay(car_frame, platoon_order, cars)

    lc_st = lane_state.get(car_id, {})
    car_frame = draw_rule_overlay(
        car_frame, car_id, servo, motor,
        lane_info, obstacle_info, role,
        steer_state, speed_state,
        platoon_order, len(obstacle_positions),
        current_lane, lc_st)

    return round(servo, 2), round(motor, 2), car_frame


# ── Optional standalone camera test (uncomment to use) ────────────────────────
# import os, sys
#
# def init_camera(cam_index=0):
#     cap = cv2.VideoCapture(cam_index,
#                            cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
#     if not cap.isOpened():
#         sys.exit("Camera not found!")
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
#     return cap
#
# if __name__ == "__main__":
#     cap     = init_camera(1)
#     car_idx = 1
#     while True:
#         ret, frame = cap.read()
#         if ret and frame is not None:
#             servo, motor, vis = run(frame, car_idx)
#             cv2.imshow("Lane Controller", vis)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#     cap.release()
#     cv2.destroyAllWindows()