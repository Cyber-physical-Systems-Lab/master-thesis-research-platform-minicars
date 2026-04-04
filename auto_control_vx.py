### Attempt of having lane following - version not used ###
"""
────────────────
Tri-oval lane following  +  leader-follower platooning  +  static obstacle avoidance.

Marker dictionaries
───────────────────
  DICT_4X4_50   → car markers       IDs: 0 (shared front), 1…N (rear per car)
  DICT_6X6_250  → lane markers      IDs: 196-201 (tri-oval anchors)
  DICT_5X5_250  → obstacle markers  any IDs  (static objects near the track)

Platoon rules
─────────────
  • Car with rear-marker ID == LEADER_ID (1) is the leader.
  • All other detected rear-marker IDs are followers.
  • Platoon order = [leader, closest follower behind, …, farthest follower].
  • Followers cannot overtake: speed is zero-clamped below BRAKE_DISTANCE.

Obstacle avoidance
──────────────────
  • Waypoint scoring penalises paths whose approach segment clips an obstacle
    bubble (radius OBSTACLE_RADIUS).  Deeply-blocked paths score ∞ → skipped.
  • Speed is independently reduced / stopped when the nearest forward obstacle
    falls inside OBSTACLE_SLOW_DIST / OBSTACLE_STOP_DIST.
  • Both constraints are active for leader AND followers; followers take the
    minimum of gap-schedule, curve-schedule, and obstacle-schedule.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time


# ── Tunable constants ──────────────────────────────────────────────────────────
LOW_THRESHOLD          = 80
HIGH_THRESHOLD         = 120
WEIGHT                 = 0.5
ANGLE_FAVOR            = 0.7

SPEED_THRESHOLD        = 15
STATUS_UPDATE_INTERVAL = 0.1
SCALING_FACTOR         = 10.3

MIN_SPEED       = 0.55
MAX_SPEED       = 0.70
MAX_SERVO       = 0.5
ANGLE_THRESHOLD = 0.02
JUMP_FILTER     = 0.78

# Lane marker IDs (6×6 ArUco)
RIGHT_END_IDS = [196, 197]
LEFT_END_IDS  = [198, 199]
RIGHT_MID_ID  = 200
LEFT_MID_ID   = 201

TRIOVAL_SAMPLES = 250

# ── Leader-follower constants ──────────────────────────────────────────────────
LEADER_ID        = 1    # rear-marker ID of the designated leader
SAFE_DISTANCE    = 120  # px — target following gap (follower-front → car-ahead-rear)
BRAKE_DISTANCE   = 55   # px — hard-stop gap threshold
CATCHUP_DISTANCE = 200  # px — gap above which follower ramps back to MAX_SPEED

# ── Static obstacle constants ──────────────────────────────────────────────────
#   5×5 ArUco markers placed on/near the trioval track line.
OBSTACLE_RADIUS    = 55   # px — exclusion bubble; path segments inside are penalised
OBSTACLE_SLOW_DIST = 150  # px — start decelerating when nearest forward obstacle ≤ this
OBSTACLE_STOP_DIST = 60   # px — full-stop threshold


# ── Global state ───────────────────────────────────────────────────────────────
tracker:            dict = {}   # {car_id: {status, center, last_time, heading, speed}}
last_sent_angles:   dict = {}   # {car_id: last_servo_float}

# Per-car platoon telemetry (written by update_platoon_metrics every frame):
#   rank, gap [px], gap_speed [px/s], relative_speed [scaled], last_gap, last_time
platoon_metrics:    dict = {}

# Obstacle center positions — refreshed every run() call from 5×5 detections
obstacle_positions: list = []   # [(x, y), …]
servo = motor = 0.0


# ── Geometric helpers ──────────────────────────────────────────────────────────
def marker_center(corner) -> tuple:
    return tuple(np.mean(corner[0], axis=0).astype(int))

def midpoint(p1: tuple, p2: tuple) -> tuple:
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

# def project_point_to_line(p, a, b) -> tuple:
#     """Orthogonal projection of point p onto the infinite line through a–b."""
#     ap    = np.array(p, dtype=float) - np.array(a, dtype=float)
#     ab    = np.array(b, dtype=float) - np.array(a, dtype=float)
#     denom = np.dot(ab, ab)
#     if denom == 0:
#         return tuple(np.array(a, dtype=int))
#     t = np.dot(ap, ab) / denom
#     return tuple((np.array(a, dtype=float) + t * ab).astype(int))


# ── Tri-oval curve fitting ─────────────────────────────────────────────────────
def fit_trioval_polar(marker_positions: list, n: int = TRIOVAL_SAMPLES) -> list:
    """
    Fit a closed polar Fourier curve through all available marker positions:

        r(θ) = R₀ + a₁·cosθ + b₁·sinθ + a₂·cos2θ + b₂·sin2θ

    Least-squares 6×5 system — robust with 3-6 markers.
    """
    pts    = np.array(marker_positions, dtype=float)
    cx, cy = pts.mean(axis=0)

    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    angles = np.arctan2(dy, dx)
    radii  = np.hypot(dx, dy)

    A = np.column_stack([
        np.ones_like(angles),
        np.cos(angles),    np.sin(angles),
        np.cos(2*angles),  np.sin(2*angles),
    ])
    coeffs, *_ = np.linalg.lstsq(A, radii, rcond=None)
    R0, a1, b1, a2, b2 = coeffs

    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r  = np.maximum(
        R0 + a1*np.cos(th) + b1*np.sin(th)
           + a2*np.cos(2*th) + b2*np.sin(2*th),
        1.0
    )
    xs = (cx + r * np.cos(th)).astype(int)
    ys = (cy + r * np.sin(th)).astype(int)

    return list(zip(xs.tolist(), ys.tolist()))

def detect_trioval(lane_markers: dict) -> dict:
    """Collect 6×6 marker centers and fit the closed tri-oval through them."""
    all_ids = RIGHT_END_IDS + LEFT_END_IDS + [RIGHT_MID_ID, LEFT_MID_ID]
    found   = [marker_center(lane_markers[i])
               for i in all_ids if i in lane_markers]
    result  = {'n_markers': len(found)}
    if len(found) >= 3:
        curve           = fit_trioval_polar(found)
        result['curve'] = curve
        result['left']  = curve   # alias consumed by steering & drawing
    return result


# ── Platoon geometry helpers ───────────────────────────────────────────────────
def project_onto_curve(pos: tuple, curve_pts: list) -> int:
    """Index of the nearest trioval point to pos."""
    arr = np.array(curve_pts, dtype=float)
    return int(np.argmin(np.linalg.norm(arr - np.array(pos, dtype=float), axis=1)))

def get_curve_direction(curve_pts: list, idx: int, heading: tuple) -> int:
    """
    +1 if increasing curve index aligns with the heading (= forward direction),
    -1 otherwise.  Determines which arc direction is 'ahead' on the oval.
    """
    n    = len(curve_pts)
    h    = np.array(heading, dtype=float)
    if np.linalg.norm(h) < 1e-6:
        return 1
    curr = np.array(curve_pts[idx],           dtype=float)
    nxt  = np.array(curve_pts[(idx + 1) % n], dtype=float)
    prv  = np.array(curve_pts[(idx - 1) % n], dtype=float)

    return 1 if np.dot(nxt - curr, h) >= np.dot(prv - curr, h) else -1

def get_platoon_order(cars: dict, curve_pts: list) -> list:
    """
    Return [LEADER_ID, closest_follower, …, farthest_follower].

    Each follower's arc-distance 'behind' the leader is computed on the closed
    trioval.  Travel direction is inferred from the first car with a non-zero
    heading vector (typically the currently-controlled car).
    """
    if LEADER_ID not in cars or not curve_pts:
        return ([LEADER_ID] if LEADER_ID in cars else []) + \
               sorted(k for k in cars if k != LEADER_ID)

    n = len(curve_pts)

    direction = 1
    for cd in cars.values():
        if np.linalg.norm(cd['heading']) > 1e-3:
            idx       = project_onto_curve(cd['midpoint'], curve_pts)
            direction = get_curve_direction(curve_pts, idx, cd['heading'])
            break

    leader_idx    = project_onto_curve(cars[LEADER_ID]['midpoint'], curve_pts)
    follower_arcs = []
    for car_id, cd in cars.items():
        if car_id == LEADER_ID:
            continue
        fidx       = project_onto_curve(cd['midpoint'], curve_pts)
        arc_behind = (direction * (leader_idx - fidx)) % n
        follower_arcs.append((arc_behind, car_id))

    follower_arcs.sort()
    return [LEADER_ID] + [cid for _, cid in follower_arcs]


# ── Inter-vehicle distance & platoon metrics ───────────────────────────────────
def compute_inter_vehicle_distance(car_ahead: dict, car_behind: dict) -> float:
    """
    Euclidean gap: car_ahead['rear'] → car_behind['front'].
    Falls back to midpoint-to-midpoint when the follower's front marker is
    not visible (i.e. the follower is not the currently-controlled car).
    """
    ref_ahead  = car_ahead.get('rear')
    ref_behind = car_behind.get('front') or car_behind.get('midpoint')
    if ref_ahead is None or ref_behind is None:
        return float('inf')
    return float(np.linalg.norm(
        np.array(ref_ahead, dtype=float) - np.array(ref_behind, dtype=float)
    ))

def update_platoon_metrics(platoon_order: list, cars: dict,
                           current_time: float) -> None:
    """
    Refresh platoon_metrics for every platoon member.

    Written per-car:
      gap            — px gap to the car directly ahead       (inf for leader)
      gap_speed      — d(gap)/dt  [px/s]                      (+ opening, − closing)
      relative_speed — v_follower − v_ahead  [scaled units]   (+ means closing)

    These values are available externally via platoon_metrics for offline
    inter-vehicle distance logging / analysis.
    """
    for rank, car_id in enumerate(platoon_order):
        if car_id not in platoon_metrics:
            platoon_metrics[car_id] = {
                'rank': rank, 'gap': float('inf'),
                'gap_speed': 0.0, 'relative_speed': 0.0,
                'last_gap': None, 'last_time': None,
            }
        m        = platoon_metrics[car_id]
        m['rank'] = rank

        if rank == 0:
            m['gap'] = m['gap_speed'] = m['relative_speed'] = 0.0
            continue

        car_ahead_id = platoon_order[rank - 1]
        if car_id not in cars or car_ahead_id not in cars:
            continue

        gap = compute_inter_vehicle_distance(cars[car_ahead_id], cars[car_id])

        if m['last_gap'] is not None and m['last_time'] is not None:
            dt             = current_time - m['last_time']
            m['gap_speed'] = (gap - m['last_gap']) / dt if dt > 1e-6 else 0.0

        v_behind = tracker.get(car_id,       {}).get('speed', 0.0)
        v_ahead  = tracker.get(car_ahead_id, {}).get('speed', 0.0)
        m['relative_speed'] = v_behind - v_ahead

        m['gap']       = gap
        m['last_gap']  = gap
        m['last_time'] = current_time

def follower_motor_speed(gap: float) -> float:
    """
    Three-zone gap → speed schedule (anti-overtake via hard-stop at close range):

      gap < BRAKE_DISTANCE              → 0.0          (emergency stop)
      BRAKE_DISTANCE ≤ gap < SAFE_DIST  → linear ramp  0 … MIN_SPEED -- not used
      SAFE_DIST ≤ gap < CATCHUP_DIST    → MIN_SPEED    (safe cruise)
      gap ≥ CATCHUP_DIST                → ramp up      MIN_SPEED … MAX_SPEED
    """
    if gap < BRAKE_DISTANCE:
        return 0.0
    # if gap < SAFE_DISTANCE:
    #     t = (gap - BRAKE_DISTANCE) / max(SAFE_DISTANCE - BRAKE_DISTANCE, 1.0)
    #     return float(np.clip(MIN_SPEED * t, 0.0, MIN_SPEED))
    if gap < SAFE_DISTANCE or gap < CATCHUP_DISTANCE:
        return MIN_SPEED
    
    t = min((gap - CATCHUP_DISTANCE) / max(CATCHUP_DISTANCE, 1.0), 1.0)
    return float(np.clip(MIN_SPEED + (MAX_SPEED - MIN_SPEED) * t,
                         MIN_SPEED, MAX_SPEED))


# ── Static obstacle geometry ───────────────────────────────────────────────────
def segment_min_clearance(p1: tuple, p2: tuple, obstacles: list) -> float:
    """
    Minimum distance from any obstacle center to the line segment p1 → p2.

    Uses the perpendicular-foot formula clamped to [0, 1] so only the finite
    segment is considered — not its infinite extension [ref: ECE 470 robotics].
    Returns inf when the obstacle list is empty.
    """
    if not obstacles:
        return float('inf')
    p1a   = np.array(p1, dtype=float)
    d     = np.array(p2, dtype=float) - p1a
    dlen2 = float(np.dot(d, d))
    min_c = float('inf')
    for obs in obstacles:
        o = np.array(obs, dtype=float)
        if dlen2 < 1e-9:
            dist = float(np.linalg.norm(o - p1a))
        else:
            t    = float(np.clip(np.dot(o - p1a, d) / dlen2, 0.0, 1.0))
            dist = float(np.linalg.norm(o - (p1a + t * d)))
        min_c = min(min_c, dist)
    return min_c

def nearest_obstacle_in_cone(car_pos: tuple, car_heading_angle: float,
                              obstacles: list, fov_deg: float = 100.0) -> float:
    """
    Distance to the nearest obstacle within ±fov_deg/2 of the car's heading.
    Used for speed-reduction scheduling ahead of a blocked section of track.
    Returns inf when the forward cone is clear.
    """
    if not obstacles:
        return float('inf')
    center   = np.array(car_pos, dtype=float)
    min_dist = float('inf')
    for obs in obstacles:
        direction = np.array(obs, dtype=float) - center
        dist      = float(np.linalg.norm(direction))
        if dist < 1e-3:
            continue
        point_angle    = np.degrees(np.arctan2(-direction[1], direction[0]))
        relative_angle = (car_heading_angle - point_angle + 360) % 360
        if relative_angle > 180:
            relative_angle -= 360
        if abs(relative_angle) <= fov_deg / 2:
            min_dist = min(min_dist, dist)
    return min_dist

def obstacle_motor_speed(dist: float) -> float:
    """
    Speed limit imposed by the nearest forward obstacle:
      dist < OBSTACLE_STOP_DIST  → 0.0        (hard stop)
      stop ≤ dist < slow_dist    → linear ramp  0 … MIN_SPEED -- used only MIN_SPEED
      dist ≥ OBSTACLE_SLOW_DIST  → MAX_SPEED  (no constraint)
    """
    if dist < OBSTACLE_STOP_DIST:
        return 0.0
    if dist < OBSTACLE_SLOW_DIST:
        # t = (dist - OBSTACLE_STOP_DIST) / max(OBSTACLE_SLOW_DIST - OBSTACLE_STOP_DIST, 1.0)
        # return float(np.clip(MIN_SPEED * t, 0.0, MIN_SPEED))
        return MIN_SPEED
    return MAX_SPEED


# ── Marker detection ───────────────────────────────────────────────────────────
def detect_all_markers(frame: np.ndarray):
    """
    Single grayscale pass detecting all three ArUco dictionaries:
      DICT_4X4_50   → car markers      {id: corners}
      DICT_6X6_250  → lane markers     {id: corners}
      DICT_5X5_250  → obstacle markers {id: corners}

    Obstacle markers are drawn with a distinct red tint via drawDetectedMarkers.
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = aruco.DetectorParameters()

    def _detect(dictionary, border_color=None):
        det             = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(dictionary), params)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            return {}
        if border_color:
            aruco.drawDetectedMarkers(frame, corners, ids, border_color)
        else:
            aruco.drawDetectedMarkers(frame, corners, ids)
        return {int(i): corners[k] for k, i in enumerate(ids.flatten())}

    car_markers  = _detect(aruco.DICT_4X4_50)
    lane_markers = _detect(aruco.DICT_6X6_250)
    obs_markers  = _detect(aruco.DICT_5X5_250,
                           border_color=(0, 0, 255))   # red border for obstacles
    return car_markers, lane_markers, obs_markers


# ── Car geometry ───────────────────────────────────────────────────────────────
def identify_cars(car_markers: dict, car_id: int) -> dict:
    """
    Build per-car geometry from 4×4 markers.

    Convention (current single-front-marker scheme):
      Marker 0     → shared front marker for the currently-controlled car_id.
      Marker N > 0 → rear of car N (unique per car).

    NOTE: For a physical multi-car deployment, each car should carry a unique
    front-marker ID.  Until then, marker 0 is interpreted as the front of
    whichever car is being processed in this run() call.
    """
    front_center = marker_center(car_markers[0]) if 0 in car_markers else None
    cars = {}
    for mid, corner in car_markers.items():
        if mid == 0:
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
        cars[mid] = {'front': front, 'rear': rear_center,
                     'midpoint': mid_pt, 'heading': heading}
    return cars

def estimate_speed(curr_pos, prev_pos, dt: float) -> float:
    """
    Instantaneous speed in scaled units (px / (dt × SCALING_FACTOR)).
    Result stored in tracker per-car and consumed by update_platoon_metrics
    for relative-speed computation between consecutive platoon pairs.
    """
    if prev_pos is None or dt == 0:
        return 0.0
    return float(np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
                 / (dt * SCALING_FACTOR))

def update_tracker(cars: dict, current_time: float) -> None:
    for rear_id, cd in cars.items():
        if rear_id not in tracker:
            tracker[rear_id] = {
                'status': None, 'center': None,
                'last_time': None, 'heading': None, 'speed': 0.0,
            }
        t         = tracker[rear_id]
        last_time = t['last_time']
        if last_time is None or (current_time - last_time >= STATUS_UPDATE_INTERVAL):
            dt     = (current_time - last_time) if last_time else 0.0
            speed  = estimate_speed(cd['midpoint'], t['center'], dt)
            status = "moving" if speed > SPEED_THRESHOLD else "stopped"
            t['speed'] = speed
            if status != t['status']:
                print(f"Car {rear_id}: {status.upper()} (Speed: {speed:.2f})")
                t['status'] = status
            t['center']    = cd['midpoint']
            t['last_time'] = current_time
            t['heading']   = cd['heading']


# ── Heading conversion ─────────────────────────────────────────────────────────
def heading_to_angle(heading_vec: tuple) -> float:
    """(dx, dy) → compass angle in degrees [0, 360), image-space corrected."""
    return np.degrees(np.arctan2(-heading_vec[1], heading_vec[0])) % 360


# ── Scoring logic ──────────────────────────────────────────────────────────────
def dynamic_threshold(relative_angle: float) -> float:
    """Angle-dependent lookahead distance — look far on straights, close in turns."""
    angle = abs(relative_angle)
    if angle < 15:
        return HIGH_THRESHOLD
    elif angle < 30:
        return LOW_THRESHOLD + (HIGH_THRESHOLD - LOW_THRESHOLD) * 0.5
    elif angle < 60:
        return LOW_THRESHOLD + (HIGH_THRESHOLD - LOW_THRESHOLD) * 0.25
    else:
        return LOW_THRESHOLD

def compute_point_score(relative_angle: float, dist: float) -> float:
    """Weighted score: lower = better.  ∞ for points outside ±90° FoV."""
    if abs(relative_angle) > 90:
        return float('inf')
    dist             = max(LOW_THRESHOLD, min(HIGH_THRESHOLD, dist))
    normalized_dist  = (dist - LOW_THRESHOLD) / (HIGH_THRESHOLD - LOW_THRESHOLD)
    normalized_angle = (abs(relative_angle) / 90) ** 2.5
    return ANGLE_FAVOR * normalized_dist + (1 - ANGLE_FAVOR) * normalized_angle

def map_angle_to_servo(relative_angle: float, dist: float):
    """Maps (relative_angle, dist) → servo in [-0.5, 0.5].  None if behind car."""
    if abs(relative_angle) > 90:
        return None
    dist             = max(LOW_THRESHOLD, min(HIGH_THRESHOLD, dist))
    normalized_dist  = (HIGH_THRESHOLD - dist) / (HIGH_THRESHOLD - LOW_THRESHOLD)
    normalized_angle = (90 - abs(relative_angle)) / 90
    weight           = (normalized_angle * normalized_dist) ** WEIGHT
    servo = -weight * MAX_SERVO if relative_angle > 0 else weight * MAX_SERVO
    return float(np.clip(servo, -MAX_SERVO, MAX_SERVO))

def select_best_boundary_point(car_pos: tuple, car_heading_angle: float,
                                boundary_points: list):
    """
    Select the lowest-score trioval waypoint in the forward hemisphere.

    Obstacle path-clearance penalty (uses global obstacle_positions):
    
    For each candidate waypoint, the straight-line segment from car_pos to
    the candidate is tested against every obstacle bubble.  The minimum
    clearance is computed using the perpendicular-foot formula clamped to the
    finite segment.

    Two-tier penalty:
      clearance < OBSTACLE_RADIUS × 0.5  →  score = ∞   (hard block — path fully
                                                           inside obstacle bubble)
      clearance < OBSTACLE_RADIUS         →  score ×= (1 + encroachment × 3)
                                                      (soft penalty — clips bubble)

    Effect: the car naturally steers toward waypoints whose approach lines are
    clear.  If ALL nearby waypoints are blocked the car momentarily holds its
    last servo angle (handled by the jump-filter in run()) while decelerating
    via obstacle_motor_speed, then resumes as soon as a clear waypoint appears.
    """
    best_point = best_angle = best_dist = None
    best_score = float('inf')
    center     = np.array(car_pos, dtype=float)

    for pt in boundary_points:
        direction = np.array(pt, dtype=float) - center
        dist      = float(np.linalg.norm(direction))
        if dist < 1e-3:
            continue
        point_angle    = np.degrees(np.arctan2(-direction[1], direction[0]))
        relative_angle = (car_heading_angle - point_angle + 360) % 360
        if relative_angle > 180:
            relative_angle -= 360

        if dist < dynamic_threshold(relative_angle):
            score = compute_point_score(relative_angle, dist)
            # ── Obstacle path-clearance penalty ──────────────────────────────
            if obstacle_positions:
                clearance = segment_min_clearance(car_pos, pt, obstacle_positions)
                if clearance < OBSTACLE_RADIUS * 0.5:
                    # Hard block — this waypoint's approach line passes through
                    # the obstacle's core bubble.  Discard entirely.
                    score = float('inf')
                elif clearance < OBSTACLE_RADIUS:
                    # Soft penalty — path clips the outer part of the bubble.
                    # Scale penalty by encroachment depth (0 → ×1 at boundary,
                    # up to ×4 at inner boundary).
                    enc   = (OBSTACLE_RADIUS - clearance) / (OBSTACLE_RADIUS * 0.5)
                    score *= (1.0 + enc * 3.0)
            # ─────────────────────────────────────────────────────────────────
            if score < best_score:
                best_point = pt
                best_angle = relative_angle
                best_dist  = dist
                best_score = score

    return best_point, best_angle, best_dist


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_trioval(frame: np.ndarray, lane: dict) -> np.ndarray:
    if 'curve' not in lane:
        return frame
    curve_arr = np.array(lane['curve'], dtype=np.int32)
    overlay   = frame.copy()
    cv2.fillPoly(overlay, [curve_arr], (0, 200, 0))
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.polylines(frame, [curve_arr], True, (0, 255, 255), 2)
    return frame

def draw_obstacles(frame: np.ndarray, obstacles: list) -> np.ndarray:
    """
    Render each static obstacle (5×5 ArUco marker) as:
      • Semi-transparent red fill    — exclusion bubble (radius OBSTACLE_RADIUS)
      • Solid red ring               — bubble boundary
      • Filled red dot               — marker center
      • 'OBS' label                  — above center

    The semi-transparent fill gives an immediate visual of which sections of
    the trioval are affected, making it easy to verify avoidance behaviour.
    """
    for obs in obstacles:
        ox, oy = obs
        # Semi-transparent fill
        overlay = frame.copy()
        cv2.circle(overlay, (ox, oy), OBSTACLE_RADIUS, (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        # Hard boundary ring
        cv2.circle(frame, (ox, oy), OBSTACLE_RADIUS, (0, 0, 255), 2)
        # Center marker dot
        cv2.circle(frame, (ox, oy), 7, (0, 0, 255), -1)
        # Label
        cv2.putText(frame, "OBS", (ox + 10, oy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return frame

def draw_cars(frame: np.ndarray, cars: dict, lane: dict) -> np.ndarray:
    for rear_id, cd in cars.items():
        front, rear, mid = cd['front'], cd['rear'], cd['midpoint']
        cv2.circle(frame, rear, 8, (0, 0, 255), -1)
        if front is not None:
            cv2.circle(frame, front, 8, (0, 255, 0), -1)
            cv2.line(frame, rear, front, (255, 0, 255), 2)
        cv2.circle(frame, mid, 6, (0, 165, 255), -1)
    return frame

def draw_platoon_overlay(frame: np.ndarray, platoon_order: list,
                          cars: dict) -> np.ndarray:
    """
    For each consecutive platoon pair draw an inter-vehicle gap line:
      green  → gap ≥ SAFE_DISTANCE       (nominal following)
      orange → BRAKE_DISTANCE ≤ gap < SAFE_DISTANCE  (closing)
      red    → gap < BRAKE_DISTANCE      (critical — emergency stop active)

    Also annotates each car rear marker with its rank badge: L / F1 / F2 …
    """
    # Rank badges
    for rank, car_id in enumerate(platoon_order):
        if car_id not in cars:
            continue
        pos   = (cars[car_id]['rear'][0] + 12, cars[car_id]['rear'][1] - 10)
        label = "L" if rank == 0 else f"F{rank}"
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 215, 255) if rank == 0 else (200, 200, 255), 2)

    # Gap lines between consecutive pairs
    for rank in range(1, len(platoon_order)):
        fid = platoon_order[rank]
        aid = platoon_order[rank - 1]
        if fid not in cars or aid not in cars:
            continue

        follower_ref = cars[fid].get('front') or cars[fid]['midpoint']
        ahead_ref    = cars[aid]['rear']
        m            = platoon_metrics.get(fid, {})
        gap          = m.get('gap',      float('inf'))
        gap_spd      = m.get('gap_speed', 0.0)

        if gap < BRAKE_DISTANCE:
            color = (0, 0, 220)         # red
        elif gap < SAFE_DISTANCE:
            color = (0, 165, 255)       # orange
        else:
            color = (0, 200, 0)         # green

        cv2.line(frame, follower_ref, ahead_ref, color, 2)
        lbl_pos = midpoint(follower_ref, ahead_ref)
        cv2.putText(frame, f"{gap:.0f}px {gap_spd:+.0f}px/s",
                    lbl_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    return frame

def draw_telemetry(frame: np.ndarray, car_id: int, cars: dict,
                   servo: float, motor: float,
                   n_markers: int, n_obstacles: int = 0,
                   platoon_order: list = None,
                   car_rank: int = -1,
                   is_leader: bool = False) -> np.ndarray:
    """HUD overlay — now includes obstacle count, platoon role, and gap metrics."""
    role = "LEADER" if is_leader else (f"FOLLOWER #{car_rank}"
                                        if car_rank >= 0 else "–")
    lines = [
        f"Car {car_id}  [{role}]",
        f"Trioval: {n_markers}/6 anchors   Obstacles: {n_obstacles}",
        f"Servo: {servo:+.3f}   Motor: {motor:.2f}",
        f"Platoon: {platoon_order or []}",
    ]
    if not is_leader and car_id in platoon_metrics:
        m = platoon_metrics[car_id]
        lines.append(
            f"Gap: {m['gap']:.0f}px  dGap/dt: {m['gap_speed']:+.1f}px/s  "
            f"Vrel: {m['relative_speed']:+.2f}"
        )
    for idx, text in enumerate(lines):
        cv2.putText(frame, text, (10, 30 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    return frame


# Main function
def run(frame: np.ndarray, car_id: int):
    """
    Process one camera frame for car_id → (servo, motor, annotated_frame).

    Extended control pipeline
    ─────────────────────────
    1.  Detect 4×4 (cars), 6×6 (lane), 5×5 (obstacles) in one grayscale pass.
    2.  Refresh global obstacle_positions from 5×5 marker centers.
    3.  Fit closed tri-oval Fourier curve through 6×6 lane markers.
    4.  Build platoon order: LEADER_ID first, then followers sorted by
        arc-distance behind the leader on the trioval.
    5.  Update tracker (per-car speed via estimate_speed) and platoon_metrics
        (gap, gap_speed, relative_speed for every following car).
    6.  Obstacle-aware waypoint selection via select_best_boundary_point:
          — paths whose segment clips an obstacle bubble are penalised;
          — paths whose segment penetrates the core bubble are discarded.
    7.  Speed = min(gap_schedule, curve_schedule, obstacle_schedule):
          gap_schedule      → follower_motor_speed(gap)  [followers only;
                               leader uses MAX_SPEED here]
          curve_schedule    → MAX_SPEED × (1 − 0.5 × |servo|/MAX_SERVO)
          obstacle_schedule → obstacle_motor_speed(nearest_forward_obstacle)
        Taking the minimum guarantees all three constraints are respected
        simultaneously without any explicit priority logic.
    8.  Draw trioval fill + reference line, obstacle exclusion zones,
        car markers, platoon gap lines with color-coded status, telemetry HUD.
    """
    global obstacle_positions, motor, servo

    car_frame    = frame.copy()
    current_time = time.time()

    # ── Step 1–2: Detect markers, refresh obstacles ────────────────────────────
    car_markers, lane_markers, obs_markers = detect_all_markers(car_frame)
    obstacle_positions = [marker_center(c) for c in obs_markers.values()]

    # ── Step 3: Lane curve + car geometry ─────────────────────────────────────
    cars = identify_cars(car_markers, car_id)
    update_tracker(cars, current_time)
    lane = detect_trioval(lane_markers)

    # ── Steps 4–5: Platoon bookkeeping ─────────────────────────────────────────
    platoon_order = []
    if 'curve' in lane:
        platoon_order = get_platoon_order(cars, lane['curve'])
        update_platoon_metrics(platoon_order, cars, current_time)

    is_leader = (car_id == LEADER_ID)
    try:
        car_rank = platoon_order.index(car_id)
    except ValueError:
        car_rank = -1

    # ── Steps 6–7: Steering + speed ────────────────────────────────────────────
    if car_id in cars and 'left' in lane:
        car_ref           = cars[car_id]['front'] or cars[car_id]['midpoint']
        car_heading_angle = heading_to_angle(cars[car_id]['heading'])

        # Obstacle-aware waypoint selection (Step 6)
        best_point, best_angle, best_dist = select_best_boundary_point(
            car_ref, car_heading_angle, lane['left']
        )

        # Nearest forward obstacle for speed scheduling (Step 7)
        obs_dist = nearest_obstacle_in_cone(
            car_ref, car_heading_angle, obstacle_positions)

        if best_point is not None:
            cv2.circle(car_frame, tuple(best_point), 7, (0, 0, 0), -1)
            # Guide line turns red when an obstacle is close ahead
            line_color = (0, 0, 255) if obs_dist < OBSTACLE_SLOW_DIST else (255, 255, 255)
            cv2.line(car_frame, car_ref, tuple(best_point), line_color, 2)

            computed = map_angle_to_servo(best_angle, best_dist)
            if computed is not None:
                last = last_sent_angles.get(car_id)
                if (last is None
                        or (abs(computed - last) >= ANGLE_THRESHOLD
                            and abs(computed - last) < JUMP_FILTER)):
                    servo = computed
                    last_sent_angles[car_id] = servo
                else:
                    servo = last if last is not None else 0.0

            # Speed schedules
            turn_strength  = abs(servo) / MAX_SERVO
            curve_speed    = float(np.clip(
                MAX_SPEED * (1.0 - 0.5 * turn_strength), MIN_SPEED, MAX_SPEED))
            obs_speed      = obstacle_motor_speed(obs_dist)

            if is_leader:
                # Leader: curve + obstacle constraints only
                motor = min(curve_speed, obs_speed)
            else:
                # Follower: all three constraints — take minimum
                gap       = platoon_metrics.get(car_id, {}).get('gap', float('inf'))
                gap_spd   = follower_motor_speed(gap)
                motor     = min(gap_spd, curve_speed, obs_speed)
        else:
            # No valid waypoint found — hold servo, stop
            servo = last_sent_angles.get(car_id, 0.0)
            motor = 0.0

    # ── Step 8: Visualisation ──────────────────────────────────────────────────
    car_frame = draw_trioval(car_frame, lane)
    car_frame = draw_obstacles(car_frame, obstacle_positions)
    car_frame = draw_cars(car_frame, cars, lane)
    if platoon_order:
        car_frame = draw_platoon_overlay(car_frame, platoon_order, cars)
    car_frame = draw_telemetry(
        car_frame, car_id, cars, servo, motor,
        lane.get('n_markers', 0),
        len(obstacle_positions),
        platoon_order, car_rank, is_leader,
    )

    return round(servo, 2), round(motor, 2), car_frame


# # ── Standalone camera test ─────────────────────────────────────────────────
# import sys, os
# def init_camera(cam_index=0):
#     cap = cv2.VideoCapture(cam_index,
#               cv2.CAP_V4L2 if os.name == "posix" else cv2.CAP_ANY)
#     if not cap.isOpened():
#         sys.exit("Camera not found!\n")
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
#     print("Camera opened successfully!\n")
#     return cap
#
# if __name__ == '__main__':
#     capture = init_camera(1)
#     car_idx = 1
#     while True:
#         ret, frame = capture.read()
#         if ret and frame is not None:
#             servo, motor, vis = run(frame, car_idx)
#             cv2.imshow("Tracking", vis)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#     capture.release()
#     cv2.destroyAllWindows()