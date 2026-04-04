### First starting point in the whole lane following computation - version not used ###

import cv2
import cv2.aruco as aruco
import numpy as np
import time
# import sys
# import os


#TODO even better version, seems to work quite OK for one car
# fine-tune a little bit further and start to build upon this version


# ── Constants ──────────────────────────────────────────────────────────────────
LOW_THRESHOLD          = 40
HIGH_THRESHOLD         = 80
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

# Lane marker IDs (6x6 ArUco) — unchanged
RIGHT_END_IDS = [196, 197]
LEFT_END_IDS  = [198, 199]
RIGHT_MID_ID  = 200
LEFT_MID_ID   = 201

# Tri-oval curve
TRIOVAL_SAMPLES  = 250   # points sampled along the closed fitted curve
# TRACK_HALF_WIDTH = 25    # px — half-width of the drawn track band


# ── Global state ───────────────────────────────────────────────────────────────
tracker:          dict = {}
last_sent_angles: dict = {}


# ── Geometric helpers (unchanged) ──────────────────────────────────────────────
def marker_center(corner) -> tuple:
    return tuple(np.mean(corner[0], axis=0).astype(int))

def midpoint(p1: tuple, p2: tuple) -> tuple:
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def project_point_to_line(p, a, b) -> tuple:
    """
    Orthogonal projection of point p onto the infinite line through a–b.
    proj = a + t·(b−a),  t = dot(a→p, a→b) / |a→b|²
    """
    ap    = np.array(p, dtype=float) - np.array(a, dtype=float)
    ab    = np.array(b, dtype=float) - np.array(a, dtype=float)
    denom = np.dot(ab, ab)
    if denom == 0:
        return tuple(np.array(a, dtype=int))
    t = np.dot(ap, ab) / denom
    return tuple((np.array(a, dtype=float) + t * ab).astype(int))


# ── Tri-oval curve fitting (replaces sample_bezier + detect_lane_curves) ───────
def fit_trioval_polar(marker_positions: list, n: int = TRIOVAL_SAMPLES) -> list:
    """
    Fit a closed polar Fourier curve through all available marker positions:

        r(θ) = R₀ + a₁·cos θ  + b₁·sin θ
                  + a₂·cos 2θ + b₂·sin 2θ

    The 1st-harmonic (a₁, b₁) produces the basic elliptical oval shape.
    The 2nd-harmonic (a₂, b₂) introduces the characteristic tri-oval
    asymmetric bulge on one straight — a continuous analogue of the
    quartic Bean curve's (x²+y²)² = x³+y³ deviation from a pure conic.

    All 6 markers constrain the fit via a least-squares 6×5 system;
    the solution is the minimum-norm best fit through all anchor points.

    Parameters
    ----------
    marker_positions : list of (x, y) — at least 3, ideally all 6
    n                : number of output points on the closed curve

    Returns
    -------
    list of (x, y) integer tuples, evenly spaced around the closed curve.
    """
    pts    = np.array(marker_positions, dtype=float)
    cx, cy = pts.mean(axis=0)

    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    angles = np.arctan2(dy, dx)         # each marker's polar angle
    radii  = np.hypot(dx, dy)           # each marker's distance from centroid

    # Design matrix: [1,  cos θ,  sin θ,  cos 2θ,  sin 2θ]
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
        1.0     # guard against degenerate negative radii
    )
    xs = (cx + r * np.cos(th)).astype(int)
    ys = (cy + r * np.sin(th)).astype(int)
    return list(zip(xs.tolist(), ys.tolist()))


# def compute_curve_offset(curve_pts: list, offset_px: int) -> list:
#     """
#     Shift every point on the curve by offset_px pixels along its local
#     left-hand normal (perpendicular to the tangent, left of travel direction).
#     Positive offset → left of forward direction.
#     Used to generate inner and outer track boundary bands.
#     """
#     pts = np.array(curve_pts, dtype=float)
#     n   = len(pts)
#     out = []
#     for i in range(n):
#         tangent = pts[(i + 1) % n] - pts[(i - 1) % n]
#         norm_t  = np.linalg.norm(tangent)
#         if norm_t < 1e-6:
#             out.append(curve_pts[i])
#             continue
#         tangent /= norm_t
#         normal = np.array([-tangent[1], tangent[0]])   # left-hand normal
#         out.append(tuple((pts[i] + offset_px * normal).astype(int)))
#     return out


def detect_trioval(lane_markers: dict) -> dict:
    """
    Replaces detect_lane_curves.

    Collects all available 6x6 marker centers (up to 6) and fits a single
    closed tri-oval curve through them via fit_trioval_polar.

    Returns
    -------
    {
      'curve'     : list[(x,y)]  — TRIOVAL_SAMPLES points on the closed curve
      'left'      : same alias   — keeps select_best_boundary_point working
                                   without any modification
      'n_markers' : int          — markers successfully used in the fit
    }
    Falls back gracefully: requires ≥ 3 markers; returns empty dict otherwise.
    """
    all_ids = RIGHT_END_IDS + LEFT_END_IDS + [RIGHT_MID_ID, LEFT_MID_ID]
    found   = [marker_center(lane_markers[i])
               for i in all_ids if i in lane_markers]

    result = {'n_markers': len(found)}
    if len(found) >= 3:
        curve           = fit_trioval_polar(found)
        result['curve'] = curve
        result['left']  = curve     # alias consumed by run() and draw_cars()
    return result


# ── Marker detection (unchanged) ───────────────────────────────────────────────
def detect_all_markers(frame: np.ndarray):
    """
    Detect 4x4_50 (car) and 6x6_250 (lane) markers in a single grayscale pass.
    Returns:
        car_markers  : {id: corners}  — IDs 0, 1, 2 …
        lane_markers : {id: corners}  — IDs 196–201
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = aruco.DetectorParameters()

    def _detect(dictionary):
        det = aruco.ArucoDetector(aruco.getPredefinedDictionary(dictionary), params)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            return {}
        aruco.drawDetectedMarkers(frame, corners, ids)
        return {int(i): corners[k] for k, i in enumerate(ids.flatten())}
    
    return _detect(aruco.DICT_4X4_50), _detect(aruco.DICT_6X6_250)


# ── Car geometry (unchanged) ───────────────────────────────────────────────────
def identify_cars(car_markers: dict, car_id: int) -> dict:
    """
    Build per-car geometry from detected 4x4 markers.
    Convention:
      Front marker → ID = 0  (shared; assumed to belong to car_id)
      Rear  marker → ID = X  (unique per car)
    For the controlled car: heading = front − rear, midpoint = mean(front, rear).
    For other cars: heading = (0, 0), midpoint = rear center.
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


# ── Heading conversion (unchanged) ────────────────────────────────────────────
def heading_to_angle(heading_vec: tuple) -> float:
    """
    Convert (dx, dy) vector to a compass angle in degrees [0, 360).
      0°→right, 90°→up, 180°→left, 270°→down  (image-space corrected).
    """
    return np.degrees(np.arctan2(-heading_vec[1], heading_vec[0])) % 360


# ── Scoring logic (unchanged) ──────────────────────────────────────────────────
def dynamic_threshold(relative_angle: float) -> float:
    """
    Angle-dependent lookahead distance.
    Straight ahead → look far; sharp turn → look close.
    """
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
    """
    Weighted score: lower = better candidate.
    Penalises both sharp angles and close distances (non-linearly).
    Returns infinity for points outside the ±90° forward field of view.
    """
    if abs(relative_angle) > 90:
        return float('inf')
    dist = max(LOW_THRESHOLD, min(HIGH_THRESHOLD, dist))
    normalized_dist  = (dist - LOW_THRESHOLD) / (HIGH_THRESHOLD - LOW_THRESHOLD)
    normalized_angle = (abs(relative_angle) / 90) ** 2.5
    return ANGLE_FAVOR * normalized_dist + (1 - ANGLE_FAVOR) * normalized_angle


def map_angle_to_servo(relative_angle: float, dist: float):
    """
    Maps a relative angle and distance to a normalized servo command.
    Returns a float in [-0.5, 0.5]:
      -0.5 → full left  (-45° physical)
       0.0 → straight
      +0.5 → full right (+45° physical)
    Returns None for points beyond ±90° (behind the vehicle).
    """
    if abs(relative_angle) > 90:
        return None
    dist = max(LOW_THRESHOLD, min(HIGH_THRESHOLD, dist))
    normalized_dist  = (HIGH_THRESHOLD - dist) / (HIGH_THRESHOLD - LOW_THRESHOLD)
    normalized_angle = (90 - abs(relative_angle)) / 90
    weight = (normalized_angle * normalized_dist) ** WEIGHT
    servo = -weight * MAX_SERVO if relative_angle > 0 else weight * MAX_SERVO
    return float(np.clip(servo, -MAX_SERVO, MAX_SERVO))


# ── Best boundary-point selection (unchanged) ──────────────────────────────────
def select_best_boundary_point(car_pos: tuple, car_heading_angle: float,
                                boundary_points: list):
    """
    Iterates over sampled boundary points, computes each point's
    relative angle and distance from the car, applies dynamic_threshold
    to filter backward/far points, and returns the lowest-score candidate.

    Returns (best_point, best_angle, best_dist) or (None, None, None).
    """
    best_point = best_angle = best_dist = None
    best_score = float('inf')
    center = np.array(car_pos, dtype=float)

    for pt in boundary_points:
        direction = np.array(pt, dtype=float) - center
        dist = np.linalg.norm(direction)
        if dist < 1e-3:
            continue
        point_angle    = np.degrees(np.arctan2(-direction[1], direction[0]))
        relative_angle = (car_heading_angle - point_angle + 360) % 360
        if relative_angle > 180:
            relative_angle -= 360
        if dist < dynamic_threshold(relative_angle):
            score = compute_point_score(relative_angle, dist)
            if score < best_score:
                best_point = pt
                best_angle = relative_angle
                best_dist  = dist
                best_score = score

    return best_point, best_angle, best_dist


# ── Drawing helpers ────────────────────────────────────────────────────────────
# def draw_trioval(frame: np.ndarray, lane: dict) -> np.ndarray:
#     """
#     Replaces draw_lane. Draws the fitted tri-oval track as three layers:
#       1. Semi-transparent green fill between inner and outer offset curves
#       2. Outer boundary (yellow  — mirrors original 'right' colour)
#       3. Inner boundary (cyan    — mirrors original 'left'  colour)
#     Also marks the 6 anchor positions with small magenta circles.
#     """
#     if 'curve' not in lane:
#         return frame

#     curve = lane['curve']
#     inner = compute_curve_offset(curve, -TRACK_HALF_WIDTH)
#     outer = compute_curve_offset(curve,  TRACK_HALF_WIDTH)

#     inner_arr = np.array(inner, dtype=np.int32)
#     outer_arr = np.array(outer, dtype=np.int32)

#     # 1. Green fill band between outer (forward) and inner (reversed)
#     overlay = frame.copy()
#     poly    = np.concatenate([outer_arr, inner_arr[::-1]])
#     cv2.fillPoly(overlay, [poly], (0, 200, 0))
#     cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

#     # 2. Outer boundary — yellow
#     cv2.polylines(frame, [outer_arr], True, (0, 255, 255), 2)

#     # 3. Inner / reference boundary — cyan
#     cv2.polylines(frame, [inner_arr], True, (255, 255, 0), 2)

#     return frame

def draw_trioval(frame: np.ndarray, lane: dict) -> np.ndarray:
    """
    Draws the fitted tri-oval as two layers:
      1. Semi-transparent green fill — interior of the closed curve polygon
      2. Single reference line (cyan) — the path the car follows
    Also marks each anchor marker with a small magenta circle.
    """
    if 'curve' not in lane:
        return frame

    curve     = lane['curve']
    curve_arr = np.array(curve, dtype=np.int32)

    # 1. Filled interior
    overlay = frame.copy()
    cv2.fillPoly(overlay, [curve_arr], (0, 200, 0))
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    # 2. Single reference line
    cv2.polylines(frame, [curve_arr], True, (0, 255, 255), 2)

    return frame

def draw_cars(frame: np.ndarray, cars: dict, lane: dict) -> np.ndarray:
    """
    Unchanged visuals. Projection now targets lane['left'] which is
    aliased to the trioval reference curve — no logic change needed.
    """
    for rear_id, cd in cars.items():
        front, rear, mid = cd['front'], cd['rear'], cd['midpoint']

        cv2.circle(frame, rear, 8, (0, 0, 255), -1)
        # cv2.putText(frame, f"R{rear_id}", (rear[0]+10, rear[1]),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        if front is not None:
            cv2.circle(frame, front, 8, (0, 255, 0), -1)
            # cv2.putText(frame, "F", (front[0]+10, front[1]),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.line(frame, rear, front, (255, 0, 255), 2)

        cv2.circle(frame, mid, 6, (0, 165, 255), -1)

        # # Projection from midpoint onto nearest trioval segment
        # if 'left' in lane and len(lane['left']) > 1:
        #     left_pts = np.array(lane['left'], dtype=float)
        #     idx      = int(np.argmin(np.linalg.norm(left_pts - np.array(mid), axis=1)))
        #     seg_a    = lane['left'][max(idx - 1, 0)]
        #     seg_b    = lane['left'][min(idx + 1, len(lane['left']) - 1)]
        #     proj     = project_point_to_line(mid, seg_a, seg_b)
        #     cv2.circle(frame, proj, 5, (0, 120, 255), -1)
        #     cv2.line(frame, mid, proj, (255, 0, 255), 1)

    return frame


def draw_telemetry(frame: np.ndarray, car_id: int, cars: dict,
                   servo: float, motor: float,
                   n_markers: int) -> np.ndarray:
    """right_mode / left_mode replaced by n_markers used in the trioval fit."""
    lines = [
        f"Car: {car_id}",
        f"Trioval anchors: {n_markers}/6",
        f"Servo: {servo:+.3f}",
        f"Motor: {motor:.2f}",
        f"Detected IDs: {sorted(cars.keys())}",
    ]
    for idx, text in enumerate(lines):
        cv2.putText(frame, text, (10, 30 + idx*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


# ── Main function ──────────────────────────────────────────────────────────────
def run(frame: np.ndarray, car_id: int):
    """
    Process one camera frame and return (servo_angle, motor_speed, annotated_frame).

    Control pipeline
    ────────────────
    1.  Detect 4x4 car markers + 6x6 lane markers.
    2.  Build per-car geometry (midpoint, heading vector).
    3.  Fit a closed tri-oval polar Fourier curve through all 6x6 markers.
    4.  For each of the TRIOVAL_SAMPLES points: compute relative angle +
        distance from the car midpoint.
    5.  Filter with dynamic_threshold (forward-only, angle-dependent range).
    6.  Score with compute_point_score (angle+distance weighted trade-off).
    7.  Map the best candidate to a servo command via map_angle_to_servo.
    8.  Throttle motor speed inversely proportional to steering intensity.
    9.  Draw trioval track band, cars, target point, and telemetry overlay.
    """
    car_frame    = frame.copy()
    current_time = time.time()

    # Steps 1–3
    car_markers, lane_markers = detect_all_markers(car_frame)
    cars = identify_cars(car_markers, car_id)
    update_tracker(cars, current_time)
    lane = detect_trioval(lane_markers)         # ← replaces detect_lane_curves

    servo = 0.0
    motor = 0.0

    if car_id in cars and 'left' in lane:
        car_ref           = cars[car_id]['front'] or cars[car_id]['midpoint']
        car_heading_angle = heading_to_angle(cars[car_id]['heading'])

        # Steps 4–7
        best_point, best_angle, best_dist = select_best_boundary_point(
            car_ref, car_heading_angle, lane['left']
        )

        if best_point is not None:
            cv2.circle(car_frame, tuple(best_point), 7, (0, 0, 0), -1)
            cv2.line(car_frame, car_ref, tuple(best_point), (255, 255, 255), 2)

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
        else:
            servo = 0.0
            motor = 0.0
            last_sent_angles[car_id] = 0.0

        # Step 8
        turn_strength = abs(servo) / MAX_SERVO
        motor = float(np.clip(
            MAX_SPEED * (1.0 - 0.5 * turn_strength), MIN_SPEED, MAX_SPEED))

    # Step 9
    car_frame = draw_trioval(car_frame, lane)   # ← replaces draw_lane
    car_frame = draw_cars(car_frame, cars, lane)
    car_frame = draw_telemetry(car_frame, car_id, cars, servo, motor,
                               lane.get('n_markers', 0))

    return round(servo, 2), round(motor, 2), car_frame


# # Just for testing purposes
# def init_camera(cam_index=0):
#     if os.name == "nt":
#         cap = cv2.VideoCapture(cam_index)
#     elif os.name == "posix":
#         cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

#     if not cap.isOpened():
#         sys.exit("Camera not found!\n")

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

#     print("Camera opened successfully!\n")
#     return cap


# # Just for testing purposes
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
