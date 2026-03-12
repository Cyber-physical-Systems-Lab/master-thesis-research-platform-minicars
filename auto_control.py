import cv2
import sys
import os
import numpy as np
from cv2 import aruco
from threading import Lock

# ─────────────────────────────────────────────
# CONFIG / TUNING CONSTANTS
# ─────────────────────────────────────────────
CAR_DICT   = aruco.DICT_4X4_50    # 4×4 markers  → RC car (front + rear share same ID)
LANE_DICT  = aruco.DICT_6X6_250   # 6×6 markers  → lane boundary corners

# Lane boundary marker ID groups
LANE_GROUP_LEFT  = (198, 199)      # two markers that form the LEFT boundary
# LANE_GROUP_LEFT  = (196, 197)     # two markers that form the LEFT boundary
# LANE_GROUP_RIGHT = (198, 199)     # two markers that form the RIGHT boundary
LANE_GROUP_RIGHT = (196, 197)     # two markers that form the RIGHT boundary

# Speed / servo limits
MIN_SPEED  = 0.45
MAX_SPEED  = 0.60
MAX_SERVO  = 0.50

# Lane-following gains
CTE_K       = 1.0 / 160.0   # cross-track error (pixels) → servo normalised [-1,1]
HEADING_K   = 1.0 / 45.0    # heading error (degrees)    → servo normalised [-1,1]
CTE_W       = 0.45           # weight of CTE    contribution to servo
HEADING_W   = 0.55           # weight of heading contribution to servo

# Speed reduction near lane edges (fraction of half-lane-width that triggers slow-down)
EDGE_SLOW_FRACTION = 0.65    # if |CTE| > 65 % of half-width → start slowing
EDGE_MIN_SPEED_MUL = 0.55    # slowest multiplier when almost at the boundary

# Curve shape parameters (tunable)
CURVE_SAMPLES      = 60      # number of points sampled along each boundary polyline
CURVE_SMOOTHING    = 0.30    # 0 = straight lines between markers, 1 = very smooth spline

# Smoothing for output commands
SMOOTH_ALPHA = 0.40

CV_LOCK = Lock()

# ─────────────────────────────────────────────
# Per-car persistent state
# ─────────────────────────────────────────────
class CarState:
    def __init__(self):
        self.last_servo = 0.0
        self.last_speed = MIN_SPEED
        self.last_frame = None


_car_states: dict[int, CarState] = {}


# ─────────────────────────────────────────────
# Low-level ArUco detection
# ─────────────────────────────────────────────
def _detect_markers(frame, dict_type: int):
    """Return (corners_list, ids_array) for *dict_type* dictionary."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adict   = aruco.getPredefinedDictionary(dict_type)
    params  = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(adict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def detect_car(frame, car_id: int):
    """
    Detect front & rear markers for *car_id* (4x4 dict).
    Returns (front_xy, rear_xy, midpoint_xy) or (None, None, None).
    Assumes smaller-y centre = front.
    """
    corners, ids = _detect_markers(frame, CAR_DICT)
    if ids is None:
        return None, None, None

    centres = []
    for c, i in zip(corners, ids.flatten()):
        if int(i) == car_id:
            centres.append(np.mean(c[0], axis=0))

    if len(centres) < 2:
        return None, None, None

    p1, p2 = np.array(centres[0]), np.array(centres[1])
    front, rear = (p1, p2) if p1[1] < p2[1] else (p2, p1)
    midpoint = 0.5 * (front + rear)
    return front, rear, midpoint


def detect_lane_markers(frame):
    """
    Detect 6x6 lane markers. Returns dict: marker_id → (x, y).
    """
    corners, ids = _detect_markers(frame, LANE_DICT)
    result = {}
    if ids is None:
        return result
    for c, i in zip(corners, ids.flatten()):
        result[int(i)] = np.mean(c[0], axis=0).astype(np.float32)
    return result


# ─────────────────────────────────────────────
# Lane geometry
# ─────────────────────────────────────────────
def _smooth_polyline(p0: np.ndarray, p1: np.ndarray,
                     n: int = CURVE_SAMPLES,
                     smooth: float = CURVE_SMOOTHING) -> np.ndarray:
    """
    Generate a smoothly curved polyline between two points.

    *smooth* in [0, 1]:
      0   → straight line (lerp)
      >0  → quadratic Bézier where the control point is offset
             perpendicular to the segment by *smooth* x half-length.

    Returns shape (n, 2).
    """
    t = np.linspace(0.0, 1.0, n)
    if smooth < 1e-3:
        pts = np.outer(1 - t, p0) + np.outer(t, p1)
        return pts.astype(np.float32)

    # midpoint control point, offset perpendicularly
    mid  = 0.5 * (p0 + p1)
    seg  = p1 - p0
    perp = np.array([-seg[1], seg[0]], dtype=np.float32)
    perp_norm = np.linalg.norm(perp)
    if perp_norm > 1e-6:
        perp = perp / perp_norm * np.linalg.norm(seg) * smooth * 0.5
    ctrl = mid + perp

    # Quadratic Bézier: B(t) = (1-t)²·P0 + 2(1-t)t·ctrl + t²·P1
    pts = (
        np.outer((1 - t) ** 2, p0)
        + np.outer(2 * (1 - t) * t, ctrl)
        + np.outer(t ** 2, p1)
    )
    return pts.astype(np.float32)


def build_lane_boundaries(marker_map: dict):
    """
    Given detected marker centres, build left and right boundary polylines.

    Each boundary consists of the two markers from its ID-group ordered
    by their x-coordinate (left marker first), connected by a smooth curve.

    Returns (left_pts, right_pts) each as np.ndarray of shape (N, 2),
    or (None, None) if not enough markers are visible.
    """
    def group_pts(group_ids):
        pts = [marker_map[i] for i in group_ids if i in marker_map]
        if len(pts) < 2:
            return None
        pts.sort(key=lambda p: p[0])        # left → right by x
        return _smooth_polyline(pts[0], pts[1])

    left_pts  = group_pts(LANE_GROUP_LEFT)
    right_pts = group_pts(LANE_GROUP_RIGHT)
    return left_pts, right_pts


def lane_centre_at(left_pts: np.ndarray, right_pts: np.ndarray,
                   query_x: float) -> tuple[np.ndarray | None, float | None]:
    """
    At a given x position, interpolate the lane centre and half-width.
    Returns (centre_xy, half_width) or (None, None).
    """
    def interp_y_at_x(pts, x):
        xs = pts[:, 0]
        ys = pts[:, 1]
        if x < xs.min() or x > xs.max():
            return None
        return float(np.interp(x, xs, ys))

    ly = interp_y_at_x(left_pts,  query_x)
    ry = interp_y_at_x(right_pts, query_x)
    if ly is None or ry is None:
        # fallback: use nearest boundary points by overall distance
        ly = left_pts[np.argmin(np.abs(left_pts[:, 0] - query_x)), 1]
        ry = right_pts[np.argmin(np.abs(right_pts[:, 0] - query_x)), 1]

    centre_y   = 0.5 * (ly + ry)
    half_width = 0.5 * abs(ry - ly)
    return np.array([query_x, centre_y], dtype=np.float32), half_width


def lane_direction_at(left_pts: np.ndarray, right_pts: np.ndarray,
                      query_x: float) -> np.ndarray:
    """
    Estimate the lane forward direction (unit vector) at *query_x* by
    differencing centre positions slightly ahead and behind.
    """
    dx = 20.0
    c_fwd, _  = lane_centre_at(left_pts, right_pts, query_x + dx)
    c_bwd, _  = lane_centre_at(left_pts, right_pts, query_x - dx)
    if c_fwd is None or c_bwd is None:
        return np.array([1.0, 0.0], dtype=np.float32)   # assume rightward travel
    vec = c_fwd - c_bwd
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.array([1.0, 0.0])


# ─────────────────────────────────────────────
# Control computation
# ─────────────────────────────────────────────
#TODO try to modify to a similar version from Ibrahim's thesis and try out with this markers and the left one as guidance
def compute_control_v2(front, rear, midpoint,
                    left_pts, right_pts,
                    state: CarState):
    """
    Compute servo_cmd and speed_cmd from lane geometry + car pose.

    Cross-track error is computed as the signed perpendicular distance from
    the car's midpoint (between its two 4x4 markers) to the local lane
    centre line, along the lane normal.
    """

    debug = {'centre': None, 'half_width': None, 'cte': None, 'heading_err': None}

    # Lane centre and half-width at the car's x-position
    centre, half_width = lane_centre_at(left_pts, right_pts, float(midpoint[0]))
    if centre is None:
        return 0.0, MIN_SPEED, debug

    # ── Local lane direction (unit vector, "forward" along corridor)
    lane_dir = lane_direction_at(left_pts, right_pts, float(midpoint[0]))
    lane_dir_norm = np.linalg.norm(lane_dir)
    if lane_dir_norm < 1e-6:
        lane_dir = np.array([1.0, 0.0], dtype=np.float32)
    else:
        lane_dir = lane_dir / lane_dir_norm

    # Left-normal to lane direction (image coordinates)
    # lane_dir = [dx, dy]; left_normal = [-dy, dx]
    left_normal = np.array([-lane_dir[1], lane_dir[0]], dtype=np.float32)

    # ── Signed cross-track error:
    # positive if car is on the left side of the lane centre (w.r.t lane_dir),
    # negative if on the right.
    vec_c2m = midpoint.astype(np.float32) - centre.astype(np.float32)
    cte     = float(np.dot(vec_c2m, left_normal))
    debug['centre']     = centre
    debug['half_width'] = half_width
    debug['cte']        = cte

    # ── Heading error using front/rear line from the two 4×4 markers
    heading_vec  = front.astype(np.float32) - rear.astype(np.float32)
    hv_norm      = np.linalg.norm(heading_vec)
    if hv_norm < 1e-6:
        return 0.0, MIN_SPEED, debug
    heading_dir   = heading_vec / hv_norm
    heading_angle = np.degrees(np.arctan2(-heading_dir[1],  heading_dir[0]))
    lane_angle    = np.degrees(np.arctan2(-lane_dir[1],     lane_dir[0]))
    heading_err   = ((lane_angle - heading_angle) + 180.0) % 360.0 - 180.0
    debug['heading_err'] = heading_err

    # ── Servo command: blend perpendicular CTE and heading error
    cte_norm        = np.clip(cte * CTE_K,              -1.0, 1.0)
    heading_norm_ct = np.clip(heading_err * HEADING_K,  -1.0, 1.0)

    # Negative sign because positive CTE (car left of centre) should steer right
    servo_raw = -(CTE_W * cte_norm + HEADING_W * heading_norm_ct)
    servo_raw = float(np.clip(servo_raw, -1.0, 1.0)) * MAX_SERVO

    # Small dead-band so it actually tracks the centre line between the two 4×4 markers
    if abs(heading_err) < 4 and abs(cte) < 5:
        servo_raw = 0.0
    elif abs(heading_err) < 2:
        servo_raw *= 0.2

    # ── Speed: slow near any boundary (using cross-track distance to half-width)
    if half_width > 1e-3:
        edge_ratio = abs(cte) / half_width
    else:
        edge_ratio = 0.0

    if edge_ratio > EDGE_SLOW_FRACTION:
        t_edge    = (edge_ratio - EDGE_SLOW_FRACTION) / (1.0 - EDGE_SLOW_FRACTION + 1e-9)
        speed_mul = 1.0 - t_edge * (1.0 - EDGE_MIN_SPEED_MUL)
    else:
        speed_mul = 1.0

    base_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (1.0 - edge_ratio)
    speed_raw  = float(np.clip(base_speed, MIN_SPEED, MAX_SPEED)) * speed_mul

    # ── Temporal smoothing
    servo_sm = SMOOTH_ALPHA * state.last_servo + (1.0 - SMOOTH_ALPHA) * servo_raw
    speed_sm = SMOOTH_ALPHA * state.last_speed + (1.0 - SMOOTH_ALPHA) * speed_raw
    state.last_servo = servo_sm
    state.last_speed = speed_sm

    servo_out = float(np.clip(servo_sm, -MAX_SERVO, MAX_SERVO))
    speed_out = float(np.clip(speed_sm,  MIN_SPEED,  MAX_SPEED))
    return servo_out, speed_out, debug

def compute_control(front, rear, midpoint,
                    left_pts, right_pts,
                    state: CarState):
    """
    Compute servo_cmd and speed_cmd from lane geometry + car pose.

    Returns (servo, speed, debug_dict).
    """
    debug = {'centre': None, 'half_width': None, 'cte': None, 'heading_err': None}

    centre, half_width = lane_centre_at(left_pts, right_pts, float(midpoint[0]))
    if centre is None:
        return 0.0, MIN_SPEED, debug

    # ── Cross-track error (signed: + = car is above/left of centre in image)
    cte = float(midpoint[1] - centre[1])   # image y increases downward
    debug['centre']     = centre
    debug['half_width'] = half_width
    debug['cte']        = cte

    # ── Heading error
    heading_vec  = front.astype(np.float32) - rear.astype(np.float32)
    heading_norm = np.linalg.norm(heading_vec)
    if heading_norm < 1e-6:
        return 0.0, MIN_SPEED, debug
    heading_dir   = heading_vec / heading_norm
    heading_angle = np.degrees(np.arctan2(-heading_dir[1],  heading_dir[0]))

    lane_dir   = lane_direction_at(left_pts, right_pts, float(midpoint[0]))
    lane_angle = np.degrees(np.arctan2(-lane_dir[1], lane_dir[0]))

    heading_err = ((lane_angle - heading_angle) + 180.0) % 360.0 - 180.0
    debug['heading_err'] = heading_err

    # ── Servo
    cte_norm     = np.clip(cte * CTE_K,     -1.0, 1.0)
    heading_norm_ = np.clip(heading_err * HEADING_K, -1.0, 1.0)
    servo_raw    = -(CTE_W * cte_norm + HEADING_W * heading_norm_)
    servo_raw    = float(np.clip(servo_raw, -1.0, 1.0)) * MAX_SERVO

    # dead-band
    if abs(heading_err) < 4 and abs(cte) < 8:
        servo_raw = 0.0
    elif abs(heading_err) < 2:
        servo_raw *= 0.15

    # ── Speed (reduce near edges)
    if half_width > 1e-3:
        edge_ratio = abs(cte) / half_width
    else:
        edge_ratio = 0.0

    if edge_ratio > EDGE_SLOW_FRACTION:
        t_edge    = (edge_ratio - EDGE_SLOW_FRACTION) / (1.0 - EDGE_SLOW_FRACTION + 1e-9)
        speed_mul = 1.0 - t_edge * (1.0 - EDGE_MIN_SPEED_MUL)
    else:
        speed_mul = 1.0

    speed_raw = float(np.clip(MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (1.0 - edge_ratio),
                               MIN_SPEED, MAX_SPEED)) * speed_mul

    # ── Smooth
    servo_sm = SMOOTH_ALPHA * state.last_servo + (1.0 - SMOOTH_ALPHA) * servo_raw
    speed_sm = SMOOTH_ALPHA * state.last_speed + (1.0 - SMOOTH_ALPHA) * speed_raw
    state.last_servo = servo_sm
    state.last_speed = speed_sm

    servo_out = float(np.clip(servo_sm, -MAX_SERVO, MAX_SERVO))
    speed_out = float(np.clip(speed_sm,  MIN_SPEED,  MAX_SPEED))
    return servo_out, speed_out, debug

# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────
def draw_lane_and_debug_v2(frame, left_pts, right_pts,
                        front, rear, midpoint, debug: dict):
    vis = frame.copy()

    # ── Draw boundary curves
    if left_pts is not None:
        pts = left_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], False, (0, 255, 0), 2)    # green = left

    if right_pts is not None:
        pts = right_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], False, (0, 0, 255), 2)    # red   = right

    # ── Filled lane overlay (semi-transparent)
    if left_pts is not None and right_pts is not None:
        lane_poly = np.vstack([left_pts, right_pts[::-1]]).astype(np.int32)
        overlay   = vis.copy()
        cv2.fillPoly(overlay, [lane_poly], (0, 200, 80))
        cv2.addWeighted(overlay, 0.20, vis, 0.80, 0, vis)

    # ── Draw car markers
    if front is not None:
        cv2.circle(vis, tuple(front.astype(int)),    7, (0, 140, 255), -1)   # orange = front
        cv2.circle(vis, tuple(rear.astype(int)),     7, (255, 0,   0), -1)   # blue   = rear
        cv2.circle(vis, tuple(midpoint.astype(int)), 7, (0, 255, 255), -1)   # yellow = mid
        cv2.line(vis, tuple(front.astype(int)), tuple(rear.astype(int)),
                 (200, 200, 200), 2)

    # ── Draw lane centre point and CTE line
    if debug.get('centre') is not None and midpoint is not None:
        c = tuple(debug['centre'].astype(int))
        cv2.circle(vis, c, 6, (255, 100, 0), -1)                           # teal = centre
        cv2.line(vis, tuple(midpoint.astype(int)), c, (0, 255, 255), 2)    # CTE line

    # ── Telemetry text
    cte_val = debug.get('cte')
    h_err   = debug.get('heading_err')
    lines   = []
    if cte_val is not None:
        lines.append(f"CTE: {cte_val:+.1f}px")
    if h_err is not None:
        lines.append(f"Hdg err: {h_err:+.1f}deg")

    for idx, txt in enumerate(lines):
        cv2.putText(vis, txt, (10, 25 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    return vis

def draw_lane_and_debug(frame, left_pts, right_pts,
                        front, rear, midpoint, debug: dict):
    vis = frame.copy()

    # ── Filled lane overlay (semi-transparent)
    if left_pts is not None and right_pts is not None:
        lane_poly = np.vstack([left_pts, right_pts[::-1]]).astype(np.int32)
        overlay   = vis.copy()
        cv2.fillPoly(overlay, [lane_poly], (0, 200, 80))
        cv2.addWeighted(overlay, 0.20, vis, 0.80, 0, vis)

    # ── Draw boundary curves
    if left_pts is not None:
        pts = left_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], False, (0, 255, 0), 2)        # green = left

    if right_pts is not None:
        pts = right_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], False, (0, 0, 255), 2)        # red   = right

    # ── Draw lane centre polyline (midpoints between boundaries)
    if left_pts is not None and right_pts is not None:
        centre_pts = (0.5 * (left_pts + right_pts)).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [centre_pts], False, (255, 255, 0), 1)  # dashed yellow

    # ── Draw car markers & body line
    if front is not None and rear is not None and midpoint is not None:
        cv2.line(vis,
                 tuple(front.astype(int)),
                 tuple(rear.astype(int)),
                 (200, 200, 200), 2)
        cv2.circle(vis, tuple(rear.astype(int)),     8, (255,  60,  60), -1)  # red    = rear
        cv2.circle(vis, tuple(front.astype(int)),    8, (0,   140, 255), -1)  # orange = front
        cv2.circle(vis, tuple(midpoint.astype(int)), 8, (0,   255, 255), -1)  # cyan   = mid
        cv2.putText(vis, "F", tuple(front.astype(int) + np.array([8, -8])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        cv2.putText(vis, "R", tuple(rear.astype(int)  + np.array([8, -8])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60,  60), 2)
    else:
        # Warn visually when car markers are not found
        cv2.putText(vis, "CAR NOT DETECTED",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ── Draw lane centre point + CTE vector line
    if debug.get('centre') is not None and midpoint is not None:
        c = debug['centre'].astype(int)
        cv2.circle(vis, tuple(c), 7, (0, 165, 255), -1)                    # orange = lane centre
        cv2.line(vis, tuple(midpoint.astype(int)), tuple(c), (0, 255, 255), 2)  # CTE line

    # ── Telemetry HUD
    cte_val = debug.get('cte')
    h_err   = debug.get('heading_err')
    hw      = debug.get('half_width')
    inside_pct = ""
    if cte_val is not None and hw is not None and hw > 1e-3:
        inside_pct = f"  edge={abs(cte_val)/hw*100:.0f}%"

    hud_lines = [
        f"CTE:     {cte_val:+.1f} px{inside_pct}" if cte_val is not None else "CTE:     --",
        f"Hdg err: {h_err:+.1f} deg"              if h_err  is not None else "Hdg err: --",
    ]
    for idx, txt in enumerate(hud_lines):
        cv2.putText(vis, txt, (10, 25 + idx * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2)

    return vis

# ─────────────────────────────────────────────
# Public entry-point (called by external thread)
# ─────────────────────────────────────────────
def run_a(capture, car_id: int):
    """
    Capture one frame, detect lane + car, compute control, return results.

    Returns
    -------
    (motor_speed, servo_angle, modified_frame)
        motor_speed   : float in [MIN_SPEED, MAX_SPEED]
        servo_angle   : float in [-MAX_SERVO, MAX_SERVO]
        modified_frame: BGR image with all overlays drawn
    """
    global _car_states

    if car_id not in _car_states:
        _car_states[car_id] = CarState()
    state = _car_states[car_id]

    # ── Grab frame
    with CV_LOCK:
        ret, frame = capture.read()
    if not ret or frame is None:
        print(f"[Car {car_id}] Frame capture failed.")
        fallback = state.last_frame if state.last_frame is not None else np.zeros((480, 640, 3), np.uint8)
        return round(state.last_speed, 2), round(state.last_servo, 2), fallback

    # ── Detect ArUco markers
    marker_map          = detect_lane_markers(frame)
    front, rear, midpoint = detect_car(frame, car_id)

    # Draw raw ArUco detections (both dictionaries)
    for dtype in (LANE_DICT, CAR_DICT):
        c, i = _detect_markers(frame, dtype)
        if i is not None:
            aruco.drawDetectedMarkers(frame, c, i, (0, 255, 255))

    # ── Build lane boundaries
    left_pts, right_pts = build_lane_boundaries(marker_map)

    # ── Compute control
    servo, speed, debug = 0.0, MIN_SPEED, {}
    if left_pts is not None and right_pts is not None and midpoint is not None:
        servo, speed, debug = compute_control(front, rear, midpoint,
                                              left_pts, right_pts, state)
    else:
        # not enough info → coast straight
        state.last_servo = 0.0
        state.last_speed = MIN_SPEED

    # ── Annotate frame
    vis = draw_lane_and_debug(frame, left_pts, right_pts,
                              front, rear, midpoint, debug)

    h = vis.shape[0]
    cv2.putText(vis, f"Car {car_id} | spd={speed:.2f} servo={servo:+.2f}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 2)

    state.last_frame = vis.copy()

    # Just for debug - comment when not needed
    cv2.imshow("Curve", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit("\nQuitting the frame!")

    return round(speed, 2), round(servo, 2), vis

def run(capture, car_id: int):
    """
    Returns (motor_speed, servo_angle, modified_frame).
    """
    global _car_states

    if car_id not in _car_states:
        _car_states[car_id] = CarState()
    state = _car_states[car_id]

    # ── Grab frame
    with CV_LOCK:
        ret, frame = capture.read()
    if not ret or frame is None:
        print(f"[Car {car_id}] Frame capture failed.")
        fallback = (state.last_frame
                    if state.last_frame is not None
                    else np.zeros((480, 640, 3), np.uint8))
        return round(state.last_speed, 2), round(state.last_servo, 2), fallback

    # ── Draw raw ArUco detections for BOTH dictionaries onto frame (once)
    for dtype in (LANE_DICT, CAR_DICT):
        corners, ids = _detect_markers(frame, dtype)
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 255))
            # Label each marker ID in magenta
            for c, i in zip(corners, ids.flatten()):
                cx, cy = np.mean(c[0], axis=0).astype(int)
                cv2.putText(frame, f"id={int(i)}",
                            (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    # ── Detect lane boundaries
    marker_map          = detect_lane_markers(frame)
    left_pts, right_pts = build_lane_boundaries(marker_map)

    # ── Detect car
    front, rear, midpoint = detect_car(frame, car_id)

    # ── Compute control (only when all data available)
    servo, speed, debug = state.last_servo, state.last_speed, {}
    if (left_pts is not None and right_pts is not None
            and front is not None and rear is not None and midpoint is not None):
        servo, speed, debug = compute_control(front, rear, midpoint,
                                              left_pts, right_pts, state)
    else:
        # Log what's missing to help debug
        missing = []
        if left_pts  is None: missing.append("left boundary")
        if right_pts is None: missing.append("right boundary")
        if front     is None: missing.append("front marker")
        if rear      is None: missing.append("rear marker")
        if missing:
            print(f"[Car {car_id}] Waiting for: {', '.join(missing)}")

    # ── Annotate frame
    vis = draw_lane_and_debug(frame, left_pts, right_pts,
                              front, rear, midpoint, debug)

    # ── Telemetry bar (always shows live values, falls back to last known)
    h = vis.shape[0]
    lane_ok  = "OK"  if (left_pts  is not None and right_pts is not None) else "NO LANE"
    car_ok   = "OK"  if front is not None else "NO CAR"
    cv2.putText(vis,
                f"Car {car_id} | spd={speed:.2f}  servo={servo:+.2f} | lane={lane_ok} | car={car_ok}",
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    # Just for debug - comment when not needed
    cv2.imshow("Curve", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit("\nQuitting the frame!")

    state.last_frame = vis.copy()
    return round(speed, 2), round(servo, 2), vis

# ─────────────────────────────────────────────
# ── PRESERVED EXACTLY AS-IS ──────────────────
# ─────────────────────────────────────────────
def init_camera(cam_index=0):
    if os.name == "nt":
        cap = cv2.VideoCapture(cam_index)
    elif os.name == "posix":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        sys.exit("Camera not found!\n")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("Camera opened successfully!\n")
    return cap


if __name__ == '__main__':
    capture = init_camera(1)
    car_idx = 1
    while 1:
        run(capture, car_idx)
        # run_a(capture, car_idx)
    capture.release()
    cv2.destroyAllWindows()
