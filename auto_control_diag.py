import cv2
import sys
import os
import numpy as np
from cv2 import aruco
from threading import Lock


# -------------------------
# CONFIG / TUNING CONSTANTS
# -------------------------
AREA_DICT = aruco.DICT_6X6_250    # dictionary used for area corner markers
CAR_DICT = aruco.DICT_4X4_50      # dictionary used for car markers (front/back)
OBJ_DICT = aruco.DICT_5X5_100     # dictionary used for object detection

MIN_SPEED = 0.45
MAX_SPEED = 0.6
MAX_SERVO = 0.5

TARGET_RADIUS_PIX = 120            # pixels threshold to consider target reached
# OBJECT_DISTANCE = 160
STEER_K_DEG_TO_NORM = 1.0 / 90.0  # convert degrees->[-1,1]
# STEER_K = 1.0 / 90.0  # convert degrees->[-1,1]
INSIDE_PENALTY = 0.5              # speed multiplier when outside area
# SMOOTH = 0.5                # smoothing factor for speed/servo (0..1)
SMOOTH_ALPHA = 0.5                # smoothing factor for speed/servo (0..1)

# thread-safe camera lock
CV_LOCK = Lock()

# persistent navigator state per car_id
_navigators = {}

# -------------------------
# Navigator class
# -------------------------
class AutonomousNavigator:
    def __init__(self, car_id):
        self.car_id = car_id
        self.ar_area = None           # logical map: 0..3 -> np.array([x,y])
        self.area_contour = None      # Nx1x2 int32 contour for polygon tests
        self.target_index = 0         # index into NAV_ORDER
        self.last_servo = 0.0
        self.last_speed = 0.0
        self.last_frame = None
        self.frame_shape = None
        self.detected_obstacles = []
        self.diagonal_mode = 'A'
        self.diag_dir = 1             # 1: A -> B, -1: B -> A

    def set_area_from_markers_spatial(self, markers6):
        """
        markers6: dict real_id -> (x,y)
        Assign logical corners by spatial arrangement:
          0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left
        """
        if len(markers6) < 4:
            return False

        # Convert to list (real_id, np.array([x,y]))
        pts = [(rid, np.array(coord, dtype=np.float32)) for rid, coord in markers6.items()]

        # Sort by y (top -> bottom)
        pts_by_y = sorted(pts, key=lambda p: p[1][1])
        top_two = pts_by_y[:2]
        bottom_two = pts_by_y[-2:]

        # order left->right by x inside each row
        top_left, top_right = sorted(top_two, key=lambda p: p[1][0])
        bottom_left, bottom_right = sorted(bottom_two, key=lambda p: p[1][0])

        ordered = {
            0: top_left[1],
            1: top_right[1],
            2: bottom_right[1],
            3: bottom_left[1],
        }
        self.ar_area = ordered
        pts_poly = np.array([ordered[i] for i in range(0,4)], dtype=np.int32).reshape((-1,1,2))
        self.area_contour = pts_poly

        return True
    
    def set_diagonal_mode(self, mode):
        """
        mode: 'A' or 'B'
        A = diagonal 0 -> 2
        B = diagonal 1 -> 3
        """
        if mode not in ('A', 'B'):
            return
        self.diagonal_mode = mode
        self.diag_dir = 1  # reset direction forward


# -------------------------
# Detection helpers
# -------------------------
def detect_area_markers(frame):
    """
    Detect any 6x6 markers (area corners). Returns dict real_id->(x,y)
    """
    corners, ids = detect_any_marker(frame, AREA_DICT)

    area_centers = {}
    if ids is None:
        return area_centers
    
    for c, idarr in zip(corners, ids):
        real_id = int(idarr[0])
        center = np.mean(c[0], axis=0)
        area_centers[real_id] = (float(center[0]), float(center[1]))
    return area_centers

def detect_car_markers_for_id(frame, car_id):
    """
    Detect car markers and return front, rear, midpoint for the given car_id.
    Car markers are 4x4; both front+rear share the same marker ID (car_id).
    Returns (front (x,y), rear (x,y), midpoint (x,y), annotated_frame)
    """
    corners, ids = detect_any_marker(frame, CAR_DICT)

    if ids is None:
        return None, None, None

    centers = [np.mean(c[0], axis=0) for c in corners]
    # group by id
    group = {}
    for idarr, center in zip(ids.flatten(), centers):
        group.setdefault(int(idarr), []).append(center)

    if car_id not in group or len(group[car_id]) < 2:
        return None, None, None

    p1, p2 = group[car_id][:2]
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    # decide front vs rear — assume smaller y is toward top of image => front
    if p1[1] < p2[1]:
        front, rear = p1, p2
    else:
        front, rear = p2, p1

    midpoint = 0.5 * (front + rear)

    # draw visuals for car markers
    vis = frame.copy()
    cv2.circle(vis, tuple(front.astype(int)), 6, (0, 165, 255), -1)  # orange front
    cv2.circle(vis, tuple(rear.astype(int)), 6, (255, 0, 0), -1)    # blue rear
    cv2.circle(vis, tuple(midpoint.astype(int)), 6, (0,255,255), -1) # yellow midpoint
    cv2.line(vis, tuple(front.astype(int)), tuple(rear.astype(int)), (0,165,255), 2)

    return front, rear, midpoint

def detect_obstacles(frame):
    """
    Detect obstacles marked with 5x5 ArUco patterns.
    Returns a list of dicts: [{id, center(x,y)}].
    """
    corners, ids = detect_any_marker(frame, OBJ_DICT)
    
    obstacles = []
    if ids is None:
        return obstacles

    for c, idarr in zip(corners, ids):
        center = np.mean(c[0], axis=0)
        obstacles.append({
            "id": int(idarr[0]),
            "center": np.array(center, dtype=np.float32)
        })
    
    return obstacles

def detect_any_marker(frame, TYPE_DICT):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    odict = aruco.getPredefinedDictionary(TYPE_DICT)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(odict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 255))
    
    return corners, ids

# -------------------------
# Geometry helpers
# -------------------------
def point_inside_polygon(contour, pt):
    """Return True if pt is inside polygon contour (Nx1x2)."""
    if contour is None:
        return False
    val = cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False)
    return val >= 0

def estimate_local_tangent_and_closest(contour, midpoint, window=6):
    """
    For a thin polyline-like contour, compute tangent vector and 
    the closest contour point to midpoint.
    Returns (tangent_vector (2,), closest_point (x,y)) or (None,None).
    """
    if contour is None or midpoint is None:
        return None, None
    pts = contour.reshape(-1,2).astype(np.float32)
    d = np.linalg.norm(pts - midpoint, axis=1)
    idx = int(np.argmin(d))
    prev_idx = max(idx - window, 0)
    next_idx = min(idx + window, len(pts) - 1)
    pt_prev = pts[prev_idx]
    pt_next = pts[next_idx]
    tangent = pt_next - pt_prev
    if np.linalg.norm(tangent) < 1e-6:
        return None, tuple(pts[idx].astype(int))
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
    return tangent, tuple(pts[idx].astype(int))

def angle_diff_signed_deg(a_deg, b_deg):
    """Smallest signed difference a - b in degrees mapped to [-180,180]."""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return d

def clamp_servo_and_speed(servo, speed):
    """Clamp to allowed ranges."""
    s = float(np.clip(servo, -MAX_SERVO, MAX_SERVO))
    v = float(np.clip(speed, MIN_SPEED, MAX_SPEED))
    return s, v

# -------------------------
# Control logic and computations
# -------------------------
def compute_obstacle_avoidance(midpoint, obstacles, A, B):
    """
    Returns a steering_offset_deg and speed_multiplier.
    Steering is positive = steer right, negative = steer left.
    """
    if midpoint is None or len(obstacles) == 0:
        return 0.0, 1.0  # no offset, full speed

    # Pick closest object to the midpoint
    distances = [np.linalg.norm(o["center"] - midpoint) for o in obstacles]
    closest = obstacles[int(np.argmin(distances))]
    dist = np.min(distances)
    obj = closest["center"]

    # Compute relative lateral direction
    AB = B - A
    lane_dir = AB / (np.linalg.norm(AB) + 1e-9)
    left_vec = np.array([-lane_dir[1], lane_dir[0]])
    rel = obj - midpoint
    lateral_dist = np.dot(rel, left_vec)  # >0 = object on left, <0 = right

    # STEERING OFFSET DEGREE
    steer_strength = np.interp(dist, [200, 50], [0, 35])  # up to 35° of avoidance
    if dist > 200:
        steer_strength = 0.0
    steer_offset_deg = steer_strength if lateral_dist < 0 else -steer_strength

    # SPEED MULTIPLIER
    speed_mul = np.interp(dist, [300, 80], [1.0, 0.3])
    speed_mul = float(np.clip(speed_mul, 0.3, 1.0))

    return steer_offset_deg, speed_mul

# def compute_control_simple(contour, front, rear, midpoint, nav):
#     """
#     Simplified diagonal-following + obstacle avoidance
#     """

#     info = {'closest_point': None, 'tangent_vec': None,
#             'target_pt': None, 'inside': False, 'lane_point': None}

#     if contour is None or front is None or rear is None or midpoint is None:
#         return 0.0, 0.0, info

#     # -----------------------------
#     # 1) Determine diagonal endpoints
#     # -----------------------------
#     if nav.diagonal_mode == "A":
#         A = nav.ar_area[0].astype(np.float32)
#         B = nav.ar_area[2].astype(np.float32)
#     else:
#         A = nav.ar_area[1].astype(np.float32)
#         B = nav.ar_area[3].astype(np.float32)

#     # forward/backward logic
#     target = B if nav.diag_dir > 0 else A
#     # target_pt = target_pt.astype(np.float32)
#     info['target_pt'] = tuple(target.astype(int))

#     # -----------------------------
#     # 2) Flip direction when close
#     # -----------------------------
#     dist_to_target = np.linalg.norm(midpoint - target)
#     if dist_to_target < TARGET_RADIUS_PIX:
#         nav.diag_dir *= -1
#         target = A if nav.diag_dir > 0 else B
#         info['target_pt'] = tuple(target.astype(int))

#     # -----------------------------
#     # 3) Compute lane direction (unit vector along diagonal)
#     # -----------------------------
#     AB = B - A
#     lane_dir = AB / (np.linalg.norm(AB) + 1e-9)

#     # lane_point (orthogonal projection of midpoint onto diagonal)
#     # P_proj = A + dot(M-A , AB̂) * AB̂
#     t = np.dot(midpoint - A, lane_dir)
#     lane_point = A + t * lane_dir
#     info["lane_point"] = tuple(lane_point.astype(int))  # drawn as orange circle

#     # -----------------------------
#     # 4) Heading angle of car
#     # -----------------------------
#     heading_vec = front - rear
#     heading_dir = heading_vec / (np.linalg.norm(heading_vec) + 1e-9)

#     heading_angle = np.degrees(np.arctan2(-heading_dir[1], heading_dir[0]))
#     lane_angle = np.degrees(np.arctan2(-lane_dir[1], lane_dir[0]))

#     # steering error
#     steer_err = (lane_angle - heading_angle + 180) % 360 - 180
#     servo_cmd = np.clip(-steer_err * STEER_K, -1, 1) * MAX_SERVO

#     # -----------------------------
#     # 5) Speed (faster in middle, slower at endpoints)
#     # -----------------------------
#     diag_len = np.linalg.norm(A - B)
#     speed_norm = dist_to_target / (diag_len + 1e-9)
#     speed_cmd = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * speed_norm

#     # -----------------------------
#     # 6) If car is outside polygon → steer inward + slow down
#     # -----------------------------
#     inside = point_inside_polygon(contour, midpoint)
#     info["inside"] = inside
#     if not inside:
#         speed_cmd *= INSIDE_PENALTY

#         poly_centroid = contour.reshape(-1,2).mean(axis=0)
#         to_center = poly_centroid - midpoint

#         cent_ang = np.degrees(np.arctan2(-to_center[1], to_center[0]))
#         steer_err2 = (cent_ang - heading_angle + 180) % 360 - 180
#         servo_cmd = np.clip(-steer_err2 * STEER_K, -1, 1) * MAX_SERVO

#     # -----------------------------
#     # 7) Obstacle avoidance (simple version)
#     # -----------------------------
#     if len(nav.detected_obstacles) > 0:
#         dists = [np.linalg.norm(o["center"] - midpoint) for o in nav.detected_obstacles]
#         idx = int(np.argmin(dists))
#         obs = nav.detected_obstacles[idx]["center"]
#         d = np.min(dists)

#         info["closest_point"] = tuple(obs.astype(int))

#         if d < OBJECT_DISTANCE:  # only react when reasonably close
#             left_vec = np.array([-lane_dir[1], lane_dir[0]])
#             lat = np.dot(obs - midpoint, left_vec)

#             # steer away from obstacle
#             steer_strength = np.interp(d, [50, 200], [35, 0])
#             servo_cmd += np.sign(lat) * steer_strength * STEER_K * MAX_SERVO

#             # slow down when close
#             speed_cmd *= np.interp(d, [80, 300], [0.3, 1.0])

#             # tangent vector is perpendicular to obstacle radial interaction
#             # just for visualization
#             tangent = np.array([left_vec[0], left_vec[1]])  # consistent sign
#             info["tangent_vec"] = tangent

#     # -----------------------------
#     # 8) Smoothing and clamping
#     # -----------------------------
#     servo_s = SMOOTH * nav.last_servo + (1 - SMOOTH) * servo_cmd
#     speed_s = SMOOTH * nav.last_speed + (1 - SMOOTH) * speed_cmd

#     nav.last_servo = servo_s
#     nav.last_speed = speed_s

#     servo_out = float(np.clip(servo_s, -MAX_SERVO, MAX_SERVO))
#     speed_out = float(np.clip(speed_s, MIN_SPEED, MAX_SPEED))

#     return servo_out, speed_out, info

def compute_control_area(contour, front, rear, midpoint, navigator):
    """
    Updated version:
      - Dynamic diagonal switching via navigator.diagonal_mode
      - Lane-like behavior: follow diagonal as a lane center
      - Smooth diagonal target transitions
    """

    info = {'closest_point': None, 'tangent_vec': None,
            'target_pt': None, 'inside': False, 'lane_point': None}

    if contour is None or midpoint is None or front is None or rear is None:
        return 0.0, 0.0, info

    # ensure contour
    if contour.ndim == 2:
        contour = contour.reshape((-1,1,2))

    # ---------------------------------------
    # Determine diagonal based on mode
    # ---------------------------------------
    if navigator.diagonal_mode == "A":
        A = navigator.ar_area[0]
        B = navigator.ar_area[2]
    else:  # mode B
        A = navigator.ar_area[1]
        B = navigator.ar_area[3]

    A = A.astype(np.float32)
    B = B.astype(np.float32)

    # assign forward/backward direction
    target_pt = B if navigator.diag_dir > 0 else A
    target_pt = target_pt.astype(np.float32)
    info['target_pt'] = tuple(target_pt.astype(int))

    # ---------------------------------------
    # Lane following: compute closest point on diagonal AB
    # ---------------------------------------
    AB = B - A
    AP = midpoint - A
    ab_norm_sq = np.dot(AB, AB)

    # projection t of midpoint onto AB
    t = np.dot(AP, AB) / (ab_norm_sq + 1e-9)
    t = np.clip(t, 0.0, 1.0)  # keep projection inside segment
    lane_point = A + t * AB
    info['lane_point'] = tuple(lane_point.astype(int))

    # cross-track error
    cte_vec = midpoint - lane_point
    cte_dist = np.linalg.norm(cte_vec)

    # direction to lane center
    if cte_dist > 1e-6:
        cte_dir = cte_vec / cte_dist
    else:
        cte_dir = np.zeros(2)

    # ---------------------------------------
    # Distance to diagonal endpoint (for speed control)
    # ---------------------------------------
    dist_to_target = np.linalg.norm(midpoint - target_pt)
    max_dist = np.linalg.norm(A - B)

    # thresholds for reversing direction
    stop_thresh = TARGET_RADIUS_PIX
    if dist_to_target < stop_thresh:
        navigator.diag_dir *= -1
        target_pt = A if navigator.diag_dir > 0 else B
        target_pt = target_pt.astype(np.float32)
        info['target_pt'] = tuple(target_pt.astype(int))

    # ---------------------------------------
    # Compute tangent line + angle
    # ---------------------------------------
    tangent_vec, closest_pt = estimate_local_tangent_and_closest(contour, midpoint)
    info['closest_point'] = closest_pt
    info['tangent_vec'] = tangent_vec

    tangent_angle = np.degrees(np.arctan2(-tangent_vec[1], tangent_vec[0])) if tangent_vec is not None else 0.0    

    # ---------------------------------------
    # Compute heading angle
    # ---------------------------------------
    heading_vec = np.array(front, dtype=np.float32) - np.array(rear, dtype=np.float32)
    if np.linalg.norm(heading_vec) < 1e-6:
        return 0.0, 0.0, info

    heading_dir = heading_vec / (np.linalg.norm(heading_vec) + 1e-9)
    heading_angle = np.degrees(np.arctan2(-heading_dir[1], heading_dir[0]))

    # ---------------------------------------
    # Combined steering target:
    #    - aim along diagonal lane direction
    #    - correct cross-track error
    # ---------------------------------------

    # diagonal main direction
    lane_dir = AB / (np.linalg.norm(AB) + 1e-9)
    lane_angle = np.degrees(np.arctan2(-lane_dir[1], lane_dir[0]))

    # direction to lane center
    if cte_dist > 1e-6:
        cte_angle = np.degrees(np.arctan2(-cte_dir[1], cte_dir[0]))
    else:
        cte_angle = lane_angle

    # blend: 70% diagonal direction, 30% lane centering
    desired_angle = 0.7 * lane_angle + 0.3 * cte_angle

    # also blend environmental tangent to smooth turns
    blend_angle = 0.8 * desired_angle + 0.2 * tangent_angle

    # steering error
    steer_err = angle_diff_signed_deg(blend_angle, heading_angle)
    steer_norm = np.clip(steer_err * STEER_K_DEG_TO_NORM, -1.0, 1.0)
    servo_cmd = -steer_norm * MAX_SERVO

    # ---------------------------------------
    # Straight-ahead servo dead-band logic
    # ---------------------------------------
    # If the desired steering angle is small -> push servo toward 0
    if abs(steer_err) < 6:     # 6° dead-zone
        servo_cmd *= 0.2       # soften → servo near 0
    if abs(steer_err) < 3:
        servo_cmd = 0.0        # perfect straightening

    # store before obstacle avoidance
    base_servo_cmd = servo_cmd

    # ---------------------------------------
    # Speed: slow near diagonal endpoints
    # ---------------------------------------
    d_norm = np.clip(dist_to_target / max_dist, 0.0, 1.0)
    speed_cmd = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * d_norm

    # ---------------------------------------
    # Being in outside area -> steer inward + penalize speed
    # ---------------------------------------
    inside = point_inside_polygon(navigator.area_contour, midpoint)
    info['inside'] = inside

    if not inside and navigator.area_contour is not None:
        speed_cmd *= INSIDE_PENALTY
        pts = navigator.area_contour.reshape(-1,2)
        centroid = np.mean(pts, axis=0)

        vec_to_c = centroid - midpoint
        if np.linalg.norm(vec_to_c) > 1e-6:
            angle_c = np.degrees(np.arctan2(-vec_to_c[1], vec_to_c[0]))
            steer_err2 = angle_diff_signed_deg(angle_c, heading_angle)
            steer_norm2 = -np.clip(steer_err2 * STEER_K_DEG_TO_NORM, -1.0, 1.0)
            servo_cmd = steer_norm2 * MAX_SERVO

    # ---------------------------------------
    # Obstacle avoidance (5x5 markers)
    # ---------------------------------------
    steer_offset_deg, obs_speed_mul = compute_obstacle_avoidance(
        midpoint,
        navigator.detected_obstacles if hasattr(navigator, "detected_obstacles") else [],
        A, B
    )

    # Adjust servo: offset in degrees -> normalized
    offset_norm = steer_offset_deg * STEER_K_DEG_TO_NORM
    servo_cmd = base_servo_cmd + offset_norm * MAX_SERVO
    speed_cmd *= obs_speed_mul        

    # ---------------------------------------
    # Smoothing & clamp
    # ---------------------------------------
    servo_sm = SMOOTH_ALPHA * navigator.last_servo + (1.0 - SMOOTH_ALPHA) * servo_cmd
    speed_sm = SMOOTH_ALPHA * navigator.last_speed + (1.0 - SMOOTH_ALPHA) * speed_cmd
    navigator.last_servo = servo_sm
    navigator.last_speed = speed_sm

    servo_out, speed_out = clamp_servo_and_speed(servo_sm, speed_sm)
    return servo_out, speed_out, info

def draw_geometrical_lines_and_points(frame_copy, real_frame, navigator, midpoint, info):
    """
    Draws all the necessary geometric elements onto the frame:
      - the polygon area (light cyan)
      - the active diagonal line (blue)
      - lane projection point (light orange)
      - connection from midpoint -> lane_point (yellow)
      - target diagonal endpoint (red)
    """
    if navigator.ar_area is None:
        return frame_copy

    # Determine diagonal endpoints
    if navigator.diagonal_mode == "A":
        A = navigator.ar_area[0]
        B = navigator.ar_area[2]
    else:
        A = navigator.ar_area[1]
        B = navigator.ar_area[3]

    A = A.astype(int)
    B = B.astype(int)

    # draw polygon area if available
    if navigator.area_contour is not None:     
        cv2.fillPoly(frame_copy, pts =[navigator.area_contour], color=(255,215,0))
        alpha = 0.4  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        cv2.addWeighted(frame_copy, alpha, real_frame, 1 - alpha, 0, frame_copy)

    # Draw diagonal line
    cv2.line(frame_copy, tuple(A), tuple(B), (255, 0, 0), 2)  # BLUE diagonal

    # draw closest contour point and line to midpoint
    if 'closest_point' in info and info['closest_point'] is not None and midpoint is not None:
        cp = info['closest_point']
        cv2.circle(frame_copy, tuple(cp), 6, (0,255,255), -1)
        cv2.line(frame_copy, tuple(np.array(cp).astype(int)), tuple(midpoint.astype(int)), (0,255,255), 2)

    # draw car markers & midpoint labels (already drawn inside detect_car_markers_for_id)
    if midpoint is not None:
        cv2.putText(frame_copy, f"M", tuple(midpoint.astype(int) + np.array([6,-6])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # Lane projection point
    if info.get("lane_point") is not None:
        lp = tuple(info["lane_point"])
        cv2.circle(frame_copy, lp, 6, (122, 208, 245), -1)  # light orange lane projection

        # Draw cross-track line from midpoint to lane point
        if midpoint is not None:
            cv2.line(frame_copy, tuple(midpoint.astype(int)), lp, (0, 255, 255), 2)  # YELLOW line

    # Draw target endpoint
    if info.get("target_pt") is not None:
        tp = tuple(info["target_pt"])
        cv2.circle(frame_copy, tp, 8, (255, 0, 255), -1) # magenta point

    # Visual debug for obstacle avoidance (closest obstacle)
    if midpoint is not None and hasattr(navigator, "detected_obstacles"):
        if len(navigator.detected_obstacles) > 0:
            
            # Choose correct diagonal endpoints based on navigator.diagonal_mode
            if navigator.diagonal_mode == "A":
                A_ = navigator.ar_area[0]
                B_ = navigator.ar_area[2]
            else:
                A_ = navigator.ar_area[1]
                B_ = navigator.ar_area[3]

            lane_dir = (B_ - A_).astype(np.float32)

            # Find closest obstacle
            dists = [np.linalg.norm(o["center"] - midpoint) for o in navigator.detected_obstacles]
            closest = navigator.detected_obstacles[int(np.argmin(dists))]
            obstacle_center = closest["center"]

            # Recalculate avoidance offset for visualization
            steer_offset_deg, _ = compute_obstacle_avoidance(
                midpoint,
                navigator.detected_obstacles,
                A_.astype(np.float32),
                B_.astype(np.float32)
            )

            # Draw everything
            draw_obstacle_debug(
                frame_copy,
                midpoint,
                obstacle_center,
                lane_dir,
                steer_offset_deg
            )

def draw_obstacle_debug(frame, midpoint, obstacle, lane_dir, steer_offset_deg):
    """
    Draws:
        - midpoint
        - obstacle point
        - line midpoint->obstacle
        - lateral left/right axis
        - steering correction vector (derived from offset)
    """

    if midpoint is None or obstacle is None:
        return frame

    mp = midpoint.astype(int)
    ob = obstacle.astype(int)

    # 1. Color-coded danger line midpoint → obstacle
    dist = np.linalg.norm(obstacle - midpoint)
    if dist > 180:
        col = (0, 255, 0)      # far (green)
    elif dist > 90:
        col = (0, 200, 255)    # medium (yellow)
    else:
        col = (0, 0, 255)      # critical (red)

    cv2.line(frame, mp, ob, col, 2)
    cv2.circle(frame, ob, 6, col, -1)

    # 2. Draw midpoint
    cv2.circle(frame, mp, 6, (255, 255, 255), -1)

    # 3. Draw lateral axis: left (blue), right (magenta)
    lane_dir = lane_dir / (np.linalg.norm(lane_dir) + 1e-9)
    left_vec = np.array([-lane_dir[1], lane_dir[0]])
    right_vec = -left_vec

    L1 = (midpoint + left_vec * 60).astype(int)
    R1 = (midpoint + right_vec * 60).astype(int)

    cv2.line(frame, mp, L1, (255, 0, 0), 2)     # left = blue
    cv2.line(frame, mp, R1, (255, 0, 255), 2)   # right = magenta

    # 4. Avoidance steering vector (yellow arrow)
    #    Direction based on steer_offset_deg sign
    magnitude = np.interp(abs(steer_offset_deg), [0, 35], [0, 70])
    if steer_offset_deg < 0:
        # steer right
        avoid_dir = right_vec
    else:
        # steer left
        avoid_dir = left_vec

    A1 = (midpoint + avoid_dir * magnitude).astype(int)
    cv2.arrowedLine(frame, mp, A1, (0, 255, 255), 3, tipLength=0.3)

    # 5. Display numeric info
    cv2.putText(frame,
                f"avoid {steer_offset_deg:+.1f} deg",
                (mp[0] + 10, mp[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2)


def init_camera(cam_index=0):
    if os.name == "nt":
        cap = cv2.VideoCapture(cam_index)
    elif os.name == "posix":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        sys.exit("Camera not found!\n")
        
    # Optional: set your preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Set the preferred FPS of the camera, if you have multiple options
    # cap.set(cv2.CAP_PROP_FPS, 60) 

    print("Camera opened successfully!\n")
    return cap


# -------------------------
# Public run() to be used by threads
# -------------------------
def run_a(capture, car_id):
    """
    Process a single frame and return (servo_angle, motor_speed, annotated_frame).
    - capture: cv2.VideoCapture object
    - car_id: integer marker ID used by this car (both front & rear share this ID)
    """
    global _navigators

    # ensure navigator exists
    if car_id not in _navigators:
        _navigators[car_id] = AutonomousNavigator(car_id)
    navigator = _navigators[car_id]

    # read frame under lock
    with CV_LOCK:
        ret, frame = capture.read()
    if not ret or frame is None:
        print(f"[Car {car_id}] Frame not captured.")
        return 0.0, 0.0, navigator.last_frame

    navigator.frame_shape = frame.shape
    navigator.set_diagonal_mode("A")

    # detect area markers and possibly set area if enough corners
    area_centers = detect_area_markers(frame)
    if len(area_centers) >= 4:
        navigator.set_area_from_markers_spatial(area_centers)

    # detect front/rear for this car_id
    front, rear, midpoint = detect_car_markers_for_id(frame, car_id)

    # detect obstacles
    obstacles = detect_obstacles(frame)
    navigator.detected_obstacles = obstacles

    servo = 0.0
    speed = 0.0
    info = {}

    if navigator.area_contour is not None and midpoint is not None and front is not None and rear is not None:
        # servo, speed, info = compute_control_simple(navigator.area_contour, front, rear, midpoint, navigator)
        servo, speed, info = compute_control_area(navigator.area_contour, front, rear, midpoint, navigator)
    else:
        # not enough info -> stop and set visualization only
        servo, speed = 0.0, 0.0
        info = {}

    # --- Visualization overlay on frame (returned to main thread) ---
    vis = frame.copy()
    draw_geometrical_lines_and_points(vis, frame, navigator, midpoint, info)
    
    # telemetry
    cv2.putText(vis, f"Car {car_id} spd={speed:.2f} ang={servo:.2f}",
                (10, vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    
    # Just for debug - comment when not needed
    # \

    navigator.last_frame = vis.copy()
    return round(servo,2), round(speed,2), vis


# Just for debug - comment when not needed
# if __name__ == '__main__':
#     capture = init_camera(1)
#     car_idx = 0
#     while 1:
#         run_a(capture, car_idx)
#     capture.release()
#     cv2.destroyAllWindows()