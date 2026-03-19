import cv2
import cv2.aruco as aruco
import numpy as np
import time
# import sys
# import os

#TODO better version, still needs some fine-tuning, but at least is turning on sharp 
# edges/curves and have the motor speed slowing down
# going straight for linear curves and almost MAX_SPEED

#  Constants 
ANGLE_THRESHOLD        = 1       # Min angle change to issue a new command
LOW_THRESHOLD          = 40      # Distance considered "too close" (px)
HIGH_THRESHOLD         = 80      # Distance considered "far" (px)
WEIGHT                 = 0.5     # Exponent for scoring non-linearity
ANGLE_FAVOR            = 0.7     # Scoring bias toward direction vs. distance

SPEED_THRESHOLD        = 15
STATUS_UPDATE_INTERVAL = 0.1
SCALING_FACTOR         = 10.3

N_SAMPLES = 40       # Left-boundary sample count
MIN_SPEED = 0.45
MAX_SPEED = 0.60
# Servo neutral is now 0.0; range is [-0.5, 0.5] → maps to [-45°, +45°] physical rotation
MAX_SERVO       = 0.5

# Threshold scaled proportionally: 1° / 45° ≈ 0.022 in normalized units
ANGLE_THRESHOLD = 0.02

# Jump-filter upper bound scaled from 70°/90° into the new [-0.5, 0.5] space
JUMP_FILTER     = 0.78

# Lane marker IDs (6x6 ArUco)
RIGHT_END_IDS = [196, 197]
LEFT_END_IDS  = [198, 199]
RIGHT_MID_ID  = 200
LEFT_MID_ID   = 201

#  Global state 
tracker:          dict = {}
last_sent_angles: dict = {}

#  Geometric helpers 
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

#  Bézier sampling 
def sample_bezier(p0: np.ndarray, p1: np.ndarray,
                  p2: np.ndarray = None, n: int = N_SAMPLES) -> list:
    """
    Sample n points on a linear (p2=None) or quadratic Bézier curve.
      Linear:    B(t) = (1-t)·p0 + t·p1
      Quadratic: B(t) = (1-t)²·p0 + 2t(1-t)·p1 + t²·p2
    Returns a list of (x, y) integer tuples.
    """
    pts = []
    for i in range(n):
        t = i / max(n - 1, 1)
        pt = ((1.0-t)*p0 + t*p1) if p2 is None else \
             ((1.0-t)**2*p0 + 2*t*(1.0-t)*p1 + t**2*p2)
        pts.append(tuple(pt.astype(int)))
    return pts

#  Marker detection 
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
        return {int(i): corners[k] for k, i in enumerate(ids.flatten())}

    return _detect(aruco.DICT_4X4_50), _detect(aruco.DICT_6X6_250)

#  Car geometry 
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

#  Lane detection (Bézier-based) 
def detect_lane_curves(lane_markers: dict) -> dict:
    """
    Build Bézier boundary curves from 6x6 lane markers.
    Right: [196]→start, [200]→optional apex, [197]→end
    Left:  [198]→start, [201]→optional apex, [199]→end
    """
    lane = {'right_mode': 'none', 'left_mode': 'none'}
    if all(i in lane_markers for i in RIGHT_END_IDS):
        p0 = np.array(marker_center(lane_markers[196]), dtype=float)
        p2 = np.array(marker_center(lane_markers[197]), dtype=float)
        if RIGHT_MID_ID in lane_markers:
            p1 = np.array(marker_center(lane_markers[RIGHT_MID_ID]), dtype=float)
            lane['right'], lane['right_mode'] = sample_bezier(p0, p1, p2), 'quadratic'
        else:
            lane['right'], lane['right_mode'] = sample_bezier(p0, p2), 'linear'
    if all(i in lane_markers for i in LEFT_END_IDS):
        p0 = np.array(marker_center(lane_markers[198]), dtype=float)
        p2 = np.array(marker_center(lane_markers[199]), dtype=float)
        if LEFT_MID_ID in lane_markers:
            p1 = np.array(marker_center(lane_markers[LEFT_MID_ID]), dtype=float)
            lane['left'], lane['left_mode'] = sample_bezier(p0, p1, p2), 'quadratic'
        else:
            lane['left'], lane['left_mode'] = sample_bezier(p0, p2), 'linear'
    return lane

#  Heading conversion 
def heading_to_angle(heading_vec: tuple) -> float:
    """
    Convert (dx, dy) vector to a compass angle in degrees [0, 360).
      0°→right, 90°→up, 180°→left, 270°→down  (image-space corrected).
    """
    return np.degrees(np.arctan2(-heading_vec[1], heading_vec[0])) % 360

#  Scoring logic
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
    # Positive relative_angle → boundary is left of heading → steer left (negative)
    servo = -weight * MAX_SERVO if relative_angle > 0 else weight * MAX_SERVO
    return float(np.clip(servo, -MAX_SERVO, MAX_SERVO))

#  Best boundary-point selection 
def select_best_boundary_point(car_pos: tuple, car_heading_angle: float,
                                boundary_points: list):
    """
    Replaces the contour-point search from Implementation 1.
    Iterates over sampled left-boundary points, computes each point's
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
            relative_angle -= 360   # Normalize to [-180, 180]

        if dist < dynamic_threshold(relative_angle):
            score = compute_point_score(relative_angle, dist)
            if score < best_score:
                best_point = pt
                best_angle = relative_angle
                best_dist  = dist
                best_score = score

    return best_point, best_angle, best_dist

# Drawing helpers
def draw_lane(frame: np.ndarray, lane: dict) -> np.ndarray:
    """
    Draw:
      1. Semi-transparent green fill between boundaries (when both present)
      2. Right boundary curve (cyan)
      3. Left  boundary curve (yellow)
    """
    for side in ['left', 'right']:
        if side in lane:
            for i in range(len(lane[side]) - 1):
                if side == 'right':
                    cv2.line(frame, lane[side][i], lane[side][i+1], (255, 255, 0), 2)
                else:
                    cv2.line(frame, lane[side][i], lane[side][i+1], (0, 255, 255), 2)
    
    if 'left' in lane and 'right' in lane:
        overlay   = frame.copy()
        right_pts = np.array(lane['right'], dtype=np.int32)
        left_pts  = np.array(lane['left'],  dtype=np.int32)
        poly      = np.concatenate([right_pts, left_pts[::-1]])
        cv2.fillPoly(overlay, [poly], (0, 200, 0))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    return frame

def draw_cars(frame: np.ndarray, cars: dict, lane: dict) -> np.ndarray:
    """Draw rear (red), front (green), heading line (white), midpoint (orange),
    and orthogonal projection from midpoint to left boundary (magenta)."""
    for rear_id, cd in cars.items():
        front, rear, mid = cd['front'], cd['rear'], cd['midpoint']

        cv2.circle(frame, rear, 8, (0, 0, 255), -1)
        cv2.putText(frame, f"R{rear_id}", (rear[0]+10, rear[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        if front is not None:
            cv2.circle(frame, front, 8, (0, 255, 0), -1)
            cv2.putText(frame, "F", (front[0]+10, front[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.line(frame, rear, front, (255, 255, 255), 2)

        cv2.circle(frame, mid, 6, (255, 165, 0), -1)

        #  Projection from midpoint onto left boundary 
        if 'left' in lane and len(lane['left']) > 1:
            # Find the nearest sampled boundary point first …
            left_pts = np.array(lane['left'], dtype=float)
            idx      = int(np.argmin(np.linalg.norm(left_pts - np.array(mid), axis=1)))

            # … then project perpendicularly onto the local segment around it
            seg_a = lane['left'][max(idx - 1, 0)]
            seg_b = lane['left'][min(idx + 1, len(lane['left']) - 1)]
            proj  = project_point_to_line(mid, seg_a, seg_b)

            cv2.circle(frame, proj, 5, (0, 120, 255), -1)          # ?? dot
            cv2.line(frame, mid, proj, (255, 0, 255), 1)            # magenta line

    return frame

def draw_telemetry(frame: np.ndarray, car_id: int, cars: dict,
                   servo: int, motor: float,
                   right_mode: str, left_mode: str) -> np.ndarray:
    lines = [
        f"Car: {car_id}",
        f"Right: {right_mode}   Left: {left_mode}",
        f"Servo: {servo:.2f}",
        f"Motor: {motor:.2f}",
        f"Detected IDs: {sorted(cars.keys())}",
    ]
    for idx, text in enumerate(lines):
        cv2.putText(frame, text, (10, 30 + idx*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame

# Main function
def run(frame: np.ndarray, car_id: int):
    """
    Process one camera frame and return (servo_angle, motor_speed, annotated_frame).

    Control pipeline
    
    1.  Detect 4x4 car markers + 6x6 lane markers.
    2.  Build per-car geometry (midpoint, heading vector).
    3.  Build Bézier left/right boundaries from lane markers.
    4.  Sample N_SAMPLES points along the left boundary.
    5.  For each sample: compute relative angle + distance from car midpoint.
    6.  Filter with dynamic_threshold (forward-only, angle-dependent range).
    7.  Score with compute_point_score (angle+distance weighted trade-off).
    8.  Map the best candidate to a servo command via map_angle_to_servo.
    9.  Throttle motor speed inversely proportional to steering intensity.
    10. Draw lane, cars, target point, and telemetry overlay.
    """
    car_frame    = frame.copy()
    current_time = time.time()

    # Steps 1–3
    car_markers, lane_markers = detect_all_markers(car_frame)
    cars = identify_cars(car_markers, car_id)
    update_tracker(cars, current_time)
    lane = detect_lane_curves(lane_markers)       

    servo = 0.0         # neutral steering --> straight ahead
    motor = 0.0         # no speed, if no lane --> still stopped

    if car_id in cars and 'left' in lane:
        car_mid           = cars[car_id]['midpoint']
        car_heading_angle = heading_to_angle(cars[car_id]['heading'])

        # Steps 4–7: find best left-boundary point
        best_point, best_angle, best_dist = select_best_boundary_point(
            car_mid, car_heading_angle, lane['left']
        )

        if best_point is not None:
            cv2.circle(car_frame, tuple(best_point), 7, (0, 0, 255), -1)
            cv2.line(car_frame, car_mid, tuple(best_point), (0, 0, 255), 2)

            computed = map_angle_to_servo(best_angle, best_dist)
            if computed is not None:
                last = last_sent_angles.get(car_id)
                if (last is None
                        or (abs(computed - last) >= ANGLE_THRESHOLD
                            and abs(computed - last) < JUMP_FILTER)):   # was < 70
                    servo = computed
                    last_sent_angles[car_id] = servo
                else:
                    servo = last if last is not None else 0.0
        else:
            servo = 0.0
            last_sent_angles[car_id] = 0.0

        # abs(servo) directly — no longer needs "− 90" offset
        turn_strength = abs(servo) / MAX_SERVO
        motor = float(np.clip(
            MAX_SPEED * (1.0 - 0.5 * turn_strength), MIN_SPEED, MAX_SPEED))

    # Step 10: visualisation
    car_frame = draw_lane(car_frame, lane)
    car_frame = draw_cars(car_frame, cars, lane)
    car_frame = draw_telemetry(car_frame, car_id, cars, servo, motor,
                               lane['right_mode'], lane['left_mode'])

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
#             motor, servo, vis = run(frame, car_idx)
#             cv2.imshow("Tracking", vis)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break

#     capture.release()
#     cv2.destroyAllWindows()
