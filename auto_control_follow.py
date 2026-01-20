# leader_follower_refactor.py
# Extended version of your script implementing a 3-car leader-follower system.
# Keep your existing imports and helper functions (aruco detection, geometry helpers).
# This file expects the helper functions you already provided are available:
#   detect_any_marker, detect_area_markers, detect_obstacles,
#   estimate_local_tangent_and_closest, angle_diff_signed_deg, clamp_servo_and_speed, etc.

import numpy as np
import cv2
import sys
import os
from cv2 import aruco
from threading import Lock

# -------------------------
# CONFIG / TUNING CONSTANTS
# -------------------------
AREA_DICT = aruco.DICT_6X6_250    # dictionary used for area corner markers
CAR_DICT = aruco.DICT_4X4_50      # dictionary used for car markers (front/back)
OBJ_DICT = aruco.DICT_5X5_100     # dictionary used for object detection

MIN_SPEED = 0.45
MAX_SPEED = 0.55
MAX_SERVO = 0.5

TARGET_RADIUS_PIX = 120
SMOOTH_ALPHA = 0.5
INSIDE_PENALTY = 0.5
STEER_K_DEG_TO_NORM = 1.0 / 90.0

CV_LOCK = Lock()


# --- Car class representing each vehicle ---
class Car:
    def __init__(self, car_id, mark_id, is_leader=False, desired_gap_px=150):
        self.car_id = car_id
        self.mark_id = mark_id
        self.is_leader = is_leader
        # pose
        self.front = None
        self.rear = None
        self.midpoint = None
        self.heading_angle = 0.0  # degrees, same convention used in your code
        # control state
        self.last_servo = 0.0
        self.last_speed = 0.0
        self.last_frame = None
        # follower specifics
        self.leader_id = None
        self.desired_gap_px = desired_gap_px
        # diagnostics / debug
        self.detected = False

    def update_pose_from_marker_centers(self, centers):
        """
        centers: list of 2D center points (list length >=2)
        sets self.front, self.rear, self.midpoint, self.heading_angle
        Uses same policy you used: smaller y -> toward top -> front.
        """
        if centers is None or len(centers) < 2:
            self.front = None
            self.rear = None
            self.midpoint = None
            self.detected = False
            return False

        p1 = np.array(centers[0], dtype=np.float32)
        p2 = np.array(centers[1], dtype=np.float32)
        if p1[1] < p2[1]:
            front, rear = p1, p2
        else:
            front, rear = p2, p1
        self.front = front
        self.rear = rear
        self.midpoint = 0.5 * (front + rear)
        heading_vec = front - rear
        if np.linalg.norm(heading_vec) < 1e-6:
            self.heading_angle = 0.0
        else:
            heading_dir = heading_vec / (np.linalg.norm(heading_vec) + 1e-9)
            self.heading_angle = float(np.degrees(np.arctan2(-heading_dir[1], heading_dir[0])))
        self.detected = True
        return True


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
def angle_diff_signed_deg(a_deg, b_deg):
    """Smallest signed difference a - b in degrees mapped to [-180,180]."""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return d


def angle_wrap(a):
    """normalize angles to continuous comparison"""
    return (a + np.pi) % (2 * np.pi) - np.pi


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


def clamp_servo_and_speed(servo, speed):
    """Clamp to allowed ranges."""
    s = float(np.clip(servo, -MAX_SERVO, MAX_SERVO))
    v = float(np.clip(speed, MIN_SPEED, MAX_SPEED))
    return s, v


# --- Fleet manager orchestrates detection + control ---
class FleetManager:
    def __init__(self):
        self.cars = []
        self.circle_center = None
        self.circle_radius = None
        # obstacle list (list of dicts with 'id' and 'center')
        self.obstacles = []

    def rebuild_car_list(self, ids):
        """
        Rebuild cars in order of detected markers.
        car_id = index in detection order
        mark_id = actual aruco id
        Leader = car 0
        Followers follow the previous car
        """
        self.cars = []
        id_list = ids.flatten().tolist()

        for i, mark in enumerate(id_list):
            is_leader = (i == 0)
            car = Car(car_id=i, mark_id=mark, is_leader=is_leader)
            if not is_leader:
                car.leader_id = i - 1
            self.cars.append(car)

    # --------------------
    # Virtual circle generation
    # --------------------
    def compute_virtual_circle_from_area(self, area_centers):
        """
        area_centers: dict real_id -> (x,y) from detect_area_markers
        Strategy: centroid of available vertices as center and mean radius to vertices.
        """
        if len(area_centers) < 3:
            return False
        pts = np.array(list(area_centers.values()), dtype=np.float32)
        centroid = pts.mean(axis=0)
        radii = np.linalg.norm(pts - centroid[None, :], axis=1)
        radius = float(np.mean(radii))
        self.circle_center = np.array(centroid, dtype=np.float32)
        self.circle_radius = float(radius)
        return True

    # --------------------
    # Angle helpers
    # --------------------
    def angle_of_point(self, p):
        """
        Return angle theta in radians for point p relative to circle_center (atan2)
        """
        if self.circle_center is None:
            return None
        dx = p[0] - self.circle_center[0]
        dy = p[1] - self.circle_center[1]
        return float(np.arctan2(dy, dx))  # radians

    def point_on_circle(self, theta):
        return self.circle_center + self.circle_radius * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

    def tangent_angle_at_theta_deg(self, theta):
        """
        For CCW traversal tangent angle (deg) is theta + 90deg.
        Convert to degrees and maintain the same sign convention as heading_angle in Car.
        """
        # Convert to degrees; also note heading uses -y in atan2 earlier: we must keep consistent
        # earlier heading used: degrees(arctan2(-y, x)). We'll compute tangent vector and convert same way.
        tx = -np.sin(theta)  # derivative of cos= -sin -> x component of tangent
        ty = np.cos(theta)   # derivative of sin = cos -> y component
        # desired heading in same convention: heading = degrees(atan2(-ty, tx))
        return float(np.degrees(np.arctan2(-ty, tx)))

    # --------------------
    # Main per-frame processing
    # --------------------
    def update_from_frame(self, frame):
        """
        1) detect area markers -> update virtual circle
        2) detect car markers -> update each Car.pose
        3) detect obstacles -> update self.obstacles
        4) compute commands for each car
        5) render a combined annotated frame and return commands
        """
        annotated = frame.copy()
        # 1) area
        area_centers = detect_area_markers(annotated)
        if len(area_centers) >= 3:
            self.compute_virtual_circle_from_area(area_centers)

        # draw circle if available
        if self.circle_center is not None:
            c = tuple(self.circle_center.astype(int))
            r = int(self.circle_radius)
            cv2.circle(annotated, c, r, (200, 200, 0), 2)
            cv2.circle(annotated, c, 4, (255, 0, 255), -1)

        # 2) detect car markers
        corners, ids = detect_any_marker(annotated, CAR_DICT)

        marker_groups = {}

        if ids is not None:
            # If number of detected markers differs from car list: rebuild
            if len(self.cars) != len(ids):
                self.rebuild_car_list(ids)

            # build marker_groups: mark_id -> list of centers
            centers = [np.mean(c[0], axis=0) for c in corners]
            for mid, center in zip(ids.flatten(), centers):
                marker_groups.setdefault(int(mid), []).append(center)

            # update pose for each car by mark_id
            for car in self.cars:
                if car.mark_id in marker_groups and len(marker_groups[car.mark_id]) >= 2:
                    car.update_pose_from_marker_centers(marker_groups[car.mark_id][:2])
                else:
                    car.update_pose_from_marker_centers(None)
        else:
            # No markers at all — clear all detection
            for car in self.cars:
                car.update_pose_from_marker_centers(None)

        # 3) detect obstacles (5x5)
        self.obstacles = detect_obstacles(annotated)

        # 4) compute controls
        commands = {}  # car_id -> (servo, speed)
        if len(self.cars) > 0:
            leader = self.cars[0]
            servo_l, speed_l = self.compute_leader_control(leader)
            commands[0] = (servo_l, speed_l)

        for i in range(1, len(self.cars)):
            car = self.cars[i]
            servo_f, speed_f = self.compute_follower_control(car)
            commands[i] = (servo_f, speed_f)

        # 5) visualization: draw each car pose & commands
        for car in self.cars:
            if car.midpoint is not None:
                mp = tuple(car.midpoint.astype(int))
                cv2.circle(annotated, mp, 5, (255, 255, 255), -1)
                cv2.putText(annotated, f"Car{car.car_id}", (mp[0] + 6, mp[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                # draw heading vector
                hd = np.array([np.cos(np.radians(-car.heading_angle)), np.sin(np.radians(-car.heading_angle))])  # convert back
                p2 = (car.midpoint + hd * 30).astype(int)
                cv2.line(annotated, mp, tuple(p2), (200, 200, 200), 2)
            # draw intended commands
            if car.car_id in commands:
                s_cmd, v_cmd = commands[car.car_id]
                cv2.putText(annotated, f"id{car.car_id} s={s_cmd:.2f} v={v_cmd:.2f}", (10, 20 + 20 * car.car_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # draw obstacles
        for o in self.obstacles:
            oc = tuple(o["center"].astype(int))
            cv2.circle(annotated, oc, 6, (0, 0, 255), -1)
            cv2.putText(annotated, f"obj{o['id']}", (oc[0] + 6, oc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

        return commands, annotated

    # --------------------
    # Leader control function
    # --------------------
    def compute_leader_control(self, leader: Car):
        """
        Leader follows the circle. Uses obstacle avoidance & tangent heading.
        Returns (servo, speed)
        """
        if leader.midpoint is None or self.circle_center is None:
            # cannot compute -> stop
            return 0.0, 0.0

        # project to circle
        theta_l = self.angle_of_point(leader.midpoint)
        # lookahead distance (px) proportional to last speed
        L_look = 80.0 + 120.0 * (leader.last_speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED + 1e-9)
        dtheta = L_look / (self.circle_radius + 1e-9)
        theta_target = theta_l + dtheta  # CCW direction

        # target_pt = self.point_on_circle(theta_target)
        desired_heading_deg = self.tangent_angle_at_theta_deg(theta_target)

        # heading error & servo
        steer_err = angle_diff_signed_deg(desired_heading_deg, leader.heading_angle)
        steer_norm = np.clip(steer_err * STEER_K_DEG_TO_NORM, -1.0, 1.0)
        servo_cmd = -steer_norm * MAX_SERVO
        # small dead-zone
        if abs(steer_err) < 6:
            servo_cmd *= 0.2
        if abs(steer_err) < 3:
            servo_cmd = 0.0

        # speed base: faster when far from target along arc
        arc_dist_to_target = np.abs(dtheta) * self.circle_radius
        arc_frac = np.clip(arc_dist_to_target / (np.pi * self.circle_radius), 0.0, 1.0)
        speed_cmd = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (0.7 + 0.3 * arc_frac)

        # inside polygon check -> penalize
        # (use existing nav area contour if you have it; here we just use area of circle)
        # obstacle avoidance
        steer_offset_deg, obs_speed_mul = compute_obstacle_avoidance(leader.midpoint, self.obstacles, self.circle_center, self.point_on_circle(theta_l + np.pi))  # B dummy
        offset_norm = steer_offset_deg * STEER_K_DEG_TO_NORM
        servo_cmd = servo_cmd + offset_norm * MAX_SERVO
        speed_cmd *= obs_speed_mul

        # smoothing & clamp
        servo_sm = SMOOTH_ALPHA * leader.last_servo + (1.0 - SMOOTH_ALPHA) * servo_cmd
        speed_sm = SMOOTH_ALPHA * leader.last_speed + (1.0 - SMOOTH_ALPHA) * speed_cmd
        leader.last_servo = servo_sm
        leader.last_speed = speed_sm
        servo_out, speed_out = clamp_servo_and_speed(servo_sm, speed_sm)

        return round(servo_out, 2), round(speed_out, 2)

    # --------------------
    # Follower control function
    # --------------------
    def compute_follower_control(self, follower: Car):
        """
        Each follower tries to keep an arc offset behind its leader on the circle.
        Uses simple pure-pursuit to a target point on the circle, with speed limiting and no-overtake rule.
        """
        if follower.midpoint is None or follower.leader_id is None:
            return 0.0, 0.0

        # SAFELY retrieve leader by index (self.cars is a list)
        leader = None
        if isinstance(follower.leader_id, int) and 0 <= follower.leader_id < len(self.cars):
            leader = self.cars[follower.leader_id]

        if leader is None or leader.midpoint is None:
            # no leader pose -> stop or be conservative
            return 0.0, 0.0

        # get leader and follower angles
        theta_L = self.angle_of_point(leader.midpoint)
        theta_F = self.angle_of_point(follower.midpoint)

        if theta_L is None or theta_F is None:
            return 0.0, 0.0

        # desired arc gap (px). Could be per-car configuration
        desired_gap = follower.desired_gap_px
        # convert arc gap to angle delta
        dtheta_gap = desired_gap / (self.circle_radius + 1e-9)
        theta_target = theta_L - dtheta_gap  # behind leader by gap (CCW ordering)

        # compute arc distance leader
        delta = angle_wrap(theta_L - theta_F)  # positive if follower is behind (CW sign depends on conventions)
        arc_dist = delta * self.circle_radius

        # target point on circle for follower to aim at
        target_pt = self.point_on_circle(theta_target)

        # compute desired heading to target_pt
        vec_to_target = target_pt - follower.midpoint
        if np.linalg.norm(vec_to_target) < 1e-6:
            desired_heading_deg = follower.heading_angle
        else:
            desired_heading_deg = float(np.degrees(np.arctan2(-vec_to_target[1], vec_to_target[0])))

        # steering
        steer_err = angle_diff_signed_deg(desired_heading_deg, follower.heading_angle)
        steer_norm = np.clip(steer_err * STEER_K_DEG_TO_NORM * 1.1, -1.0, 1.0)
        servo_cmd = -steer_norm * MAX_SERVO
        if abs(steer_err) < 4:
            servo_cmd *= 0.3
        if abs(steer_err) < 2:
            servo_cmd = 0.0

        # speed: cannot exceed leader's speed (no-overtake)
        v_leader = leader.last_speed if leader.last_speed is not None else MAX_SPEED
        # proportional to arc distance error
        gap_error = arc_dist - desired_gap
        # if follower too close -> slow
        if arc_dist < follower.desired_gap_px * 0.6:
            speed_cmd = 0.0
        else:
            speed_cmd = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * np.clip((gap_error / (follower.desired_gap_px + 1e-9)), 0.0, 1.0)
            # cap to leader speed
            speed_cmd = min(speed_cmd, v_leader)

        # extra safeguard: predicted overtaking test:
        # predict follower next arc if it drove at speed_cmd for dt and compare arc index (approx).
        # Here we just enforce a hard rule: never allow speed_cmd > v_leader.
        speed_cmd = min(speed_cmd, v_leader)

        # smoothing & apply last
        servo_sm = SMOOTH_ALPHA * follower.last_servo + (1.0 - SMOOTH_ALPHA) * servo_cmd
        speed_sm = SMOOTH_ALPHA * follower.last_speed + (1.0 - SMOOTH_ALPHA) * speed_cmd
        follower.last_servo = servo_sm
        follower.last_speed = speed_sm
        servo_out, speed_out = clamp_servo_and_speed(servo_sm, speed_sm)

        return round(servo_out, 2), round(speed_out, 2)

    def run(self, frame, car_id=0):
        """
        Public API.
        Returns strictly (servo, speed, annotated_frame), never None for numeric or frame.
        """
        with CV_LOCK:
            commands, vis = self.update_from_frame(frame)

        if car_id in commands:
            servo, speed = commands[car_id]
        else:
            servo, speed = 0.0, 0.0

        return round(float(servo), 2), round(float(speed), 2), vis


# Instantiate a single FleetManager to preserve state across calls
fm = FleetManager()

# -------------------------------
# HOW TO USE
# -------------------------------
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
#TODO fix the frame showing
# adapt and fine tune the parameters
def run_f(capture, car_id):
    with CV_LOCK:
        ret, frame = capture.read()

    if not ret or frame is None:
        # Always return a valid frame (black fail-safe with same resolution as init defaults)
        safe_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        return 0.0, 0.0, safe_frame

    servo, speed, vis = fm.run(frame, car_id)

    # # Just for debug - comment when not needed
    # cv2.imshow("Follow", vis)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     sys.exit("\nQuitting the frame!")

    return servo, speed, vis


# # Just for debug - comment when not needed
# if __name__ == '__main__':
#     capture = init_camera(1)
#     car_idx1 = 0
#     # car_idx2 = 1
#     # car_idx3 = 2
#     while 1:
#         run_f(capture, car_idx1)
#         # run_f(capture, car_idx2)
#         # run_f(capture, car_idx3)
#     capture.release()
#     cv2.destroyAllWindows()
