"""
auto_control_calibrated.py
==========================
Centralised vision-based controller for the minicar testbed.

Camera handling
───────────────
Two classes handle all camera operations:

  RawCamera
    ─ No lens-distortion correction.
    ─ Frames are used as-is from cv2.VideoCapture.
    ─ Use this when no calibration data is available.

  CalibratedCamera
    ─ Undistorts every frame using a camera matrix and distortion
      coefficients obtained from a prior calibration run.
    ─ Calibration can be performed interactively before the control
      loop starts (chessboard OR ChArUco board supported).
    ─ Pass use_calibration=True at launch (--calibrate flag) to run
      the calibration wizard; afterwards the corrected frames flow
      into the same control pipeline unchanged.

Both classes expose the same interface:
    cam.read() → (ok: bool, frame: np.ndarray)
    cam.release()

Usage
─────
  # No calibration (default)
  python auto_control_calibrated.py -n 1

  # With calibration wizard
  python auto_control_calibrated.py -n 1 --calibrate

  # Load previously saved calibration file
  python auto_control_calibrated.py -n 1 --calib-file calib.npz

Architecture (sense → plan → act)
──────────────────────────────────
Sensing : Overhead USB camera + ArUco marker detection.
  Three marker dictionaries are used in parallel:
  • 4×4_50  – car identification (front + rear markers)
  • 5×5_50  – track boundary   (inner / middle / outer rings)
  • 6×6_250 – static obstacles
Planning : Pure-pursuit path tracking on a stadium-fitted (pill-shape) curve,
  combined with rule-based distance-threshold coordination.
Acting : Quantised servo (steering) and motor (speed) commands sent
  as UDP packets to each minicar's Raspberry Pi.
"""

import argparse
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
SPEED_THRESHOLD        = 15
STATUS_UPDATE_INTERVAL = 0.1
SCALING_FACTOR         = 57.5    # px/cm  (146 PPI → 146/2.54 ≈ 57.5)
COMMAND_INTERVAL       = 0.40

# ── ArUco dictionaries ────────────────────────────────────────────────────────
CAR_DICT        = aruco.DICT_4X4_50
FRONT_MARKER_ID = 49
TRACK_DICT      = aruco.DICT_5X5_50
OBSTACLE_DICT   = aruco.DICT_6X6_250

# ── Track marker groupings (by ArUco ID) ─────────────────────────────────────
INNER_SET  = [0, 1, 2, 3]
MIDDLE_SET = [4, 5, 6, 7]
OUTER_SET  = [8, 9, 10, 11]

STADIUM_SAMPLES = 270   # total sample points on the fitted stadium curve

# ── Obstacle geometry ─────────────────────────────────────────────────────────
OBSTACLE_RADIUS         = 45
OBSTACLE_TRACK_HALF_WIDTH = 41
OBSTACLE_LOOKAHEAD      = 81

# ── Speed levels ──────────────────────────────────────────────────────────────
MAX_SPEED    = 0.60
CRUISE_SPEED = 0.55
SLOW_SPEED   = 0.45
STOP_SPEED   = 0.00

# ── Servo limits (radians, sent as normalised floats) ─────────────────────────
MAX_SERVO       = 0.50
MAX_MID_SERVO   = 0.35
MED_SERVO       = 0.22
MED_SMALL_SERVO = 0.12
SMALL_SERVO     = 0.05

# ── Discrete output sets ──────────────────────────────────────────────────────
SERVO_STEPS = [
    -MAX_SERVO, -MAX_MID_SERVO, -MED_SERVO, -MED_SMALL_SERVO, -SMALL_SERVO,
    0.0,
    SMALL_SERVO, MED_SMALL_SERVO, MED_SERVO, MAX_MID_SERVO, MAX_SERVO,
]
MOTOR_STEPS = [STOP_SPEED, SLOW_SPEED, CRUISE_SPEED, MAX_SPEED]

# ── Pure-pursuit (bicycle model) parameters ───────────────────────────────────
WHEELBASE_PX     = 85.0
DELTA_MAX        = MAX_SERVO
LOOKAHEAD_DIST   = 110
MIN_LOOKAHEAD    = 70
K_CURVATURE_LD   = 300.0
K_BOUNDARY_PUSH  = 0.012

# ── Lane-change / coordination ────────────────────────────────────────────────
LANE_CHANGE_HOLD      = 2.5
D_SAFE                = 23
D_WARN                = 52
COOP_MIN_GAP          = 158
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

# ── Experiment metadata ───────────────────────────────────────────────────────
LOG_SCENARIO = "S1"
LOG_POLICY   = "cooperative"

# ── Curvature thresholds (empirically tuned: κ ≈ 0.4–0.5 on semicircles) ─────
CURVE_SHARP_THRESH = 0.30
CURVE_LIGHT_THRESH = 0.08

# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE
# ══════════════════════════════════════════════════════════════════════════════

tracker:             dict = {}
obstacle_positions:  list = []
lane_state:          dict = {}
last_command_time:   dict = {}
coop_slowdown_until: dict = {}
_pp_waiting:         dict = {}
_log_entries:        list = []
_cycle_counter:      int  = 0

# ══════════════════════════════════════════════════════════════════════════════
# CAMERA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class RawCamera:
    """
    Raw (uncalibrated) camera source.

    Wraps cv2.VideoCapture and forwards frames without any lens-distortion
    correction.  Use this when no calibration data is available or when
    the overhead camera has negligible lens distortion.

    Parameters
    ----------
    cam_index : int
        OpenCV camera index (default 0 = first USB camera).
    width, height, fps : int
        Requested capture resolution and frame rate.  The camera may
        silently ignore values it does not support.
    """

    MODE = "RAW (no calibration)"

    def __init__(self, cam_index: int = 0,
                 width: int = 1920, height: int = 1080, fps: int = 60):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
        self._cap = cv2.VideoCapture(cam_index, backend)
        if not self._cap.isOpened():
            sys.exit(f"[camera] Cannot open camera index {cam_index}.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)
        self._print_info()

    def _print_info(self):
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[camera] Mode   : {self.MODE}")
        print(f"[camera] Resolution: {w}×{h}  FPS: {f}")

    def read(self):
        """Return (ok, frame).  Frame is passed through unchanged."""
        return self._cap.read()

    def release(self):
        self._cap.release()


class CalibratedCamera(RawCamera):
    """
    Calibrated camera source with lens-distortion correction.

    Extends RawCamera by undistorting every frame before returning it.
    Calibration can be:

      a) Loaded from a previously saved .npz file  (--calib-file PATH)
      b) Run interactively in this session          (--calibrate flag)
         Supports both chessboard and ChArUco board patterns.

    The calibration wizard
    ──────────────────────
    1. Choose board type: 'chess' or 'charuco'.
    2. For chess: specify (cols-1, rows-1) inner corners and square size (mm).
       For charuco: specify cols, rows, square size (mm), marker size (mm).
    3. Point the camera at the board from several angles.
    4. Press SPACE to capture a frame, ESC when done (≥ 20 frames recommended).
    5. Calibration runs automatically and the result is saved to calib.npz.

    Parameters
    ----------
    cam_index : int
        OpenCV camera index.
    calib_file : str or None
        Path to a .npz file containing 'camera_matrix' and 'dist_coeffs'.
        If None the wizard is launched automatically.
    board_type : str
        'chess' or 'charuco' — selects the calibration pattern.
    chess_size : tuple (cols, rows)
        Number of inner corners for chessboard (cols-1 × rows-1).
    square_mm : float
        Physical size of one square in millimetres.
    marker_mm : float
        Physical size of ChArUco marker (only used for charuco board).
    width, height, fps : int
        Capture settings forwarded to RawCamera.
    """

    MODE = "CALIBRATED (lens-distortion corrected)"

    def __init__(self, cam_index: int = 0,
                 calib_file: str = None,
                 board_type: str = "chess",
                 chess_size: tuple = (8, 6),
                 square_mm: float = 38.0,
                 marker_mm: float = 18.0,
                 width: int = 1920, height: int = 1080, fps: int = 60):
        super().__init__(cam_index, width, height, fps)
        self._map1 = None
        self._map2 = None

        if calib_file and os.path.isfile(calib_file):
            self._load(calib_file)
        else:
            if calib_file:
                print(f"[calib] File '{calib_file}' not found — launching wizard.")
            mtx, dist = self._run_wizard(board_type, chess_size,
                                         square_mm, marker_mm)
            self._build_maps(mtx, dist)
            self._save("calib.npz", mtx, dist)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load(self, path: str):
        data = np.load(path)
        mtx  = data["camera_matrix"]
        dist = data["dist_coeffs"]
        self._build_maps(mtx, dist)
        print(f"[calib] Loaded calibration from '{path}'.")
        print(f"[calib] Camera matrix:\n{mtx}")
        print(f"[calib] Distortion coefficients: {dist.ravel()}")

    def _save(self, path: str, mtx, dist):
        np.savez(path, camera_matrix=mtx, dist_coeffs=dist)
        print(f"[calib] Calibration saved → '{path}'.")

    def _build_maps(self, mtx, dist):
        """Pre-compute undistortion maps for fast per-frame remapping."""
        ok, frame = self._cap.read()
        if not ok:
            sys.exit("[calib] Cannot read a frame to build undistortion maps.")
        h, w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            mtx, dist, None, new_mtx, (w, h), cv2.CV_16SC2)
        print(f"[calib] Undistortion maps built  ({w}×{h}).")

    def _run_wizard(self, board_type: str, chess_size: tuple,
                    square_mm: float, marker_mm: float):
        """
        Interactive calibration wizard.

        Captures frames from the live camera feed.
        SPACE → capture current frame
        ESC   → finish and compute calibration
        """
        print("\n" + "═" * 60)
        print("  CAMERA CALIBRATION WIZARD")
        print("═" * 60)
        print(f"  Board type : {board_type.upper()}")
        if board_type == "chess":
            print(f"  Inner corners: {chess_size[0]} × {chess_size[1]}")
            print(f"  Square size  : {square_mm} mm")
        else:
            print(f"  ChArUco grid : {chess_size[0]} × {chess_size[1]}")
            print(f"  Square size  : {square_mm} mm  |  Marker size: {marker_mm} mm")
        print()
        print("  Hold the board in front of the camera.")
        print("  SPACE = capture frame   |   ESC = done & calibrate")
        print("  Aim for ≥ 20 frames from varied angles.")
        print("═" * 60 + "\n")

        if board_type == "charuco":
            return self._wizard_charuco(chess_size, square_mm, marker_mm)
        return self._wizard_chess(chess_size, square_mm)

    def _wizard_chess(self, chess_size, square_mm):
        cols, rows = chess_size
        obj_p = np.zeros((cols * rows, 3), np.float32)
        obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

        obj_points, img_points = [], []
        captured = 0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        while True:
            ok, frame = self._cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, (cols, rows), corners, found)
                cv2.putText(display, "Board detected — SPACE to capture",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No board found",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, f"Captured: {captured}  (ESC = done)",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Calibration — Chessboard", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                break
            if key == 32 and found:  # SPACE
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(obj_p)
                img_points.append(corners2)
                captured += 1
                print(f"[calib] Chessboard frame {captured} captured.")

        cv2.destroyWindow("Calibration — Chessboard")
        if captured < 5:
            sys.exit("[calib] Not enough frames (need ≥ 5). Exiting.")

        h, w = gray.shape
        print(f"[calib] Computing calibration from {captured} frames …")
        ret, mtx, dist, _, _ = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None)
        print(f"[calib] RMS reprojection error: {ret:.4f} px")
        return mtx, dist

    def _wizard_charuco(self, grid_size, square_mm, marker_mm):
        cols, rows = grid_size
        # ArUco dictionary for ChArUco markers
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        board = aruco.CharucoBoard((cols, rows),
                                   square_mm / 1000.0,
                                   marker_mm / 1000.0,
                                   aruco_dict)
        detector_params = aruco.DetectorParameters()
        charuco_params  = aruco.CharucoParameters()
        charuco_detector = aruco.CharucoDetector(board, charuco_params,
                                                 detector_params)

        all_charuco_corners, all_charuco_ids = [], []
        captured = 0

        while True:
            ok, frame = self._cap.read()
            if not ok:
                continue
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()

            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
            if charuco_ids is not None and len(charuco_ids) >= 4:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners,
                                                     charuco_ids)
                cv2.putText(display, f"ChArUco detected ({len(charuco_ids)} corners) — SPACE",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No ChArUco board found",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display, f"Captured: {captured}  (ESC = done)",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Calibration — ChArUco", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32 and charuco_ids is not None and len(charuco_ids) >= 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                captured += 1
                print(f"[calib] ChArUco frame {captured} captured.")

        cv2.destroyWindow("Calibration — ChArUco")
        if captured < 5:
            sys.exit("[calib] Not enough frames (need ≥ 5). Exiting.")

        ok, frame = self._cap.read()
        h, w = frame.shape[:2]
        print(f"[calib] Computing calibration from {captured} frames …")
        ret, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, board, (w, h), None, None)
        print(f"[calib] RMS reprojection error: {ret:.4f} px")
        return mtx, dist

    # ── public interface ──────────────────────────────────────────────────────

    def read(self):
        """Return (ok, undistorted_frame)."""
        ok, frame = self._cap.read()
        if not ok or self._map1 is None:
            return ok, frame
        undistorted = cv2.remap(frame, self._map1, self._map2,
                                cv2.INTER_LINEAR)
        return True, undistorted


# ══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def _build_log_entry(t, k, car_id, policy, pose, lane, segment, curvature,
                     servo, motor, waiting, lateral_error, heading_error,
                     obstacle_info, cars, events) -> dict:
    ids = sorted(cars.keys())
    distances = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pa = np.array(cars[a]["midpoint"], dtype=float)
            pb = np.array(cars[b]["midpoint"], dtype=float)
            distances[f"{a}-{b}"] = round(float(np.linalg.norm(pa - pb)), 2)
    obs_d = obstacle_info["distance"]
    return {
        "t": round(t, 5), "k": k, "car_id": car_id, "policy": policy,
        "pose": [round(pose[0], 2), round(pose[1], 2), round(pose[2], 2)],
        "lane": lane, "segment": segment, "curvature": curvature,
        "command": {"servo": round(servo, 3), "motor": round(motor, 3)},
        "waiting": waiting,
        "lateral_error": round(lateral_error, 2),
        "heading_error":  round(heading_error, 2),
        "obstacle": {
            "state": obstacle_info["state"],
            "distance_px": round(obs_d, 2) if obs_d < float("inf") else None,
        },
        "distances": distances,
        "events": events,
    }


def save_log(path: str = "experiment_log.json") -> None:
    payload = {
        "meta": {
            "scenario": LOG_SCENARIO, "policy": LOG_POLICY,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_frames": len(_log_entries),
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
    return tuple(np.mean(corner[0], axis=0).astype(int))

def midpoint(p1: tuple, p2: tuple) -> tuple:
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def wrap_angle_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def heading_to_angle(heading_vec: tuple) -> float:
    return float(np.degrees(np.arctan2(-heading_vec[1], heading_vec[0])) % 360)

def dominant_state(state_dict: dict) -> str:
    if not state_dict:
        return "-"
    return max(state_dict.items(), key=lambda kv: kv[1])[0]


# ══════════════════════════════════════════════════════════════════════════════
# TRACK GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def _fit_stadium_params(marker_positions: list):
    pts = np.array(marker_positions, dtype=float)
    if len(pts) >= 4:
        left_centre  = (pts[0] + pts[3]) / 2.0
        right_centre = (pts[1] + pts[2]) / 2.0
        cx    = float((left_centre[0] + right_centre[0]) / 2.0)
        cy    = float((left_centre[1] + right_centre[1]) / 2.0)
        axis_vec = right_centre - left_centre
        angle = float(np.arctan2(axis_vec[1], axis_vec[0]))
        a     = float(np.linalg.norm(axis_vec) / 2.0)
        r     = max((np.linalg.norm(pts[0]-pts[3]) + np.linalg.norm(pts[1]-pts[2])) / 4.0, 1.0)
    else:
        cx = float(pts[:, 0].mean()); cy = float(pts[:, 1].mean())
        centred = pts - np.array([cx, cy])
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        angle = float(np.arctan2(Vt[0, 1], Vt[0, 0]))
        proj_long  = centred @ Vt[0]; proj_short = centred @ Vt[1]
        r = max(float(np.mean(np.abs(proj_short))), 1.0)
        a = max(float(np.max(np.abs(proj_long))) - r, 0.0)
    return cx, cy, a, r, angle


def fit_stadium(marker_positions: list, n: int = STADIUM_SAMPLES,
                cx: float = None, cy: float = None) -> list:
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
    for i in range(n_semi):
        th = -math.pi / 2.0 + math.pi * i / max(n_semi - 1, 1)
        pts_out.append(centre + a * u + r * (math.cos(th) * u + math.sin(th) * v))
    for i in range(1, n_str + 1):
        t = i / (n_str + 1)
        pts_out.append(centre + (1.0 - 2.0 * t) * a * u + r * v)
    for i in range(n_semi):
        th = math.pi / 2.0 + math.pi * i / max(n_semi - 1, 1)
        pts_out.append(centre - a * u + r * (math.cos(th) * u + math.sin(th) * v))
    for i in range(1, n_str + 1):
        t = i / (n_str + 1)
        pts_out.append(centre + (2.0 * t - 1.0) * a * u - r * v)
    pts_arr = np.array(pts_out, dtype=float)
    m = len(pts_arr)
    if m != n:
        idx = np.round(np.linspace(0, m - 1, n)).astype(int)
        pts_arr = pts_arr[idx]
    return list(zip(pts_arr[:, 0].astype(int).tolist(),
                    pts_arr[:, 1].astype(int).tolist()))


def detect_lanes(track_markers: dict) -> dict:
    result = {}
    inner_pts  = [marker_center(track_markers[i]) for i in INNER_SET  if i in track_markers]
    middle_pts = [marker_center(track_markers[i]) for i in MIDDLE_SET if i in track_markers]
    outer_pts  = [marker_center(track_markers[i]) for i in OUTER_SET  if i in track_markers]
    result.update({"n_inner": len(inner_pts), "n_middle": len(middle_pts),
                   "n_outer": len(outer_pts)})
    all_pts = inner_pts + middle_pts + outer_pts
    if len(all_pts) < 3:
        return result
    all_arr = np.array(all_pts, dtype=float)
    cx, cy  = float(all_arr[:, 0].mean()), float(all_arr[:, 1].mean())
    if len(inner_pts)  >= 2:
        result["inner_curve"]  = fit_stadium(inner_pts,  n=STADIUM_SAMPLES, cx=cx, cy=cy)
    if len(middle_pts) >= 2:
        result["middle_curve"] = fit_stadium(middle_pts, n=STADIUM_SAMPLES, cx=cx, cy=cy)
    if len(outer_pts)  >= 2:
        result["outer_curve"]  = fit_stadium(outer_pts,  n=STADIUM_SAMPLES, cx=cx, cy=cy)
    # Build true lane centrelines (equidistant from both boundaries)
    if "inner_curve" in result and "middle_curve" in result:
        ia = np.array(result["inner_curve"],  dtype=float)
        ma = np.array(result["middle_curve"], dtype=float)
        ca = ((ia + ma) / 2.0).astype(int)
        result["lane1_centre"] = list(zip(ca[:, 0].tolist(), ca[:, 1].tolist()))
        result["lane1_ref"]    = result["lane1_centre"]
        result["lane1_ready"]  = True
    if "middle_curve" in result and "outer_curve" in result:
        ma2 = np.array(result["middle_curve"], dtype=float)
        oa  = np.array(result["outer_curve"],  dtype=float)
        ca2 = ((ma2 + oa) / 2.0).astype(int)
        result["lane2_centre"] = list(zip(ca2[:, 0].tolist(), ca2[:, 1].tolist()))
        result["lane2_ref"]    = result["lane2_centre"]
        result["lane2_ready"]  = True
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MARKER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_all_markers(frame: np.ndarray):
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
# CAR IDENTIFICATION & TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def identify_cars(car_markers: dict, car_id: int) -> dict:
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
        cars[mid] = {"front": front, "rear": rear_center,
                     "midpoint": mid_pt, "heading": heading}
    return cars


def estimate_speed(curr_pos, prev_pos, dt: float) -> float:
    if prev_pos is None or dt <= 0:
        return 0.0
    return float(np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
                 / (dt * SCALING_FACTOR))


def update_tracker(cars: dict, current_time: float) -> None:
    for rear_id, cd in cars.items():
        if rear_id not in tracker:
            tracker[rear_id] = {"status": None, "center": None,
                                 "last_time": None, "heading": None, "speed": 0.0}
        t = tracker[rear_id]
        last_time = t["last_time"]
        if last_time is None or (current_time - last_time >= STATUS_UPDATE_INTERVAL):
            dt    = (current_time - last_time) if last_time else 0.0
            speed = estimate_speed(cd["midpoint"], t["center"], dt)
            t["speed"]   = speed
            t["status"]  = "moving" if speed > SPEED_THRESHOLD else "stopped"
            t["center"]  = cd["midpoint"]
            t["last_time"] = current_time
            t["heading"] = cd["heading"]


# ══════════════════════════════════════════════════════════════════════════════
# PATH & CURVE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def project_onto_curve(pos: tuple, curve_pts: list) -> int:
    arr = np.array(curve_pts, dtype=float)
    return int(np.argmin(np.linalg.norm(arr - np.array(pos, dtype=float), axis=1)))


def local_path_frame(curve_pts: list, idx: int):
    n      = len(curve_pts)
    p_prev = np.array(curve_pts[(idx - 1) % n], dtype=float)
    p_curr = np.array(curve_pts[idx],            dtype=float)
    p_next = np.array(curve_pts[(idx + 1) % n], dtype=float)
    tangent = p_next - p_prev
    norm_t  = np.linalg.norm(tangent)
    tangent = tangent / norm_t if norm_t >= 1e-6 else np.array([1.0, 0.0])
    left_normal = np.array([-tangent[1], tangent[0]])
    return p_curr, tangent, left_normal


def compute_lane_measurements(car_pos: tuple, heading_vec: tuple,
                               curve_pts: list) -> dict:
    idx = project_onto_curve(car_pos, curve_pts)
    p_ref, tangent, left_normal = local_path_frame(curve_pts, idx)
    pos_vec       = np.array(car_pos, dtype=float) - p_ref
    lateral_error = float(np.dot(pos_vec, left_normal))
    tangent_angle = float(np.degrees(np.arctan2(-tangent[1], tangent[0])) % 360)
    if np.linalg.norm(np.array(heading_vec, dtype=float)) < 1e-6:
        car_heading_angle = tangent_angle
    else:
        car_heading_angle = heading_to_angle(heading_vec)
    heading_error = wrap_angle_deg(tangent_angle - car_heading_angle)
    return {
        "idx": idx, "path_point": tuple(p_ref.astype(int)),
        "tangent": tangent, "left_normal": left_normal,
        "lateral_error": lateral_error, "heading_error": heading_error,
        "tangent_angle": tangent_angle, "car_heading_angle": car_heading_angle,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OBSTACLE DETECTION & CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _classify_obstacle_distance(d: float) -> dict:
    def _ls(x, a, b):  return 1.0 if x<=a else 0.0 if x>=b else (b-x)/(b-a+1e-9)
    def _tri(x,a,b,c):
        if x<=a or x>=c: return 0.0
        return (x-a)/(b-a+1e-9) if x<b else (c-x)/(c-b+1e-9)
    def _rs(x, a, b):  return 0.0 if x<=a else 1.0 if x>=b else (x-a)/(b-a+1e-9)
    return {"blocking": _ls(d,45,70), "near": _tri(d,60,110,170), "clear": _rs(d,130,180)}


def nearest_relevant_obstacle(car_pos, tangent, left_normal, obstacles) -> dict:
    _CLEAR = {"distance": float("inf"), "side_offset": 0.0,
              "state": "clear", "point": None}
    if not obstacles:
        return _CLEAR
    t  = tangent     / (np.linalg.norm(tangent)     + 1e-9)
    n  = left_normal / (np.linalg.norm(left_normal) + 1e-9)
    p0 = np.array(car_pos, dtype=float)
    best_dist, best = float("inf"), None
    for obs in obstacles:
        vec   = np.array(obs, dtype=float) - p0
        along = float(np.dot(vec, t))
        side  = float(np.dot(vec, n))
        eucl  = float(np.linalg.norm(vec))
        if along < 0 or along > OBSTACLE_LOOKAHEAD:
            continue
        if abs(side) > OBSTACLE_TRACK_HALF_WIDTH:
            continue
        if eucl < best_dist:
            best_dist, best = eucl, (obs, side)
    if best is None:
        return _CLEAR
    return {"distance": best_dist, "side_offset": best[1],
            "state": dominant_state(_classify_obstacle_distance(best_dist)),
            "point": best[0]}


# ══════════════════════════════════════════════════════════════════════════════
# LANE-CHANGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _lane_obstacle_free(car_pos, lanes, lane_num) -> bool:
    ref = lanes.get(f"lane{lane_num}_ref", [])
    if not ref:
        return False
    idx = project_onto_curve(car_pos, ref)
    _, tangent, ln = local_path_frame(ref, idx)
    return nearest_relevant_obstacle(car_pos, tangent, ln,
                                     obstacle_positions)["state"] == "clear"


def adjacent_lane_cars(cars, lane_num) -> list:
    return [cid for cid in cars
            if lane_state.get(cid, {}).get("lane", 1) == lane_num]


def cooperative_gap_ok(car_pos, adj_cars, cars) -> bool:
    p0 = np.array(car_pos, dtype=float)
    for cid in adj_cars:
        if cid not in cars:
            continue
        gap     = float(np.linalg.norm(np.array(cars[cid]["midpoint"],
                                                 dtype=float) - p0))
        closing = tracker.get(cid, {}).get("speed", 0.0)
        if gap < COOP_MIN_GAP:
            return False
        if closing > COOP_MAX_CLOSING_SPEED and gap < COOP_MIN_GAP * 1.5:
            return False
    return True


def apply_implicit_coop(adj_cars, duration: float = 1.5) -> None:
    expiry = time.time() + duration
    for cid in adj_cars:
        coop_slowdown_until[cid] = expiry


# ══════════════════════════════════════════════════════════════════════════════
# DRIVING POLICY BRANCHES
# ══════════════════════════════════════════════════════════════════════════════

def egocentric_decide_lane(car_id, car_pos, lanes, obstacle_info, now) -> int:
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
    if car_id not in lane_state:
        lane_state[car_id] = {"lane": 1, "timer": now, "overtaking": False}
    st      = lane_state[car_id]
    current = st["lane"]
    obs_bad = obstacle_info["state"] in ("blocking", "near")
    if obs_bad:
        adjacent = 2 if current == 1 else 1
        if lanes.get(f"lane{adjacent}_ready", False) and adjacent != current:
            adj_cars = adjacent_lane_cars(cars, adjacent)
            if (_lane_obstacle_free(car_pos, lanes, adjacent)
                    and cooperative_gap_ok(car_pos, adj_cars, cars)):
                apply_implicit_coop(adj_cars)
                st["lane"] = adjacent; st["timer"] = now; st["overtaking"] = True
    if st["overtaking"] and st["lane"] == 2:
        if (now - st["timer"] >= LANE_CHANGE_HOLD) and not obs_bad:
            if _lane_obstacle_free(car_pos, lanes, 1):
                st["lane"] = 1; st["timer"] = now; st["overtaking"] = False
    return st["lane"]


def decide_lane(car_id, car_pos, lanes, obstacle_info, cars, now) -> int:
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

def _compute_lookahead(car_pos, heading_vec, lane_ref,
                       outer_ref=None) -> tuple:
    """
    Pure-pursuit steering angle (bicycle kinematic model).

    1. Find nearest point on lane_ref (true lane centreline).
    2. Walk forward LOOKAHEAD_DIST px of arc (adaptive: shorter on curves).
    3. Apply pure-pursuit formula δ = atan2(2·L·sin α, l_d).
    4. Boundary-push: if car crosses the outer boundary, add corrective steer.
    5. Cap by segment, snap to nearest SERVO_STEPS entry.
    """
    arr = np.array(lane_ref, dtype=float)
    pos = np.array(car_pos,  dtype=float)
    ni  = int(np.argmin(np.linalg.norm(arr - pos, axis=1)))
    n   = len(lane_ref)

    # Adaptive lookahead distance (shorter on curves)
    p_prev = arr[(ni - 2) % n]; p_curr = arr[ni]; p_next = arr[(ni + 2) % n]
    v1, v2    = p_curr - p_prev, p_next - p_curr
    cross     = abs(float(np.cross(v1, v2)))
    seg_len   = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + 1e-9
    curvature = cross / (seg_len ** 2)
    ld_dist   = max(MIN_LOOKAHEAD, LOOKAHEAD_DIST - K_CURVATURE_LD * curvature)

    # Walk curve to find lookahead target
    accum, target = 0.0, arr[ni]
    for i in range(ni, ni + n - 1):
        i0  = i % n;  i1 = (i + 1) % n
        seg = np.linalg.norm(arr[i1] - arr[i0])
        if accum + seg >= ld_dist:
            frac   = (ld_dist - accum) / (seg + 1e-9)
            target = arr[i0] + frac * (arr[i1] - arr[i0])
            break
        accum += seg

    # Pure-pursuit formula
    hx, hy = heading_vec
    theta  = math.atan2(hy, hx) if math.hypot(hx, hy) > 1e-4 else 0.0
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    alpha  = math.atan2(math.sin(math.atan2(dy, dx) - theta),
                        math.cos(math.atan2(dy, dx) - theta))
    ld    = math.hypot(dx, dy) + 1e-9
    delta = math.atan2(2.0 * WHEELBASE_PX * math.sin(alpha), ld)

    # Boundary-push: steer back if car crosses the outer lane boundary
    if outer_ref is not None:
        outer_arr = np.array(outer_ref, dtype=float)
        oi  = int(np.argmin(np.linalg.norm(outer_arr - pos, axis=1)))
        op  = outer_arr[oi]
        ot  = outer_arr[(oi + 1) % len(outer_arr)] - outer_arr[(oi - 1) % len(outer_arr)]
        ot  = ot / (np.linalg.norm(ot) + 1e-9)
        on  = np.array([-ot[1], ot[0]])
        overshoot = float(np.dot(pos - op, on))
        if overshoot > 0:
            delta -= K_BOUNDARY_PUSH * overshoot

    return float(np.clip(delta, -DELTA_MAX, DELTA_MAX)), tuple(target.astype(int))


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED COORDINATION
# ══════════════════════════════════════════════════════════════════════════════

def _apply_coordination(cars, base_speed) -> dict:
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
                speeds[yielder_id] = STOP_SPEED
                _pp_waiting[yielder_id] = True
            elif d <= D_WARN:
                speeds[yielder_id] = min(speeds[yielder_id], SLOW_SPEED)
                _pp_waiting[yielder_id] = True
            else:
                _pp_waiting[yielder_id] = False
    return speeds


# ══════════════════════════════════════════════════════════════════════════════
# CURVE SPEED & SEGMENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _curvature_at(car_pos, ref) -> float:
    """Compute local curvature κ at the nearest point on ref."""
    n   = len(ref)
    idx = project_onto_curve(car_pos, ref)
    p0  = np.array(ref[(idx - 2) % n], dtype=float)
    p1  = np.array(ref[idx],            dtype=float)
    p2  = np.array(ref[(idx + 2) % n], dtype=float)
    v1, v2    = p1 - p0, p2 - p1
    cross     = abs(float(np.cross(v1, v2)))
    seg_len   = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + 1e-9
    return cross / (seg_len ** 2)


def _classify_segment(car_pos, lanes, lane: int = 1) -> tuple:
    ref = lanes.get(f"lane{lane}_ref", [])
    if not ref or len(ref) < 5:
        return "straight", 0.0
    kappa = _curvature_at(car_pos, ref)
    if kappa > CURVE_SHARP_THRESH:
        return "sharp_curve", round(kappa, 3)
    if kappa > CURVE_LIGHT_THRESH:
        return "light_curve", round(kappa, 3)
    return "straight", round(kappa, 3)


def _apply_curve_slowdown(car_pos: tuple, lanes: dict, lane: int = 1) -> float:
    ref = lanes.get(f"lane{lane}_ref", [])
    if not ref or len(ref) < 5:
        return CRUISE_SPEED
    kappa = _curvature_at(car_pos, ref)
    if kappa > CURVE_SHARP_THRESH:
        return SLOW_SPEED
    if kappa > CURVE_LIGHT_THRESH:
        return CRUISE_SPEED * 0.85
    return CRUISE_SPEED


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING / VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def draw_lanes(frame, lanes):
    overlay = frame.copy()
    if "inner_curve" in lanes and "middle_curve" in lanes:
        cv2.fillPoly(overlay, [np.vstack([np.array(lanes["inner_curve"],  np.int32),
                                          np.array(lanes["middle_curve"], np.int32)[::-1]])],
                     LANE1_FILL_COLOR)
    if "middle_curve" in lanes and "outer_curve" in lanes:
        cv2.fillPoly(overlay, [np.vstack([np.array(lanes["middle_curve"], np.int32),
                                          np.array(lanes["outer_curve"],  np.int32)[::-1]])],
                     LANE2_FILL_COLOR)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    for key, color in [("inner_curve",  INNER_LINE_COLOR),
                       ("middle_curve", MIDDLE_LINE_COLOR),
                       ("outer_curve",  OUTER_LINE_COLOR)]:
        if key in lanes:
            cv2.polylines(frame, [np.array(lanes[key], np.int32)], True, color, 2)
    vis_centres = []
    for k1, k2 in [("inner_curve", "middle_curve"), ("middle_curve", "outer_curve")]:
        if k1 in lanes and k2 in lanes:
            mid = ((np.array(lanes[k1], float) + np.array(lanes[k2], float)) / 2).astype(int)
            vis_centres.append([tuple(p) for p in mid.tolist()])
    for pts in vis_centres:
        nn = len(pts)
        for i in range(0, nn, 8):
            cv2.line(frame, pts[i], pts[(i + 4) % nn], REF_LINE_COLOR, 1)
    return frame


def draw_guide_line(frame, ref_pose, target, obstacle_close):
    color = (0, 0, 220) if obstacle_close else (255, 255, 255)
    cv2.line(frame,   ref_pose, target, color, 2)
    cv2.circle(frame, target, 9, color, 2)
    cv2.circle(frame, target, 3, color, -1)
    return frame


def draw_obstacles(frame, obstacles):
    for obs in obstacles:
        ox, oy  = obs
        overlay = frame.copy()
        cv2.circle(overlay, (ox, oy), OBSTACLE_RADIUS, (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.circle(frame, (ox, oy), OBSTACLE_RADIUS, (0, 0, 255), 2)
        cv2.circle(frame, (ox, oy), 6, (0, 0, 255), -1)
        cv2.putText(frame, "OBS", (ox + 8, oy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return frame


def draw_cars(frame, cars):
    for _, cd in cars.items():
        cv2.circle(frame, cd["rear"],     8, (0,   0, 255), -1)
        cv2.circle(frame, cd["midpoint"], 6, (0, 165, 255), -1)
        if cd["front"] is not None:
            cv2.circle(frame, cd["front"], 8, (0, 255, 0), -1)
            cv2.line(frame, cd["rear"], cd["front"], (255, 0, 255), 2)
    return frame


def _label_lateral(e: float) -> str:
    if e < -25: return "far_right"
    if e <  -8: return "slgt_right"
    if e >  25: return "far_left"
    if e >   8: return "slgt_left"
    return "aligned"

def _label_heading(e: float) -> str:
    if e < -18 or e < -6: return "need_right"
    if e >  18 or e >  6: return "need_left"
    return "hdg_aligned"


def draw_hud(frame, car_id, servo, motor, lane_info, obstacle_info,
             steer_state, speed_state, current_lane, lc_state, policy,
             cam_mode: str = "RAW"):
    FONT, SCALE, THICKNESS, LINE_H, PAD = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1, 14, 6
    lc_tag   = "OVT" if lc_state.get("overtaking") else "NRM"
    coop_tag = " C!" if speed_state.get("coop_hint") else ""
    obs_d    = obstacle_info["distance"]
    obs_str  = f"{obs_d:.0f}px" if obs_d < float("inf") else "---"
    wait_tag = " [W]" if _pp_waiting.get(car_id, False) else ""
    hud_lines = [
        f"Car {car_id} | Ln{current_lane} {lc_tag} | {policy[:4].upper()}{wait_tag}",
        f"Srv {servo:+.2f} Mtr {motor:.2f}{coop_tag}",
        f"Lat {lane_info['lateral_error']:+.0f}px {steer_state.get('lateral','-')[:4]}",
        f"Hd  {lane_info['heading_error']:+.0f}° {steer_state.get('heading','-')[:4]}",
        f"Obs {obs_str} {obstacle_info['state'][:4]}",
        f"Spd {speed_state.get('zone','-')} {steer_state.get('obstacle','-')[:4]}",
        f"Cam {cam_mode}",          # ← shows calibration status in HUD
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
# MAIN CONTROL ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(frame: np.ndarray, car_id: int, cam_mode: str = "RAW"):
    """
    Execute one full sense–plan–act cycle.
    Returns (servo, motor, annotated_frame).
    """
    global obstacle_positions, _cycle_counter
    _cycle_counter += 1
    car_frame = frame.copy()
    now       = time.time()

    # 1. Detect
    car_markers, track_markers, obs_markers = detect_all_markers(car_frame)
    obstacle_positions = [marker_center(c) for c in obs_markers.values()]

    # 2. Scene state
    cars  = identify_cars(car_markers, car_id)
    lanes = detect_lanes(track_markers)
    update_tracker(cars, now)

    # Safe defaults
    servo        = 0.0; motor = 0.0
    steer_state  = {"lateral": "-", "heading": "-", "obstacle": "-"}
    speed_state  = {"zone": "-", "coop_hint": False}
    lane_info    = {"path_point": (0, 0), "lateral_error": 0.0, "heading_error": 0.0,
                    "tangent": np.array([1.0, 0.0]), "left_normal": np.array([0.0, -1.0])}
    obstacle_info = {"distance": float("inf"), "side_offset": 0.0,
                     "point": None, "state": "clear"}
    current_lane  = lane_state.get(car_id, {}).get("lane", 1)
    target_pt     = None
    policy        = DRIVING_POLICY.get(car_id, DEFAULT_POLICY)
    ref_curve     = lanes.get(f"lane{current_lane}_ref", [])
    events: list  = []

    if not ref_curve:
        return round(servo, 2), round(motor, 2), car_frame

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

        outer_boundary = (lanes.get("middle_curve") if current_lane == 1
                          else lanes.get("outer_curve"))
        raw_delta, target_pt = _compute_lookahead(
            car_ref, cars[car_id]["heading"], ref_curve, outer_ref=outer_boundary)

        # Segment-aware cap → snap to SERVO_STEPS
        seg_result = _classify_segment(car_ref, lanes, lane=current_lane)
        seg_name   = seg_result[0] if isinstance(seg_result, tuple) else seg_result
        if seg_name == "straight":
            delta_cap = MED_SMALL_SERVO
        elif seg_name == "light_curve":
            delta_cap = MED_SERVO
        else:
            delta_cap = MAX_MID_SERVO
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
        safety_stop   = (obstacle_info["state"] in ("blocking", "near")
                         and not_overtaking
                         and (single_lane or policy == "cooperative"))

        if safety_stop:
            motor = STOP_SPEED;  events.append("safety_stop")
        else:
            motor = float(np.clip(raw_motor, 0.0, MAX_SPEED))

        coop_active = time.time() < coop_slowdown_until.get(car_id, 0.0)
        speed_state = {
            "zone":      "stop" if motor == STOP_SPEED else ("slow" if motor <= SLOW_SPEED else "go"),
            "coop_hint": coop_active,
        }
        last_command_time[car_id] = now

    # 4. Draw
    car_frame = draw_lanes(car_frame, lanes)
    car_frame = draw_obstacles(car_frame, obstacle_positions)
    car_frame = draw_cars(car_frame, cars)
    if car_id in cars and cars[car_id]["rear"] is not None and target_pt is not None:
        car_frame = draw_guide_line(car_frame, cars[car_id]["rear"], target_pt,
                                    obstacle_info["state"] in ("blocking", "near"))
    lc_st     = lane_state.get(car_id, {})
    car_frame = draw_hud(car_frame, car_id, servo, motor, lane_info, obstacle_info,
                         steer_state, speed_state, current_lane, lc_st, policy,
                         cam_mode=cam_mode)

    # 5. Log
    car_data  = cars.get(car_id, {})
    mid_      = car_data.get("midpoint", (0, 0))
    hv        = car_data.get("heading",  (1, 0))
    theta_est = (float(np.degrees(np.arctan2(-hv[1], hv[0])) % 360)
                 if np.linalg.norm(hv) > 1e-6 else 0.0)
    seg_r     = _classify_segment(mid_, lanes, lane=current_lane)
    entry = _build_log_entry(
        t=now, k=_cycle_counter, car_id=car_id, policy=policy,
        pose=[float(mid_[0]), float(mid_[1]), theta_est],
        lane=current_lane,
        segment=seg_r[0] if isinstance(seg_r, tuple) else seg_r,
        curvature=seg_r[1] if isinstance(seg_r, tuple) else 0.0,
        servo=servo, motor=motor,
        waiting=_pp_waiting.get(car_id, False),
        lateral_error=lane_info["lateral_error"],
        heading_error=lane_info["heading_error"],
        obstacle_info=obstacle_info, cars=cars, events=events,
    )
    _log_entries.append(entry)

    return round(servo, 2), round(motor, 2), car_frame


# ══════════════════════════════════════════════════════════════════════════════
# LAUNCHER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def boot(ip, username, password):
    os.system(f'putty -ssh {username}@{ip} -pw {password} -m "./player/launch.txt"')


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minicar lane controller — with optional camera calibration.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # No calibration (raw frames)\n"
            "  python auto_control_calibrated.py -n 1\n\n"
            "  # Run calibration wizard before starting (chessboard)\n"
            "  python auto_control_calibrated.py -n 1 --calibrate\n\n"
            "  # Run calibration wizard with ChArUco board\n"
            "  python auto_control_calibrated.py -n 1 --calibrate --board charuco\n\n"
            "  # Load previously saved calibration\n"
            "  python auto_control_calibrated.py -n 1 --calib-file calib.npz\n"
        ),
    )
    parser.add_argument("-n", "--cars", nargs="+", type=int, default=None,
                        help="Car ID(s) to control")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run the calibration wizard before starting")
    parser.add_argument("--calib-file", type=str, default=None,
                        metavar="PATH",
                        help="Path to a saved .npz calibration file")
    parser.add_argument("--board", type=str, default="chess",
                        choices=["chess", "charuco"],
                        help="Calibration board type  (default: chess)")
    parser.add_argument("--chess-size", type=int, nargs=2,
                        default=[8, 6], metavar=("COLS", "ROWS"),
                        help="Inner corners of chessboard  (default: 8 6)")
    parser.add_argument("--square-mm", type=float, default=38.0,
                        help="Square size in mm  (default: 38.0)")
    parser.add_argument("--marker-mm", type=float, default=18.0,
                        help="ChArUco marker size in mm  (default: 18.0)")
    args = parser.parse_args()

    if args.cars is None:
        print("No minicar selected!  Use -n <ID>.")
        sys.exit(0)

    # ── Camera selection ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    use_calib = args.calibrate or (args.calib_file is not None)
    if use_calib:
        print("  Camera mode : CALIBRATED")
        cam = CalibratedCamera(
            cam_index  = 0,
            calib_file = args.calib_file,
            board_type = args.board,
            chess_size = tuple(args.chess_size),
            square_mm  = args.square_mm,
            marker_mm  = args.marker_mm,
        )
    else:
        print("  Camera mode : RAW  (no calibration)")
        cam = RawCamera(cam_index=0)
    print("═" * 60 + "\n")

    cam_mode_label = cam.MODE   # shown in the HUD

    # ── Network setup ─────────────────────────────────────────────────────────
    ip          = "192.168.0.201"
    username    = "cpslab1"
    password    = "cpslab1"
    remote_port = 6789

    response = os.system("ping -n 1 -w 200 {} | find \"Reply\"".format(ip))
    if response != 0:
        print("No connection to car Pi.")
        cam.release()
        sys.exit(0)

    import socket, struct, keyboard
    from threading import Thread

    s       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    car_idx = args.cars[0]

    th = Thread(target=boot, args=(ip, username, password), daemon=True)
    th.start()

    running = True
    clean   = False

    try:
        while running:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue

            servo, motor, vis = run(frame, car_idx, cam_mode=cam_mode_label)

            if keyboard.is_pressed("esc"):
                running = False
                clean   = True

            buffer = bytearray(struct.pack("<fff?", motor, servo, 0, clean))
            s.sendto(buffer, (ip, remote_port))
            print(f"Motor: {motor:.2f}  Servo: {servo:+.2f}\r", end="")
            time.sleep(1.0 / 100.0)

            cv2.imshow("Lane Controller", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cam.release()
        s.close()
        save_log()
        cv2.destroyAllWindows()
        print("\n[done] Log saved.")
