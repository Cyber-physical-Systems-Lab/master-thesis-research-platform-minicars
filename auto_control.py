"""
auto_control.py  (EKF removed — direct ArUco pose + PI speed control)
======================================================================
Centralised vision-based controller for the minicar testbed.

Architecture (sense → plan → act)
──────────────────────────────────
Sensing  : Overhead USB camera + ArUco marker detection.
           Three ArUco dictionaries run in parallel:
             4×4_50  – car markers (front/rear ID pairs per CAR_MARKER_PAIRS;
                                       car 0:(49,0), car 1:(46,1), car 2:(47,2))
             5×5_50  – track boundary (inner / middle / outer rings)
             6×6_250 – static obstacles
Planning : The rear-marker centroid (falling back to the front-marker
           centroid when the rear marker is occluded) is fed directly —
           no filter/estimator in between — into a pure-pursuit
           controller. Rule-based distance-threshold coordination is
           applied on top.
Acting   : Quantised servo + motor commands sent via TCP to each minicar.
           Longitudinal speed is regulated by a lightweight PI controller
           with output-rate smoothing (mirrors _smooth_servo()).

Entry point
───────────
run(frame, car_id) → (servo, motor, annotated_frame)
Called once per camera frame from the main loop.

Pose source: raw_rear / front marker centroids from identify_cars()
(no filtering/estimation stage — direct measurement).

Log schema (per entry, written by save_log)
────────────────────────────────────────────
  t, k, car_id, policy, pose[x,y,θ], lane, segment, curvature,
  command{servo,motor}, waiting, lateral_error, heading_error,
  obstacle{state,distance_px}, distances{a-b}, events[…]
"""

import json, math, os, sys, time, threading
import random
import cv2
import numpy as np
from cv2 import aruco

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ArUco dictionaries
CAR_DICT       = aruco.DICT_4X4_50
# Per-car marker pairs: car_id → (front_marker_id, rear_marker_id)
# Car 0: front=49, rear=0  | Car 1: front=46, rear=1  | Car 2: front=47, rear=2
CAR_MARKER_PAIRS: dict = {
    0: (49, 0),
    1: (46, 1),
    2: (47, 2),
}
FRONT_MARKER_ID      = 49   # kept for backward-compat (car 0 front); use CAR_MARKER_PAIRS
# FRONT_OWN_MAX_DIST_PX: if a front marker is farther than this from its paired
# rear marker, ownership is not claimed (treat as occluded for that car).
FRONT_OWN_MAX_DIST_PX = 220   # px — max rear↔front dist to claim ownership

# Car IDs that currently have an open UDP socket in player_launcher.
# Populated externally by player_launcher via _ac.ACTIVE_CAR_IDS.add(cid).
# When a car ID is NOT in this set and its markers are visible on the
# track (spanning both lanes or parked), each visible marker for that
# car is treated as a static obstacle position.
ACTIVE_CAR_IDS: set = set()   # filled by player_launcher at startup
TRACK_DICT     = aruco.DICT_5X5_50
OBSTACLE_DICT  = aruco.DICT_6X6_250

INNER_SET  = [0, 1, 2, 3]
MIDDLE_SET = [4, 5, 6, 7]
OUTER_SET  = [8, 9, 10, 11]
STADIUM_SAMPLES = 300

LANE_HALF_WIDTH  = 25    # px — approximate half lane width
LANE_HYSTERESIS  = 10    # px — dead-band to prevent classification chattering

# Obstacle geometry — plain distance-threshold rule (no drawn zones), the
# same four-state principle used for car-to-car coordination.
# RENAMED (methodology alignment): d_safe is now the MOST PERMISSIVE /
# farthest boundary and d_warn is the STRICTER boundary closer to
# collision, matching the paper's d_col < d_warn < d_safe ordering.
#   d <= OBSTACLE_D_COL                     -> "collision" (physical contact)
#   OBSTACLE_D_COL  < d <= OBSTACLE_D_WARN  -> "blocking"  (stop / override motion)
#   OBSTACLE_D_WARN < d <= OBSTACLE_D_SAFE  -> "near"      (decision / yield zone)
#   d > OBSTACLE_D_SAFE                     -> "clear"     (continue nominal motion)
OBSTACLE_LOOKAHEAD        = 120
OBSTACLE_TRACK_HALF_WIDTH = 52    # fallback corridor only — real checks use each lane's own half-width
OBSTACLE_D_COL  = 30    # px — physical-contact radius (car-to-object), safety-metric only
OBSTACLE_D_WARN = 70    # px — hard stop radius (was OBSTACLE_D_SAFE's value)
OBSTACLE_D_SAFE = 170   # px — decision/yield radius, most permissive (was OBSTACLE_D_WARN's value)

# Speed levels (normalised PWM, 0–1)
MAX_SPEED    = 0.50
CRUISE_SPEED = 0.42
SLOW_SPEED   = 0.38
MIN_SPEED    = 0.38
STOP_SPEED   = 0.00

# Servo / steering limits (radians, sent as normalised floats)
MAX_SERVO       = 0.50

# Bicycle model / pure-pursuit
# Camera frame rate — keep in sync with cap.set(CAP_PROP_FPS) in init_camera()
CAMERA_FPS: float = 60.0

WHEELBASE_PX    = 68           # rear-axle to front-axle in pixels
K_DD            = 165          # look-ahead distance in pixels (~2 car-lengths)
_RAD_TO_SERVO   = math.pi / 4.0

# Coordination thresholds (methodology §Coordination Logic)
# RENAMED (methodology alignment): d_safe is the most permissive / farthest
# decision boundary, d_warn is the stricter near-miss boundary, matching
# the paper's d_col < d_warn < d_safe ordering (previously D_SAFE held the
# smaller/stricter value and D_WARN the larger/lenient one — inverted).
D_COL                  = 25    # px — physical-contact radius (car-to-car), safety-metric only
D_WARN                 = 57    # px — near-miss boundary (was D_SAFE's value)
D_SAFE                 = 115   # px — decision boundary, most permissive (was D_WARN's value)
COOP_MIN_GAP           = 290
COOP_MAX_CLOSING_SPEED = 30
LANE_CHANGE_HOLD       = 2.5

# Driving policies
DRIVING_POLICY: dict = {}
DEFAULT_POLICY = "cooperative"

# Marker caching
# Car-marker occlusion uses tiers. Static markers cached indefinitely.

# Experiment metadata  (overridden by --scenario / --policy CLI args)
LOG_SCENARIO   = "S1"
LOG_POLICY     = "cooperative"
LOG_RUN_NAME: str | None = None  # set from --run-name; None = no sub-folder
LOG_REPETITION: int = 1           # set from --repetition; used only in the log filename

# ── Physical scale (set once with a tape measure) ───────────────────────────
# Measure the real-world distance between outer markers 9 and 10 (the longest
# straight on the track) and enter it here in centimetres.  This is the ONLY
# physical measurement required by the pipeline.
#
# Non-calibrated mode: px_per_cm = (pixel distance 9↔10) / TRACK_LONG_AXIS_CM
# Calibrated mode    : same value seeds the world-coordinate frame origin
#                      (track centroid) so both modes share the same unit.
#
# Cross-check pairs (logged in meta for traceability — no extra measurement
# needed at runtime, all computed automatically from marker pixel positions):
#   LONG axis : markers 9  ↔ 10   ← baseline (longest straight)
#   SHORT axis: markers 2  ↔ 6    ← shortest straight, independent cross-check
TRACK_LONG_AXIS_CM: float = 148.5   # ← fill in with tape measure before first run

# ── Persistent run counter (survives restarts, like EEPROM) ─────────────────

# Visualisation colours (BGR)
LANE1_FILL_COLOR  = (0, 160, 60)
LANE2_FILL_COLOR  = (0, 100, 180)
INNER_LINE_COLOR  = (0, 230, 120)
MIDDLE_LINE_COLOR = (255, 255, 0)
OUTER_LINE_COLOR  = (80, 200, 255)
REF_LINE_COLOR    = (45, 45, 45)
OBSTACLE_RADIUS   = 50           # px — visual radius of obstacle circles

# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE
# ══════════════════════════════════════════════════════════════════════════════

_obstacle_positions: list = []
_obstacle_marker_ids: dict = {}     # position tuple -> ArUco marker_id, rebuilt every run() frame
_inactive_car_obstacles: list = []  # refreshed every identify_cars() call; merged into
                                     # _obstacle_positions in run() so parked/unregistered
                                     # car markers feed the SAME obstacle pipeline as 6x6 markers
_lane_state:         dict = {}
_coop_slowdown_until: dict = {}
_pp_waiting:         dict = {}
_TRACK_FORWARD             = None
_TRACK_GT_SNAPSHOT: dict   = {}   # refreshed every detect_lanes call (both lanes); see Limitations re: self-referential reference truth

# Marker persistence
_last_track_markers: dict = {}   # id → (corner_array, timestamp)
_last_obs_positions: dict = {}   # id → ((x,y), timestamp)
_last_lanes:         dict = {}   # cached from most recent detect_lanes() call
_car_marker_cache:   dict = {}   # car_id → {front_corner, rear_corner, *_time}
_last_marker_debug:  dict = {}   # cached raw ArUco corners/ids for draw_marker_debug()

# Raw-pose velocity cache per car (replaces EKF state): car_id → {'pos': (x,y), 't': float, 'v': float}
# Used to derive a simple finite-difference speed estimate directly from the
# raw marker centroid (rear, or front on rear-occlusion) — no filter/estimator.
_vel_cache: dict = {}

# PI longitudinal controller state per car: car_id → {i, e_prev, t_prev, v_prev, pos_prev, cmd_prev}
_spd_pid: dict = {}


# ── Servo low-pass filter (per car) ─────────────────────────────────────────
# Exponential moving average damps single-frame ArUco heading-noise spikes.
# SERVO_ALPHA: 0.0 = frozen output, 1.0 = no filtering.
# Increase toward 0.4 if sluggish; lower toward 0.15 for more smoothing.
# Servo convention: negative = LEFT turn, positive = RIGHT turn, 0.0 = straight.
# At 60 Hz, SERVO_ALPHA=0.25 → time constant τ ≈ 1/(60*(1-0.75)) ≈ 0.067 s (~4 frames).
# Lower toward 0.15 for more smoothing; raise toward 0.4 if response feels sluggish.
SERVO_ALPHA: float = 0.25
_last_servo:               dict  = {}    # car_id -> smoothed servo
_last_cmd:                 dict  = {}    # car_id -> (servo, motor) last sent
_car_fwd_step:             dict  = {}    # car_id -> confirmed fwd step
_CAR_OCCLUDE_HARD_STOP_FRAMES: int = 20
_OCCLUDE_VEL_DECAY:        float = 0.85
_car_occlusion_state:      dict  = {}    # car_id -> {tier, frames_occluded, prev_tier}


def _smooth_servo(car_id: int, new_servo: float) -> float:
    """First-order EMA low-pass on the servo command.

    Reduces single-frame ArUco heading-jitter impact on physical hardware.
    Time constant ≈ 3 frames at 20 Hz (SERVO_ALPHA = 0.25).
    Servo: negative = LEFT turn, positive = RIGHT turn, 0.0 = straight.
    """
    prev = _last_servo.get(car_id, new_servo)
    out  = SERVO_ALPHA * new_servo + (1.0 - SERVO_ALPHA) * prev
    _last_servo[car_id] = out
    return out


# ── PI speed-controller tuning knobs ────────────────────────────────────────
# Replaces the former EKF-driven PID. Velocity feedback now comes from
# _cached_velocity() (raw marker-centroid finite difference) instead of
# EKF state, so gains were re-tuned lower to avoid amplifying raw-pose
# jitter (the EKF used to smooth this out before the PID ever saw it).
_PI_KP        = 0.35   # proportional gain (NEEDS TUNING on hardware)
_PI_KI        = 0.05   # integral gain (NEEDS TUNING on hardware)
_PI_I_CLAMP   = 0.30   # anti-windup integral clamp (normalised motor units)
# Motor command smoothing (EMA, mirrors _smooth_servo()/SERVO_ALPHA below)
_MOTOR_ALPHA   = 0.25   # 0.0 = frozen output, 1.0 = no filtering (NEEDS TUNING)
# CTE speed penalty — slow down when lateral error is large
_CTE_THRESHOLD   = 15.0  # px — below this, no speed reduction
_CTE_MAX_PENALTY = 0.15  # max fractional speed reduction (e.g. 0.15 → −15 %)
_CTE_PENALTY_K   = 0.005 # px⁻¹ — penalty slope beyond the threshold


# Logger
_log_entries:   list = []
_cycle_counter: int  = 0

# Calibration
_CAM_K       = np.eye(3,    dtype=np.float64)
_CAM_D       = np.zeros(5,  dtype=np.float64)
_CALIBRATED      = False
_DFOV      = None    # diagonal FOV tag parsed from .npz filename; "90", "78", or None
# Camera-height lookup (request 4): fixed mount heights measured with a tape
# measure for each supported DFOV rig. Populated automatically in
# load_calibration() ONLY when the corresponding DFOV tag is detected —
# no CLI flag needed, no guessing for untagged/unknown calibration files.
_DFOV_HEIGHT_CM = {"90": 170.0, "78": 220.0}
# _CAMERA_HEIGHT_CM = "None"   # numeric camera height in cm; set from _DFOV_HEIGHT_CM, else None
_CALIB_PATH      = None    # path of the .npz that was loaded; used to persist rvec/tvec back
_CAM_RVEC        = np.zeros((3, 1), dtype=np.float64)  # extrinsic rotation  (world→camera)
_CAM_TVEC        = np.zeros((3, 1), dtype=np.float64)  # extrinsic translation (world→camera)
_extrinsics_done = False   # set True once estimate_extrinsics() succeeds
_calib_lock      = threading.Lock()
# _calib_running = threading.Event()

CALIB_DEFAULT_OUT = "calib.npz"
CALIB_MIN_FRAMES  = 15
CALIB_CRITERIA    = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 35, 0.001)

# ══════════════════════════════════════════════════════════════════════════════
# RAW-POSE VELOCITY ESTIMATE (replaces EKF — direct finite-difference speed)
# ══════════════════════════════════════════════════════════════════════════════

def _update_velocity_cache(car_id: int, pos: tuple, now: float) -> float:
    """
    Finite-difference speed estimate (px/s) from consecutive raw marker
    centroids — replaces the EKF-derived velocity used previously.

    No filtering/estimation: this is a direct measurement, cached per car
    so callers (PI speed controller, cooperative gap-check) can read the
    last-known speed without recomputing it every call.
    """
    prev = _vel_cache.get(car_id)
    if prev is None or prev.get("t") is None:
        _vel_cache[car_id] = {"pos": pos, "t": now, "v": 0.0}
        return 0.0
    dt = now - prev["t"]
    if dt <= 0 or dt > 1.0:
        _vel_cache[car_id] = {"pos": pos, "t": now, "v": prev.get("v", 0.0)}
        return prev.get("v", 0.0)
    disp = float(np.linalg.norm(np.array(pos, float) - np.array(prev["pos"], float)))
    v = disp / dt
    _vel_cache[car_id] = {"pos": pos, "t": now, "v": v}
    return v


def _cached_velocity(car_id: int) -> float:
    """Return the last computed raw-pose speed estimate (px/s), or 0.0."""
    return _vel_cache.get(car_id, {}).get("v", 0.0)

# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def _lane_curve_len(lane_num: int) -> int:
    """Number of samples in the given lane's reference curve (for modular
    wraparound arithmetic on a closed-loop track). Returns 0 if unavailable."""
    curve = _last_lanes.get(f"lane{lane_num}_ref", [])
    return len(curve)


def _leader_follower_gap(car_a: int, car_b: int, cars: dict):
    """
    Resolve which of two same-lane cars is ahead (leader) and which is
    behind (follower) using their projected position along the shared lane
    reference curve, then return the physical bumper-to-bumper gap: the
    distance from the FOLLOWER's front-marker reference to the LEADER's
    rear-marker reference (falls back to whichever single marker centroid
    each car has available — mirrors the same occlusion-fallback already
    used by Pure Pursuit, so a temporarily occluded marker degrades to a
    conservative estimate instead of breaking the check).

    Returns (leader_id, follower_id, gap_px) or None if lane/segment data
    is unavailable for either car (e.g. cars not in the same lane).
    """
    ca, cb = cars.get(car_a), cars.get(car_b)
    if ca is None or cb is None:
        return None
    la, lb = ca.get("lane"), cb.get("lane")
    ia, ib = ca.get("segment_idx"), cb.get("segment_idx")
    if la is None or lb is None or la != lb or ia is None or ib is None:
        return None

    n = _lane_curve_len(la)
    if n <= 0:
        return None

    # How many curve-samples ahead b is of a (going forward along the
    # track direction the reference curve is stored in), and vice versa.
    # Whichever "ahead" distance is smaller identifies the true leader.
    ahead_b_of_a = (ib - ia) % n
    ahead_a_of_b = (ia - ib) % n
    if ahead_b_of_a <= ahead_a_of_b:
        leader, follower = car_b, car_a
    else:
        leader, follower = car_a, car_b

    ld, fd = cars[leader], cars[follower]
    # Leader's rear reference: real rear marker if visible, else whatever
    # fallback identify_cars() already resolved (front-as-rear substitute
    # on rear-occlusion) — same convention Pure Pursuit itself relies on.
    leader_rear = ld.get("rear") or ld.get("midpoint")
    # Follower's front reference: real front marker if visible, else fall
    # back to its own rear/midpoint (a conservative underestimate of the
    # true gap, which is the safe direction to err for a safety check).
    follower_front = fd.get("front") or fd.get("rear") or fd.get("midpoint")
    if leader_rear is None or follower_front is None:
        return None

    gap = float(np.linalg.norm(
        np.array(follower_front, float) - np.array(leader_rear, float)))
    return leader, follower, gap


def _build_log_entry(t: float, k: int, cars_data: dict, cars: dict) -> dict:
    """
    One log entry per camera frame — all cars nested under a single timestamp.

    Schema
    ------
    {
      "t":         float,
      "k":         int,
      "distances": {"0-1": {"euclidean_px": float, "same_lane": bool|null,
                             "lane_a": int|null, "lane_b": int|null,
                             "seg_delta": int|null,
                             "leader": int|null, "follower": int|null}, ...},
                             # NOTE: for same_lane pairs, euclidean_px is the
                             # leader-follower gap (follower front-marker to
                             # leader rear-marker), not raw centroid distance.
      "cars": {
        "<car_id>": {
          "policy":        str,
          "pose":          [x_px, y_px, theta_deg],
          "lane":          int,
          "segment":       str,
          "curvature":     float,
          "command":       {"servo": float, "motor": float},
          "waiting":       bool,
          "lateral_error": float,
          "heading_error": float,
          "obstacle":      {"state": str, "distance_px": float|null},
          "events": {
              "minicar_events": [str],
              "safety_events": {"car": [str], "object": [str]}
          },
          "occlusion_tier": str,
          "maneuvering":   bool
        }
      }
    }
    """
    # Pairwise inter-car distances (frame-level, same for all cars)
    ids = sorted(cars.keys())
    distances = {}
    # Same-lane car-to-car states are relabeled for the log to read as
    # physical-gap semantics (car-following context) rather than the
    # generic collision-avoidance wording used for cross-lane/object
    # checks. Internal branching (emergency_stop vs. hold-gap coordination)
    # still keys off the original _classify_safety() bucket below — only
    # the *label written to the log* changes here.
    #   safe      -> "no_gap"    (plenty of following distance)
    #   near_miss -> "hold_gap"  (should maintain constant distance)
    #   decision  -> "hold_gap"  (early yield zone, same treatment as near_miss)
    #   collision -> "small_gap" (physical contact / emergency_stop)
    _SAME_LANE_LABELS = {
        "safe": "no_gap",
        "decision": "hold_gap",
        "near_miss": "hold_gap",
        "collision": "small_gap",
    }

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            la  = cars[a].get("lane", None)
            lb  = cars[b].get("lane", None)
            sa  = cars[a].get("segment_idx", None)
            sb  = cars[b].get("segment_idx", None)
            same_lane = (la is not None and lb is not None and la == lb)

            # ── Same-lane pairs: leader-follower gap (front-of-follower to
            # rear-of-leader), not raw centroid distance — reflects the
            # actual physical clearance between two elongated car bodies
            # traveling the same path in sequence. ────────────────────────
            leader_id = follower_id = None
            if same_lane:
                lf = _leader_follower_gap(a, b, cars)
                if lf is not None:
                    leader_id, follower_id, eucl = lf
                else:
                    eucl = round(float(np.linalg.norm(
                        np.array(cars[a]["midpoint"], float) -
                        np.array(cars[b]["midpoint"], float))), 2)
            else:
                eucl = round(float(np.linalg.norm(
                    np.array(cars[a]["midpoint"], float) -
                    np.array(cars[b]["midpoint"], float))), 2)
            eucl = round(eucl, 2)

            # 4-way classification (request 1 + 2): collision/near_miss/decision/safe,
            # using the renamed D_COL < D_WARN < D_SAFE ordering.
            inter_state = _classify_safety(eucl, D_COL, D_WARN, D_SAFE)
            # Cars on different lanes are only flagged as near_miss/decision/
            # collision when at least one of the two is actively maneuvering
            # (overtaking or mid lane_switch). A car cruising in its own lane
            # must never be flagged against a car in the other lane just
            # because they are geometrically close at a bend.
            man_a = bool(cars_data.get(a, {}).get("maneuvering", False))
            man_b = bool(cars_data.get(b, {}).get("maneuvering", False))
            cross_lane_checked = bool(man_a or man_b)
            if not same_lane and not cross_lane_checked:
                inter_state = "safe"

            # Log label: same-lane pairs use gap-semantics wording
            # ("no_gap"/"hold_gap"/"small_gap"); cross-lane / object-style
            # pairs keep the original collision-avoidance wording as-is.
            logged_state = (_SAME_LANE_LABELS.get(inter_state, inter_state)
                            if same_lane else inter_state)

            distances[f"{a}-{b}"] = {
                "euclidean_px": eucl,
                "interaction_state": logged_state,
                "same_lane":    same_lane if (la is not None and lb is not None) else None,
                "lane_a": la, "lane_b": lb,
                "seg_delta": abs(sa - sb) if (sa is not None and sb is not None) else None,
                "cross_lane_checked": cross_lane_checked,
                "leader": leader_id, "follower": follower_id,
            }
            # "decision"/"hold_gap" now also fires as a discrete, queryable
            # event (request 1) alongside collision/near_miss, instead of
            # only being inferable indirectly from minicar_events. These
            # are CAR-to-CAR events -> recorded under safety_events["car"].
            if inter_state in ("collision", "near_miss", "decision"):
                for _pid in (a, b):
                    if _pid in cars_data:
                        _se = cars_data[_pid].setdefault("safety_events",
                                                          {"car": [], "object": []})
                        _se.setdefault("car", []).append(logged_state)

            if same_lane and leader_id is not None:
                # Same-lane semantics (request): "near_miss" (and the softer
                # "decision" zone) mean the follower should hold a constant
                # gap rather than closing further — NOT an emergency stop.
                # Only an actual "collision" (gap <= D_COL) triggers a hard
                # emergency_stop, and only for the follower (the car that is
                # actually closing the gap onto the car ahead).
                if inter_state == "collision":
                    _lane_state.setdefault(follower_id, {})["emergency_stop"] = True
                elif inter_state in ("near_miss", "decision"):
                    # Ask the PI speed controller (via _apply_implicit_coop /
                    # _coop_slowdown_until, already used by _apply_coordination)
                    # to cap the follower's speed and match the leader's pace
                    # instead of continuing to close the gap.
                    _apply_implicit_coop([follower_id], duration=1.0)
            elif inter_state == "collision":
                # Cross-lane / no-lane-data collision: keep prior behaviour —
                # both cars involved get an emergency stop.
                _lane_state.setdefault(a, {})["emergency_stop"] = True
                _lane_state.setdefault(b, {})["emergency_stop"] = True

    # Per-car nested block
    cars_block = {}
    for cid, cd in cars_data.items():
        obs_d = cd["obstacle_info"]["distance"]
        cars_block[str(cid)] = {
            "policy":         cd["policy"],
            "pose":           [round(cd["pose"][0], 2),
                               round(cd["pose"][1], 2),
                               round(cd["pose"][2], 2)],
            "lane":           cd["lane"],
            "segment":        cd["segment"],
            "curvature":      round(cd["curvature"], 4),
            "command":        {"servo": round(cd["servo"], 3),
                               "motor": round(cd["motor"], 3)},
            "waiting":        cd["waiting"],
            "lateral_error":  round(cd["lateral_error"], 2),
            "heading_error":  round(cd["heading_error"], 2),
            "obstacle":       {"driving_state": cd["obstacle_info"]["state"],
                               "distance_px":    round(obs_d, 2)
                                                 if obs_d < float("inf") else None},
            "obstacle_interactions": cd.get("obstacle_interactions", []),
            "events": {
                "minicar_events": cd.get("minicar_events", cd.get("events", [])),
                "safety_events":  {
                    "car":    cd.get("safety_events", {}).get("car", ["no_collision"])
                              if isinstance(cd.get("safety_events"), dict) else ["no_collision"],
                    "object": cd.get("safety_events", {}).get("object", ["no_collision"])
                              if isinstance(cd.get("safety_events"), dict) else cd.get("safety_events", ["no_collision"]),
                },
            },
            "occlusion_tier": cd.get("occlusion_tier", "BOTH_VISIBLE"),
            "emergency_stop": cd.get("emergency_stop", False),
            "maneuvering":    cd.get("maneuvering", False),
        }

    return {"t": round(t, 5), "k": k, "distances": distances, "cars": cars_block}


def _compute_px_per_cm() -> tuple:
    """
    Compute px_per_cm from the two cross-check marker pairs using the cached
    track-marker positions (_last_track_markers).

    Primary baseline  : markers 9 ↔ 10 (longest straight, outer ring).
                        TRACK_LONG_AXIS_CM must be set for a valid result.
    Cross-check axis  : markers 2 ↔ 6  (shortest straight, middle ring).

    Returns
    -------
    px_per_cm           float or None  (None when TRACK_LONG_AXIS_CM == 0)
    long_axis_px        float or None  pixel distance 9↔10
    short_axis_px       float or None  pixel distance 2↔6
    long_axis_cm_comp   float or None  computed long-axis length in cm (= TRACK_LONG_AXIS_CM when primary pair visible)
    short_axis_cm_comp  float or None  computed short-axis length in cm (cross-check)
    """
    def _marker_px(mid):
        entry = _last_track_markers.get(mid)
        if entry is None:
            return None
        corner, _ = entry
        return marker_center(corner)

    p9  = _marker_px(9)
    p10 = _marker_px(10)
    p2  = _marker_px(2)
    p6  = _marker_px(6)

    long_px  = float(np.linalg.norm(np.array(p9,  float) - np.array(p10, float))) if (p9  and p10) else None
    short_px = float(np.linalg.norm(np.array(p2,  float) - np.array(p6,  float))) if (p2  and p6)  else None

    if TRACK_LONG_AXIS_CM <= 0 or long_px is None:
        return None, long_px, short_px, None, None

    px_per_cm        = long_px / TRACK_LONG_AXIS_CM
    long_cm_comp     = TRACK_LONG_AXIS_CM   # by definition (this is the seed)
    short_cm_comp    = round(short_px / px_per_cm, 2) if short_px is not None else None

    return round(px_per_cm, 4), round(long_px, 2), round(short_px, 2) if short_px else None, round(long_cm_comp, 2), short_cm_comp


def save_log(path: str = None) -> None:
    """
    Flush in-memory log to JSON.  Call once at shutdown.

    Filename is auto-generated as:
        exp-log-{scenario}-r{repetition}-{dfov}fov-{height}cm-{calib}-{policy}.json
    where {repetition} is the value of --repetition passed on the CLI
    (default 1).  No counter file is read or written — the caller is
    fully responsible for choosing the repetition number.
    Pass *path* explicitly to override the auto-naming logic entirely.

    Meta block written to the JSON:
        scenario, policy, calibration, dfov, camera_height_cm,
        coordinate_origin, unit,
        px_per_cm, track_long_axis_cm,
        cross_check { long_axis_9_10_px, long_axis_9_10_cm,
                      short_axis_2_6_px, short_axis_2_6_cm },
        saved_at, n_frames

    camera_height_cm (request 4) is populated automatically from the
    detected DFOV tag (_DFOV_HEIGHT_CM lookup in load_calibration) — it is
    only ever non-None when a recognised "90" or "78" DFOV tag was parsed
    from the loaded .npz filename; unrecognised/untagged calibrations log
    camera_height_cm = None rather than guessing.
    """
    calib_tag = "calib" if _CALIBRATED else "non-calib"

    if path is None:
        # ── Structured output path ────────────────────────────────────────────
        # Folder : .exp/{scenario}/  (or .exp/{scenario}/{run_name}/)
        # File   : exp-log-{scenario}-r{repetition}-{dfov}fov-{height}cm-{calib}-{policy}.json
        # e.g.   : .exp/S1/exp-log-S1-r3-90fov-170cm-calib-cooperative.json
        #
        # The repetition index comes entirely from --repetition on the CLI.
        # No persistent counter file is read or written.
        dfov_tag   = f"{_DFOV}fov" if _DFOV else "unknownfov"
        camera_height = _DFOV_HEIGHT_CM.get(_DFOV)
        height_tag = f"-{int(camera_height)}cm" if camera_height else ""
        run_dir  = os.path.join("./exp", LOG_SCENARIO)
        if LOG_RUN_NAME:
            run_dir = os.path.join(run_dir, LOG_RUN_NAME)
        os.makedirs(run_dir, exist_ok=True)
        fname    = (f"exp-log-{LOG_SCENARIO}-r{LOG_REPETITION}"
                    f"-{dfov_tag}{height_tag}-{calib_tag}-{LOG_POLICY}.json")
        path     = os.path.join(run_dir, fname)

    px_per_cm, long_px, short_px, long_cm, short_cm = _compute_px_per_cm()

    payload = {
        "meta": {
            "scenario":          LOG_SCENARIO,
            "policy":            LOG_POLICY,
            "calibration":       calib_tag,
            "dfov":              _DFOV,           # "90", "78", or None
            "camera_height_cm":  camera_height,  # 170.0 / 220.0 / None (request 4)
            "coordinate_origin": "track_centroid",
            "unit":              "cm" if px_per_cm else "px",
            "px_per_cm":         px_per_cm,
            "track_long_axis_cm": TRACK_LONG_AXIS_CM if TRACK_LONG_AXIS_CM > 0 else None,
            "cross_check": {
                # Primary baseline (seeds px_per_cm — should match TRACK_LONG_AXIS_CM)
                "long_axis_9_10_px": long_px,
                "long_axis_9_10_cm": long_cm,
                # Independent cross-check (shortest straight — compare to tape measure)
                "short_axis_2_6_px": short_px,
                "short_axis_2_6_cm": short_cm,
            },
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_frames": len(_log_entries),  # one entry = one camera frame
            "n_exp": 1,  # one log file = one experiment run
        },
        "frames": _log_entries,
        "track_ground_truth": _TRACK_GT_SNAPSHOT if _TRACK_GT_SNAPSHOT else None,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[log] {len(_log_entries)} frames saved → {path}")



def _classify_safety(d: float, d_col: float, d_warn: float, d_safe: float) -> str:
    """Four-way safety bucket reused for both car-car and car-object pairs
    (request 1 + 2), matching the methodology exactly:

    d <= d_col                 -> "collision"
    d_col  < d <= d_warn       -> "near_miss"
    d_warn < d <= d_safe       -> "decision"   (object-avoidance trigger zone)
    d > d_safe                 -> "safe"

    NOTE: call sites must now pass three thresholds in (d_col, d_warn, d_safe)
    order — d_safe is the most permissive/farthest boundary, d_warn is the
    stricter one closer to collision (methodology-aligned naming).
    """
    if d <= d_col:
        return "collision"
    if d <= d_warn:
        return "near_miss"
    if d <= d_safe:
        return "decision"
    return "safe"


def marker_center(corner) -> tuple:
    pts = np.array(corner).reshape(-1, 2)  # handles (1,4,2), (4,2), (4,1,2)
    return tuple(int(v) for v in np.mean(pts, axis=0))

def midpoint(p1: tuple, p2: tuple) -> tuple:
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def wrap_angle_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def heading_to_angle(hv: tuple) -> float:
    return float(np.degrees(np.arctan2(-hv[1], hv[0])) % 360)

def dominant_state(sd: dict) -> str:
    return max(sd.items(), key=lambda kv: kv[1])[0] if sd else "-"


# ══════════════════════════════════════════════════════════════════════════════
# TRACK GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def _sort_stadium_markers(pts: np.ndarray) -> np.ndarray:
    centre = pts.mean(axis=0)
    angles = np.arctan2(-(pts[:,1] - centre[1]), pts[:,0] - centre[0])
    return pts[np.argsort(angles)[::-1]]


def _fit_stadium_params(marker_positions: list):
    pts = np.array(marker_positions, dtype=float)
    if len(pts) >= 4:
        pts = _sort_stadium_markers(pts[:4])
        lc = (pts[0] + pts[3]) / 2.0
        rc = (pts[1] + pts[2]) / 2.0
        cx, cy = float((lc[0] + rc[0]) / 2), float((lc[1] + rc[1]) / 2)
        ax = rc - lc
        angle = float(np.arctan2(ax[1], ax[0]))
        a = float(np.linalg.norm(ax) / 2)
        r = max((np.linalg.norm(pts[0]-pts[3]) + np.linalg.norm(pts[1]-pts[2])) / 4, 1.0)
    else:
        cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())
        _, _, Vt = np.linalg.svd(pts - [cx, cy], full_matrices=False)
        angle    = float(np.arctan2(Vt[0,1], Vt[0,0]))
        proj     = (pts - [cx, cy]) @ Vt.T
        r = max(float(np.mean(np.abs(proj[:,1]))), 1.0)
        a = max(float(np.max(np.abs(proj[:,0]))) - r, 0.0)
    return cx, cy, a, r, angle


def _marker_world_positions(
        inner_pts: list, middle_pts: list, outer_pts: list,
        px_per_cm: float) -> dict:
    """
    Derive world-frame positions (cm, track-centred) for all detected track
    markers using the fitted stadium geometry.  Requires px_per_cm > 0.

    The track centroid is the origin (0, 0).  Axes are aligned with the
    stadium long-axis (u) and short-axis (v).

    Returns {marker_id: [x_cm, y_cm, 0.0], ...} for every marker whose
    pixel position is in the cache.
    """
    if px_per_cm <= 0:
        return {}

    all_pts   = inner_pts + middle_pts + outer_pts
    if len(all_pts) < 3:
        return {}

    all_arr   = np.array(all_pts, dtype=float)
    cx, cy    = float(all_arr[:, 0].mean()), float(all_arr[:, 1].mean())

    # Fit per-ring params for radius lookup
    def _ring_r(pts):
        if len(pts) < 2:
            return 0.0
        arr = np.array(pts, dtype=float)
        return float(np.mean(np.linalg.norm(arr - np.array([cx, cy]), axis=1)))

    # Stadium orientation from all pts
    _, _, Vt = np.linalg.svd(all_arr - np.array([cx, cy]), full_matrices=False)
    angle     = float(np.arctan2(Vt[0, 1], Vt[0, 0]))
    ca, sa    = math.cos(angle), math.sin(angle)
    u_ax      = np.array([ca,  sa])   # long axis
    v_ax      = np.array([-sa, ca])   # short axis

    result = {}
    rings = [
        (INNER_SET,  inner_pts),
        (MIDDLE_SET, middle_pts),
        (OUTER_SET,  outer_pts),
    ]
    for id_set, ring_pts in rings:
        if len(ring_pts) < 1:
            continue
        for mid, (px, py) in zip(sorted(id_set), ring_pts):
            # Project onto track axes
            rel   = np.array([px - cx, py - cy])
            x_cm  = round(float(np.dot(rel, u_ax)) / px_per_cm, 2)
            y_cm  = round(float(np.dot(rel, v_ax)) / px_per_cm, 2)
            result[mid] = [x_cm, y_cm, 0.0]
    return result


def fit_stadium(marker_positions: list, n: int = STADIUM_SAMPLES,
                cx: float = None, cy: float = None,
                forward_hint: np.ndarray = None) -> list:
    """Fit a closed stadium curve through track-marker tangent corners."""
    scx, scy, a, r, angle = _fit_stadium_params(marker_positions)
    if cx is not None: scx = cx
    if cy is not None: scy = cy
    centre = np.array([scx, scy])
    perim_semi  = math.pi * r
    perim_str   = 2.0 * a
    total_perim = 2.0 * perim_semi + 2.0 * perim_str
    n_semi = max(2, (n - 2 * max(1, round(n * perim_str / (total_perim + 1e-9)))) // 2)
    n_str  = max(1, round(n * perim_str / (total_perim + 1e-9)))
    ca, sa = math.cos(angle), math.sin(angle)
    u, v   = np.array([ca, sa]), np.array([-sa, ca])
    pts    = []
    for i in range(n_semi):
        th = -math.pi/2 + math.pi * i / max(n_semi-1, 1)
        pts.append(centre + a*u + r*(math.cos(th)*u + math.sin(th)*v))
    for i in range(1, n_str+1):
        t = i / (n_str+1); pts.append(centre + (1-2*t)*a*u + r*v)
    for i in range(n_semi):
        th = math.pi/2 + math.pi * i / max(n_semi-1, 1)
        pts.append(centre - a*u + r*(math.cos(th)*u + math.sin(th)*v))
    for i in range(1, n_str+1):
        t = i / (n_str+1); pts.append(centre + (2*t-1)*a*u - r*v)
    arr = np.array(pts, dtype=float)
    m   = len(arr)
    if m != n:
        arr = arr[np.round(np.linspace(0, m-1, n)).astype(int)]
    return list(zip(arr[:,0].astype(int).tolist(), arr[:,1].astype(int).tolist()))


def detect_lanes(track_markers: dict) -> dict:
    """Fit inner/middle/outer stadium curves; build lane1 and lane2 centrelines."""
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
    cx, cy  = float(all_arr[:,0].mean()), float(all_arr[:,1].mean())

    for name, pts in [("inner_curve",  inner_pts),
                      ("middle_curve", middle_pts),
                      ("outer_curve",  outer_pts)]:
        if len(pts) >= 2:
            c = fit_stadium(pts, cx=cx, cy=cy, forward_hint=_TRACK_FORWARD)
            result[name] = c[::-1]          # flip to CW image-space order

    def _centre(a_key, b_key, ref_key, ready_key, hw_key):
        if a_key in result and b_key in result:
            a  = np.array(result[a_key], float)
            b  = np.array(result[b_key], float)
            c  = ((a + b) / 2).astype(int)
            result[ref_key]  = list(zip(c[:,0].tolist(), c[:,1].tolist()))
            result[ready_key] = True
            half = float(np.mean(np.linalg.norm(b - a, axis=1)) / 2)
            result[hw_key]   = max(half, LANE_HALF_WIDTH)

    _centre("inner_curve",  "middle_curve", "lane1_ref", "lane1_ready", "lane1_halfwidth")
    _centre("middle_curve", "outer_curve",  "lane2_ref", "lane2_ready", "lane2_halfwidth")

    # ── "Reference truth" snapshot (request 3) ───────────────────────────
    # Previously captured ONCE (first successful detect_lanes call) and then
    # frozen for the rest of the run. Now refreshed EVERY frame for BOTH
    # lanes, so track_ground_truth in the saved log always reflects the
    # most recently fitted centreline geometry rather than a single
    # snapshot from start-up (when marker visibility/jitter may have been
    # worse). This is intentionally lightweight — it just re-stores the
    # already-computed lane1_ref/lane2_ref lists, no extra computation.
    #
    # LIMITATION (documented, not solved here): this is still a
    # self-referential reference. lane1_ref / lane2_ref are themselves
    # derived from the same ArUco/camera pipeline being evaluated, so this
    # is a *running estimate of track geometry consistency*, not an
    # independent ground truth in the strict methodology sense. A true
    # independent reference would require an external validation source
    # (e.g. manually placed, tape-measured markers compared frame-by-frame
    # against the vision-estimated pose/distance) — see Limitations.
    global _TRACK_GT_SNAPSHOT, _extrinsics_done
    _snap = {}
    for key in ("lane1_ref", "lane2_ref"):
        if key in result:
            _snap[key] = result[key]
    if _snap:
        _TRACK_GT_SNAPSHOT = _snap

    # ── Auto-estimate extrinsics once all rings visible ──────────────────
    # Uses all available markers from the fitted stadium geometry +
    # TRACK_LONG_AXIS_CM.  No external JSON file needed.  rvec/tvec are
    # persisted back into the .npz so future sessions load them instantly.
    if _CALIBRATED and not _extrinsics_done and TRACK_LONG_AXIS_CM > 0:
        _px_cm, *_ = _compute_px_per_cm()
        if _px_cm and _px_cm > 0:
            _wp = _marker_world_positions(inner_pts, middle_pts, outer_pts, _px_cm)
            if len(_wp) >= 4:
                _w_arr = np.array(list(_wp.values()), dtype=np.float64)
                _i_arr = np.array(
                    [marker_center(_last_track_markers[mid][0])
                     for mid in _wp if mid in _last_track_markers],
                    dtype=np.float64)
                if len(_i_arr) == len(_w_arr):
                    if estimate_extrinsics(_w_arr, _i_arr):
                        # Persist back to the file that was loaded (--calib-file),
                        # or fall back to the default output path.
                        _cf = _CALIB_PATH or CALIB_DEFAULT_OUT
                        with _calib_lock:
                            _rv, _tv = _CAM_RVEC.copy(), _CAM_TVEC.copy()
                        save_calibration(_cf, _CAM_K, _CAM_D, _rv, _tv)

    return result


def get_track_direction(track_markers: dict) -> np.ndarray:
    pts = [marker_center(track_markers[i]) for i in MIDDLE_SET if i in track_markers]
    if len(pts) < 4:
        return np.array([1.0, 0.0])
    centre = np.mean(pts, axis=0)
    angles = np.arctan2(-np.array([p[1] for p in pts]) + centre[1],
                         np.array([p[0] for p in pts]) - centre[0])
    sp = [p for _, p in sorted(zip(angles, pts), reverse=True)]
    fwd = np.array(sp[1]) - np.array(sp[0])
    return fwd / (np.linalg.norm(fwd) + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# MARKER DETECTION & POSE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_all_markers(frame):
    """
    Detect all ArUco marker types. Returns (car_markers, track_markers, obs_markers).

    Parameters
    ----------
    frame       : raw (undecorated) frame — used for detection only.

    Drawing
    ───────
    This function performs NO drawing. It only detects markers and caches
    the raw (corners, ids) debug geometry for each dictionary in the module
    global `_last_marker_debug`, which the display thread (player_launcher.py)
    reads once per frame via draw_marker_debug() to render the ArUco debug
    rectangles exactly once on the shared canvas — instead of once per
    connected car, which the old draw_target-based approach did implicitly
    since run() called this once per car per frame.

    Caching policy
    ──────────────
    Static (track, obstacle): live detections immediately overwrite cache;
      occluded markers return the last valid cache entry — no reconstruction.
    Car markers: returned live only; fallback is handled in identify_cars.
    """
    global _last_track_markers, _last_obs_positions, _last_marker_debug
    now  = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = aruco.DetectorParameters()

    def _detect(dictionary):
        det = aruco.ArucoDetector(aruco.getPredefinedDictionary(dictionary), params)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            return {}, [], None
        return {int(i): corners[k] for k, i in enumerate(ids.flatten())}, corners, ids

    car_markers, car_corners, car_ids = _detect(CAR_DICT)

    live_track, track_corners, track_ids = _detect(TRACK_DICT)
    for mid, corner in live_track.items():
        _last_track_markers[mid] = (corner, now)   # always refresh with live
    track_markers = dict(live_track)
    for mid, (corner, _t) in list(_last_track_markers.items()):
        if mid not in track_markers:
            track_markers[mid] = corner

    live_obs, obs_corners, obs_ids = _detect(OBSTACLE_DICT)
    for mid, corner in live_obs.items():
        _last_obs_positions[mid] = (marker_center(corner), now)
    obs_markers = dict(live_obs)
    _ghost_obs = []   # cached-only obstacle positions — drawn as circles, not squares
    for mid, (pos, _t) in list(_last_obs_positions.items()):
        if mid not in obs_markers:
            px, py = pos
            fake   = np.array([[px-6,py-6],[px+6,py-6],[px+6,py+6],[px-6,py+6]],
                               dtype=np.float32)
            obs_markers[mid] = fake
            _ghost_obs.append(pos)

    # Cache everything the display thread needs to redraw the same debug
    # rectangles/labels that aruco.drawDetectedMarkers used to draw directly.
    _last_marker_debug = {
        "car":   (car_corners,   car_ids,   None),
        "track": (track_corners, track_ids, None),
        "obs":   (obs_corners,   obs_ids,   (0, 0, 255)),
        "ghost_obs": _ghost_obs,
    }
    return car_markers, track_markers, obs_markers


def draw_marker_debug(canvas):
    """Draw cached ArUco debug rectangles/labels from the last detect_all_markers() call.

    Call this ONCE per display frame (like draw_lanes/draw_obstacles) — not
    once per car — since detect_all_markers() runs once per car per frame
    but always against the same physical camera frame, so the geometry is
    identical across cars within one frame.
    """
    dbg = _last_marker_debug
    if not dbg:
        return
    for key in ("car", "track", "obs"):
        corners, ids, color = dbg.get(key, (None, None, None))
        if not corners or ids is None:
            continue
        if color is None:
            aruco.drawDetectedMarkers(canvas, corners, ids)
        else:
            aruco.drawDetectedMarkers(canvas, corners, ids, color)
    for pos in dbg.get("ghost_obs", []):
        cv2.circle(canvas, pos, 6, (0, 160, 160), 2)
        cv2.putText(canvas, "OBS", (pos[0] + 8, pos[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 160, 160), 1)



def identify_cars(car_markers: dict, car_id: int) -> dict:
    """
    Marker Occlusion Tiers.
    raw_rear = marker_center() = centroid of the 4 ArUco corners.
    This is the true physical centre of the marker square (not a corner vertex,
    not the midpoint between two markers). Every computation — EKF fuse, control
    reference, heading — uses this centroid.

    Marker pairs (from CAR_MARKER_PAIRS):
        car 0: front=49, rear=0  | car 1: front=46, rear=1  | car 2: front=47, rear=2
    Each car's front/rear markers are fully independent — no shared marker ID exists
    between cars, so identify_cars resolves ownership directly via the pair table.

    BOTH_VISIBLE  -> raw_rear = centroid of rear marker; heading = front_c - rear_c.
    FRONT_ONLY    -> raw_rear = centroid of front marker; heading from cached rear/EKF.
    REAR_ONLY     -> raw_rear = centroid of rear marker; heading from EKF.
    BOTH_OCCLUDED -> car_id absent; run() applies velocity-damped EKF prediction.
    """
    global _car_marker_cache, _car_occlusion_state
    if car_id not in _car_marker_cache:
        _car_marker_cache[car_id] = {
            "front_corner": None, "rear_corner": None,
            "front_time":   None, "rear_time":   None,
        }
    if car_id not in _car_occlusion_state:
        _car_occlusion_state[car_id] = {"tier": "BOTH_VISIBLE",
                                         "frames_occluded": 0,
                                         "prev_tier": "BOTH_VISIBLE"}

    cache  = _car_marker_cache[car_id]
    ostate = _car_occlusion_state[car_id]
    now    = time.time()

    # Resolve this car's front/rear marker IDs from the pair table.
    # Falls back to legacy FRONT_MARKER_ID=49 / car_id for unknown car IDs.
    _pair       = CAR_MARKER_PAIRS.get(car_id, (FRONT_MARKER_ID, car_id))
    fmid        = _pair[0]   # front marker ID for this car
    rmid        = _pair[1]   # rear  marker ID for this car (== car_id by convention)

    # Determine live visibility from the detected marker set.
    # Because every car now has its own unique front marker ID, ownership
    # is direct (no proximity competition needed).  The FRONT_OWN_MAX_DIST_PX
    # guard is still applied to catch stray detections.
    rear_live  = rmid in car_markers
    front_live = False
    if fmid in car_markers:
        if rear_live:
            # Both markers visible — verify front belongs to this car via
            # proximity: must be within FRONT_OWN_MAX_DIST_PX of its own rear.
            fc_pos      = marker_center(car_markers[fmid])
            my_rear_pos = marker_center(car_markers[rmid])
            my_dist     = math.hypot(fc_pos[0] - my_rear_pos[0],
                                     fc_pos[1] - my_rear_pos[1])
            front_live  = my_dist <= FRONT_OWN_MAX_DIST_PX
        else:
            # Rear not visible this frame but we have a cached rear position.
            # Anchor the front against the cached rear to verify ownership —
            # prevents a stray front marker from being claimed incorrectly.
            # If no cache yet, allow it: first-ever detection, only marker seen.
            cached_rear = cache.get("rear_corner")
            if cached_rear is not None:
                fc_pos      = marker_center(car_markers[fmid])
                cr_pos      = marker_center(cached_rear)
                cached_dist = math.hypot(fc_pos[0] - cr_pos[0], fc_pos[1] - cr_pos[1])
                # Use a looser distance bound for cached rear (car may have
                # moved a few frames since the cache was written).
                front_live = cached_dist <= FRONT_OWN_MAX_DIST_PX * 1.5
            else:
                # No cache at all — accept the detection (first-ever sighting).
                front_live = True

    if front_live:
        cache["front_corner"] = car_markers[fmid]
        cache["front_time"]   = now
    if rear_live:
        cache["rear_corner"]  = car_markers[rmid]
        cache["rear_time"]    = now

    if front_live and rear_live:
        tier = "BOTH_VISIBLE"
    elif front_live and not rear_live:
        tier = "FRONT_ONLY"
    elif rear_live and not front_live:
        tier = "REAR_ONLY"
    else:
        tier = "BOTH_OCCLUDED"

    prev_tier = ostate["tier"]
    ostate["frames_occluded"] = ostate["frames_occluded"] + 1 if tier == "BOTH_OCCLUDED" else 0
    ostate["tier"]      = tier
    ostate["prev_tier"] = prev_tier

    # Collect all marker IDs that are already claimed as front/rear by any known car.
    # These must not be added as spurious "unknown car" entries in the cars dict.
    _all_claimed_ids: set = set()
    for _pid_pair in CAR_MARKER_PAIRS.values():
        _all_claimed_ids.add(_pid_pair[0])   # front
        _all_claimed_ids.add(_pid_pair[1])   # rear

    # ── Inactive-car markers → treat as static obstacles ──────────────────
    # When a known car is NOT connected (not in ACTIVE_CAR_IDS) and its
    # markers are visible, the car is parked / manually placed on the track
    # (or an unregistered car such as ID=0 sitting across both lanes).
    # BOTH visible markers (front AND rear) are collected here and fed back
    # to run() via _inactive_car_obstacles, which run() merges into
    # _obstacle_positions right alongside the live 6x6 obstacle detections —
    # i.e. the SAME obstacle pipeline, same lane-assignment, same
    # _both_lanes_blocked()/emergency-stop logic, same draw_obstacles() call.
    # This also handles the straddle case: if the parked/unregistered car
    # sits across the lane boundary, each marker lands in a different lane,
    # giving _both_lanes_blocked() something concrete to act on.
    # Active (connected) cars are never injected here — their markers are
    # processed as usual through the tier system below.
    #
    # NOTE: this list is rebuilt from scratch every call (not appended to
    # forever) so a marker that goes out of view is correctly dropped, and
    # run() is the single place that owns merging it into _obstacle_positions
    # (identify_cars must never write _obstacle_positions directly — run()
    # rebuilds that list from obs_markers every frame and would otherwise
    # silently wipe out anything written here).
    global _inactive_car_obstacles
    _new_inactive_obs = []
    for _oc_id, _oc_pair in CAR_MARKER_PAIRS.items():
        if _oc_id in ACTIVE_CAR_IDS:
            continue   # this car is active — handle normally
        for _oc_mid in _oc_pair:   # check both front and rear marker IDs
            if _oc_mid in car_markers:
                _pos = marker_center(car_markers[_oc_mid])
                if not any(_pos == _p for _, _p in _new_inactive_obs):
                    _new_inactive_obs.append((_oc_mid, _pos))   # keep marker_id alongside pos
    _inactive_car_obstacles = _new_inactive_obs

    cars = {}
    for mid, corner in car_markers.items():
        if mid in _all_claimed_ids:
            continue
        rc = marker_center(corner)   # centroid of 4 corners
        cars[mid] = {"front": None, "rear": rc, "midpoint": rc, "raw_rear": rc,
                     "heading": (0, 0), "using_front_fallback": False,
                     "occlusion_tier": "BOTH_VISIBLE"}

    if tier == "BOTH_VISIBLE":
        fc = marker_center(cache["front_corner"])   # centroid of front marker
        rc = marker_center(cache["rear_corner"])    # centroid of rear marker
        cars[car_id] = {
            "front": fc, "rear": rc, "midpoint": rc, "raw_rear": rc,
            "heading": (fc[0] - rc[0], fc[1] - rc[1]),
            "using_front_fallback": False,
            "occlusion_tier": "BOTH_VISIBLE",
        }
    elif tier == "FRONT_ONLY":
        fc = marker_center(cache["front_corner"])
        if cache["rear_corner"] is not None:
            rc_c = marker_center(cache["rear_corner"])
            dx, dy = fc[0] - rc_c[0], fc[1] - rc_c[1]
            nh = math.hypot(dx, dy)
            hdg = (int(dx / nh * WHEELBASE_PX), int(dy / nh * WHEELBASE_PX)) if nh > 1e-4 else (0, 0)
        else:
            hdg = (0, 0)
        cars[car_id] = {
            "front": fc, "rear": fc, "midpoint": fc, "raw_rear": fc,
            "heading": hdg, "using_front_fallback": True,
            "occlusion_tier": "FRONT_ONLY",
        }
    elif tier == "REAR_ONLY":
        rc = marker_center(cache["rear_corner"])    # centroid of rear marker
        hdg = (0, 0)
        cars[car_id] = {
            "front": None, "rear": rc, "midpoint": rc, "raw_rear": rc,
            "heading": hdg, "using_front_fallback": False,
            "occlusion_tier": "REAR_ONLY",
        }
    # BOTH_OCCLUDED: car_id absent; run() handles velocity-damped prediction.
    return cars


def aruco_heading_rad(car: dict) -> float:
    """Convert the car's pixel heading vector to radians (math convention, y-up)."""
    hx, hy = float(car["heading"][0]), float(car["heading"][1])
    return math.atan2(-hy, hx)


# ══════════════════════════════════════════════════════════════════════════════
# PATH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def cross2d(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def project_onto_curve(pos: tuple, curve_pts: list) -> int:
    arr = np.array(curve_pts, dtype=float)
    return int(np.argmin(np.linalg.norm(arr - np.array(pos, dtype=float), axis=1)))


def local_path_frame(curve_pts: list, idx: int):
    """Return (p_curr, unit_tangent, unit_left_normal) at curve index idx."""
    n      = len(curve_pts)
    p_prev = np.array(curve_pts[(idx-1) % n], dtype=float)
    p_curr = np.array(curve_pts[idx],          dtype=float)
    p_next = np.array(curve_pts[(idx+1) % n], dtype=float)
    tan    = p_next - p_prev
    nt     = np.linalg.norm(tan)
    tan    = tan / nt if nt >= 1e-6 else np.array([1.0, 0.0])
    return p_curr, tan, np.array([tan[1], -tan[0]])


def compute_lane_measurements(car_pos: tuple, heading_vec: tuple,
                               curve_pts: list) -> dict:
    """
    Project car onto the lane reference and return tracking errors.

    Methodology §Pose tracking error:
      lateral_error  = signed perpendicular distance to nearest curve point (px)
      heading_error  = signed angular difference between path tangent and heading (°)
    """
    idx           = project_onto_curve(car_pos, curve_pts)
    p_ref, tan, ln = local_path_frame(curve_pts, idx)
    lat_err       = float(np.dot(np.array(car_pos, float) - p_ref, ln))
    tan_angle     = float(np.degrees(np.arctan2(-tan[1], tan[0])) % 360)
    if np.linalg.norm(np.array(heading_vec, float)) < 1e-6:
        car_angle = tan_angle
    else:
        car_angle = heading_to_angle(heading_vec)
    raw_err = wrap_angle_deg(tan_angle - car_angle)
    # The ref curve may be stored CW in image-space while the car travels CCW
    # (or vice-versa), making the stored tangent point opposite to travel direction.
    # pure_pursuit detects this and flips its internal step; we mirror that here
    # so heading_error is always the small residual (−90° … +90°) relative to
    # the actual direction of travel, never the ~±180° artefact.
    if abs(raw_err) > 90.0:
        tan_angle = (tan_angle + 180.0) % 360.0
        raw_err   = wrap_angle_deg(tan_angle - car_angle)
    return {
        "idx":               idx,
        "path_point":        tuple(p_ref.astype(int)),
        "tangent":           tan,
        "left_normal":       ln,
        "lateral_error":     lat_err,
        "heading_error":     raw_err,
        "tangent_angle":     tan_angle,
        "car_heading_angle": car_angle,
    }


def _local_curvature(curve_pts: list, idx: int) -> float:
    """Cross-product curvature estimate at idx using a ±5-step stencil.

    The wider stencil (~20 px at N=300) averages out 1–2 px digitisation
    noise in the fitted curve, preventing spurious light_curve readings on
    what are physically straight sections near bend entries.
    """
    n  = len(curve_pts)
    p0 = np.array(curve_pts[(idx-5) % n], float)
    p1 = np.array(curve_pts[idx],          float)
    p2 = np.array(curve_pts[(idx+5) % n], float)
    v1, v2 = p1-p0, p2-p1
    seg = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2 + 1e-9
    return abs(float(cross2d(v1, v2))) / seg**2


def _classify_segment(car_pos, lanes, lane: int = 1,
                      ref_idx: int = None) -> tuple:
    """Classify the track segment at the car's projected reference point.

    If ref_idx is provided (e.g. already computed by compute_lane_measurements)
    it is used directly, avoiding a redundant projection call and ensuring
    both heading_error and segment type are anchored to the same ref point.
    """
    ref = lanes.get(f"lane{lane}_ref", [])
    if not ref or len(ref) < 5: return "straight", 0.0
    idx = ref_idx if ref_idx is not None else project_onto_curve(car_pos, ref)
    kappa = _local_curvature(ref, idx)
    if kappa > 0.3:  return "sharp_curve", round(kappa, 3)
    if kappa > 0.12: return "light_curve", round(kappa, 3)
    return "straight", round(kappa, 3)


def _curve_speed(car_pos: tuple, lanes: dict, lane: int = 1) -> float:
    ref = lanes.get(f"lane{lane}_ref", [])
    if not ref or len(ref) < 5: return CRUISE_SPEED
    kappa = _local_curvature(ref, project_onto_curve(car_pos, ref))
    if kappa > 0.3:  return SLOW_SPEED
    if kappa > 0.12: return CRUISE_SPEED * 0.85
    return CRUISE_SPEED


# ══════════════════════════════════════════════════════════════════════════════
# PURE-PURSUIT CONTROLLER (uses raw marker-centroid pose, no EKF)
# ══════════════════════════════════════════════════════════════════════════════

# def compute_lookahead(car_midpoint: tuple, heading_vec: tuple,
#     """
#     Pure-pursuit steering from EKF-corrected pose.

#     Returns (raw_delta_rad, cross_track_error_px, path_curvature, lookahead_point_px)

#     Steps
#     ─────
#     1. Nearest curve index ni.
#     2. Winding-direction guard (flip ni if path stored backwards).
#     3. Forward-snap ni until the candidate is in the vehicle's forward half-plane.
#     4. Arc-walk K_DD px ahead to produce the lookahead target T.
#     5. Compute α = bearing(T) − heading; apply δ = atan2(2L·sin α, ld).
#     6. Optional outer-boundary push correction.
#     """
#     arr = np.array(lane_ref, dtype=float)
#     pos = np.array(car_midpoint, dtype=float)
#     n   = len(lane_ref)

#     # 1. Nearest index
#     ni = int(np.argmin(np.linalg.norm(arr - pos, axis=1)))

#     # 2. Path curvature at ni (for segment tagging)
#     path_curvature = _local_curvature(lane_ref, ni)

#     # 3. Winding-direction guard
#     if _TRACK_FORWARD is not None:
#         tan_ni = (arr[(ni+1)%n] - arr[(ni-1)%n]).astype(float)
#         if np.dot(tan_ni, _TRACK_FORWARD) < 0:
#             ni = (ni + n//2) % n

#     # 3b. Forward-snap: advance ni until arr[ni] is in the forward half-plane
#     hx, hy       = float(heading_vec[0]), float(heading_vec[1])
#     heading_norm = math.hypot(hx, hy)
#     heading_known = heading_norm >= 1e-4

#     if heading_known:
#         uh = np.array([hx / heading_norm, hy / heading_norm])
#         # Cap snap to ~K_DD worth of curve steps so we never skip a whole corner
#         _snap_limit = max(1, int(n * K_DD / max(
#             float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))), 1.0)))
#         for _ in range(_snap_limit):
#             if np.dot(arr[ni] - pos, uh) >= 0.0:
#                 break
#             ni = (ni + 1) % n
            
#     # 4. Adaptive lookahead distance.
#     #    Problem diagnosed from log data: _local_curvature returns 0.0 on ~29 % of
#     #    frames mid-corner (stencil lands on a transition sample), which inflates
#     #    _ld back to K_DD and throws the target point past the bend (image 2).
#     #    Fix: take the MAX curvature over the next _kwin steps so a single
#     #    zero-kappa sample cannot reset the distance to full.
#     #    Coefficient 3.0 (was 8.0) keeps _ld >= WHEELBASE_PX for kappa < 0.10,
#     #    and the hard clamp guarantees pure-pursuit geometry stays valid everywhere.
#     _kwin = 12  # ≈ one wheelbase of arc at 300-sample resolution
#     path_curvature = max(
#         _local_curvature(lane_ref, (ni + s) % n) for s in range(_kwin)
#     )
#     _ld = float(np.clip(K_DD / (1.0 + 3.0 * path_curvature),
#                         WHEELBASE_PX, float(K_DD)))

#     # Arc-walk from ni — shorten lookahead on tight curves so the target
#     # stays within the current bend rather than overshooting to the far straight
#     # _ld = K_DD / (1.0 + 8.0 * path_curvature)  # adaptive: shorter on curves
#     accum  = 0.0
#     target = arr[ni].copy()
#     for step in range(1, n):
#         i0  = (ni + step - 1) % n
#         i1  = (ni + step)     % n
#         seg = float(np.linalg.norm(arr[i1] - arr[i0]))
#         if accum + seg >= _ld:
#             frac   = (_ld - accum) / (seg + 1e-9)
#             target = arr[i0] + frac * (arr[i1] - arr[i0])
#             break
#         accum += seg

#     # 5. Cross-track error
#     _, _, ln_ni = local_path_frame(lane_ref, ni)
#     cte = float(np.dot(pos - arr[ni], ln_ni))

#     if not heading_known:
#         k_fb  = 2.0 * cte / (_ld**2 + 1e-9)
#         delta = math.atan(k_fb * WHEELBASE_PX)
#         delta = float(np.clip(delta, -_RAD_TO_SERVO, _RAD_TO_SERVO))
#         return delta, cte, path_curvature, tuple(target.astype(int))

#     # α — signed angle from heading to lookahead chord
#     theta   = math.atan2(-hy, hx)
#     dx, dy  = target[0] - pos[0], target[1] - pos[1]
#     bearing = math.atan2(-dy, dx)
#     alpha   = (bearing - theta + math.pi) % (2*math.pi) - math.pi
#     ld      = math.hypot(dx, dy) + 1e-9
#     delta   = math.atan2(2.0 * WHEELBASE_PX * math.sin(alpha), ld)

#     # 6. Outer-boundary push
#         oi  = int(np.argmin(np.linalg.norm(oa - pos, axis=1)))
#         ot /= np.linalg.norm(ot) + 1e-9
#         overshoot = float(np.dot(pos - oa[oi], np.array([ot[1], -ot[0]])))
#         if overshoot > 0:
#             delta -= K_BOUNDARY_PUSH * overshoot

#     delta = float(np.clip(delta, -_RAD_TO_SERVO, _RAD_TO_SERVO))
#     return delta, cte, path_curvature, tuple(target.astype(int))

def compute_lookahead(car_midpoint, heading_vec, lane_ref, car_id: int = -1):
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
    cross_prod    = abs(float(cross2d(v1, v2)))
    chord_len     = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + 1e-9
    path_curvature = cross_prod / (chord_len ** 2)

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

    global _car_fwd_step
    tangent_fwd = _path_tangent(ni, +1)

    if heading_known:
        ref_dir = np.array([hx / heading_norm, hy / heading_norm])
    elif _TRACK_FORWARD is not None:
        ref_dir = np.array(_TRACK_FORWARD, dtype=float)
        norm_tf = np.linalg.norm(ref_dir)
        ref_dir = ref_dir / norm_tf if norm_tf > 1e-6 else np.array([1.0, 0.0])
    else:
        ref_dir = tangent_fwd
    candidate_step = -1 if np.dot(tangent_fwd, ref_dir) < 0.0 else 1
    if car_id not in _car_fwd_step:
        _car_fwd_step[car_id] = candidate_step
    elif heading_known and heading_norm > 0.5:
        stored = _car_fwd_step[car_id]
        if candidate_step != stored:
            if np.dot(_path_tangent(ni, stored), ref_dir) < -0.5:
                _car_fwd_step[car_id] = candidate_step
    fwd_step = _car_fwd_step.get(car_id, candidate_step)


    # Fixed lookahead distance: use K_DD on all segment types.
    # Adaptive scaling (÷ curvature) caused the yellow dot to collapse to
    # 70–103 px on corners — less than WHEELBASE_PX — breaking pure-pursuit
    # geometry and making the target appear right in front of / under the car.
    # The path-tangent forward-snap (Step 3 below) already ensures the target
    # stays on the upcoming arc rather than leaping to the far straight, so
    # no additional distance reduction is needed on curves.
    # lookahead_dist = float(K_DD)
    # Adaptive lookahead: longer on straights (less sensitive to lateral noise
    # from ArUco jitter), shorter on curves (target stays within the bend).
    if path_curvature < 0.08:          # straight
        _kdd_target = K_DD * 1.5       # ~420 px (~3 car-lengths)
    elif path_curvature < 0.20:        # light curve
        _kdd_target = float(K_DD)      # 280 px (original K_DD)
    else:                              # sharp curve
        _kdd_target = K_DD * 0.75      # ~210 px (tighter – stays in bend)
    lookahead_dist = float(np.clip(_kdd_target, 1.5 * WHEELBASE_PX, K_DD * 1.5))

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
    theta   = math.atan2(hy, hx)                       # heading angle (math coords) change later if crash
    dx, dy  = float(target[0] - pos[0]), float(target[1] - pos[1])
    bearing = math.atan2(dy, dx)                        # chord bearing (math coords)
    alpha   = (bearing - theta + math.pi) % (2.0 * math.pi) - math.pi  # ∈ (−π, +π]

    # ── Pure-pursuit formula ──────────────────────────────────────────────────
    #   δ = atan2(2 · L · sin α,  l_d)
    l_d   = math.hypot(dx, dy) + 1e-9
    delta = math.atan2(2.0 * WHEELBASE_PX * math.sin(alpha), l_d)

    # print(f"alpha: {math.degrees(alpha):.2f} deg, delta: {math.degrees(delta):.2f} deg\n")
    delta = float(np.clip(delta, -(math.pi / 4), +(math.pi / 4)))
    # print(f"delta (post-clip): {math.degrees(delta):.2f} deg\n")
    return delta, cte, path_curvature, tuple(target.astype(int))


# ══════════════════════════════════════════════════════════════════════════════
# PID LONGITUDINAL CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

def pid_speed(car_id: int, v_ref: float, now: float) -> float:
    """
    PI speed controller for longitudinal motor command.

    v_ref  : desired speed in normalised motor units [0, MAX_SPEED]
    v_est  : raw-pose velocity estimate (px/s, from _cached_velocity()),
             normalised to [0, MAX_SPEED] via WHEELBASE_PX. No EKF/filter
             stage — this is the direct finite-difference measurement
             computed once per frame in run() via _update_velocity_cache().

    Returns a continuous motor command in [STOP_SPEED, MAX_SPEED], with
    an additional EMA smoothing pass (_MOTOR_ALPHA) applied to the final
    command to damp abrupt changes/oscillations — mirrors _smooth_servo().
    Anti-windup: integral clamped to ±_PI_I_CLAMP.
    The I-term is reset to 0 when v_ref drops to STOP_SPEED (braking event).
    """
    if car_id not in _spd_pid:
        _spd_pid[car_id] = {
            "i": 0.0, "e_prev": 0.0, "t_prev": now,
            "pos_prev": None, "v_prev": 0.0, "cmd_prev": None,
        }

    s  = _spd_pid[car_id]
    dt = max(now - s["t_prev"], 1e-4)

    # Velocity estimate from raw-pose displacement, normalised to motor units
    v_est_raw = _cached_velocity(car_id)
    v_est     = float(np.clip(v_est_raw / max(WHEELBASE_PX, 1.0), 0.0, MAX_SPEED))

    e = v_ref - v_est

    # Reset integrator on full stop command to avoid windup during braking
    if v_ref <= STOP_SPEED:
        s["i"] = 0.0
    else:
        s["i"] = float(np.clip(s["i"] + e * dt, -_PI_I_CLAMP, _PI_I_CLAMP))

    s["e_prev"] = e
    s["t_prev"] = now

    cmd = _PI_KP * e + _PI_KI * s["i"]
    raw_cmd = float(np.clip(v_ref + cmd, MIN_SPEED, MAX_SPEED))

    # EMA smoothing on the final motor command — reduces abrupt speed
    # changes/oscillations, same pattern as _smooth_servo()/SERVO_ALPHA.
    prev_cmd = s.get("cmd_prev")
    out_cmd  = raw_cmd if prev_cmd is None else (
        _MOTOR_ALPHA * raw_cmd + (1.0 - _MOTOR_ALPHA) * prev_cmd)
    s["cmd_prev"] = out_cmd
    return out_cmd


def cte_speed_penalty(v_ref: float, cte_px: float) -> float:
    """
    Reduce speed proportionally when lateral (cross-track) error is large.

    Rationale: a car with large CTE is mid-correction; slowing it down
    prevents overshoot and oscillation, especially on curves (S1) and
    during car-following (S2).  The penalty is additive on top of any
    coordination-derived setpoint.

    Penalty = min(_CTE_MAX_PENALTY, max(0, |cte| − threshold) × K)
    """
    excess  = max(0.0, abs(cte_px) - _CTE_THRESHOLD)
    penalty = min(excess * _CTE_PENALTY_K, _CTE_MAX_PENALTY)
    return float(max(MIN_SPEED, v_ref - penalty * MAX_SPEED))

# ══════════════════════════════════════════════════════════════════════════════
# LANE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _classify_lane(front_pos, rear_pos, lanes: dict, current_lane: int) -> int:
    """Lane classification from marker positions (either or both may be None).

    Geometry (CW image-space, y-down):
      lane1_ref = inner..middle centreline
      lane2_ref = middle..outer centreline

    local_path_frame left_normal points toward the INNER ring on a CW path.
      signed > +hyst  -> inner side  -> lane 1
      signed < -hyst  -> outer side  -> lane 2
      |signed| <= hyst -> ambiguous  -> hold current_lane

    Rules:
    - All available (non-None) readings agree on the same lane -> return it.
    - Any ambiguity -> hold current_lane (hysteresis).
    - Lane 2 not yet fitted -> always lane 1.
    """
    middle = lanes.get("middle_curve")
    if middle is None or not lanes.get("lane2_ready", False):
        return 1

    hyst = LANE_HYSTERESIS
    arr  = np.array(middle, dtype=float)

    def _signed(pos):
        p   = np.array(pos, dtype=float)
        idx = int(np.argmin(np.linalg.norm(arr - p, axis=1)))
        _, _, ln = local_path_frame(middle, idx)
        return float(np.dot(p - arr[idx], ln))

    def _side(d):
        if d >  hyst: return 1   # inner side (left_normal points inward on CW path)
        if d < -hyst: return 2   # outer side
        return 0                 # ambiguous

    sides = []
    if front_pos is not None:
        sides.append(_side(_signed(front_pos)))
    if rear_pos is not None:
        sides.append(_side(_signed(rear_pos)))

    if not sides:
        return current_lane

    non_zero = [s for s in sides if s != 0]
    if non_zero and all(s == non_zero[0] for s in non_zero):
        return non_zero[0]

    return current_lane


def _obstacle_lane(obs_pos, lanes) -> int:
    """
    Assign a single obstacle position to EXACTLY ONE lane, using the same
    signed-side test already used for car lane classification
    (_car_straddles_lanes._side) against the middle boundary curve.

    This is what actually keeps lane-1 and lane-2 obstacle corridors
    disjoint: previously both lanes tested the SAME shared _obstacle_positions
    list against their own half-width, and because the two lane centrelines
    sit only ~2*LANE_HALF_WIDTH apart, a single obstacle near the middle
    boundary could fall within both lanes' half-width corridors at once —
    causing a single one-lane obstacle to read as "near"/"blocking" in BOTH
    lanes simultaneously (the both_lanes_near_slowdown / emergency-stop bug).
    By pre-assigning each obstacle to one lane before the distance check,
    an obstacle can only ever affect the lane it is physically standing in.
    Two markers of the SAME straddling car (e.g. unregistered car ID=0) will
    naturally resolve to different lanes here, which is exactly the case
    that SHOULD be able to legitimately block both lanes at once.
    """
    middle = lanes.get("middle_curve")
    if middle is None or not lanes.get("lane2_ready", False):
        return 1   # lane 2 not yet fitted — everything is lane 1
    arr = np.array(middle, dtype=float)
    p   = np.array(obs_pos, dtype=float)
    idx = int(np.argmin(np.linalg.norm(arr - p, axis=1)))
    _, _, ln = local_path_frame(middle, idx)
    d = float(np.dot(p - arr[idx], ln))
    return 1 if d >= 0 else 2


def _lane_obstacle_state(car_pos, lanes, lane_num: int) -> str:
    """Plain distance-threshold obstacle state ('clear'/'near'/'blocking')
    for the given lane reference — same three-state rule as car-to-car.

    The lateral corridor used to decide whether an obstacle is "in this
    lane" is the lane's OWN measured half-width (lanes["laneN_halfwidth"],
    computed in detect_lanes from the actual inner/middle/outer ring
    spacing) — not the shared OBSTACLE_TRACK_HALF_WIDTH constant.

    In addition, only obstacles pre-assigned to THIS lane by _obstacle_lane()
    are considered at all, so the two lanes' obstacle sets are always
    disjoint — a single-lane obstacle can never be picked up by both lanes
    just because the half-width corridors geometrically overlap near the
    middle boundary.
    """
    ref = lanes.get(f"lane{lane_num}_ref", [])
    if not ref: return "blocking"   # no lane geometry yet — treat as unsafe, not "free"
    half_width = min(lanes.get(f"lane{lane_num}_halfwidth", LANE_HALF_WIDTH), LANE_HALF_WIDTH)
    idx = project_onto_curve(car_pos, ref)
    _, tan, ln = local_path_frame(ref, idx)
    relevant = [o for o in _obstacle_positions if _obstacle_lane(o, lanes) == lane_num]
    return nearest_relevant_obstacle(car_pos, tan, ln, relevant,
                                      half_width=half_width)["state"]


def _lane_obstacle_free(car_pos, lanes, lane_num: int) -> bool:
    return _lane_obstacle_state(car_pos, lanes, lane_num) == "clear"


def _adjacent_lane_cars(cars, lane_num: int) -> list:
    return [cid for cid in cars if _lane_state.get(cid, {}).get("lane", 1) == lane_num]


def _coop_gap_ok(car_pos, adj_cars, cars) -> bool:
    p0 = np.array(car_pos, dtype=float)
    for cid in adj_cars:
        if cid not in cars: continue
        gap     = float(np.linalg.norm(np.array(cars[cid]["midpoint"], float) - p0))
        closing = _cached_velocity(cid)
        if gap < COOP_MIN_GAP: return False
        if closing > COOP_MAX_CLOSING_SPEED and gap < COOP_MIN_GAP * 1.5: return False
    return True


def _apply_implicit_coop(adj_cars, duration: float = 1.5) -> None:
    expiry = time.time() + duration
    for cid in adj_cars:
        _coop_slowdown_until[cid] = expiry


def _both_lanes_blocked(car_pos, lanes) -> bool:
    """
    Return True ONLY when BOTH lanes are within the hard-stop radius
    (state == "blocking") — the d_safe threshold applied per-lane.
    A single-lane obstacle, or an obstacle only in the "near"/yield band on
    one or both lanes, must NEVER trigger this — those cases are handled by
    the normal overtake / lane-switch decision and the near-band slowdown.
    """
    return (_lane_obstacle_state(car_pos, lanes, 1) == "blocking" and
            _lane_obstacle_state(car_pos, lanes, 2) == "blocking")


def _both_lanes_near(car_pos, lanes) -> bool:
    """True when both lanes are at least in the yield/near band ahead —
    used only to reduce speed early, never to trigger an emergency stop."""
    s1 = _lane_obstacle_state(car_pos, lanes, 1)
    s2 = _lane_obstacle_state(car_pos, lanes, 2)
    return s1 in ("near", "blocking") and s2 in ("near", "blocking")


def _car_straddles_lanes(front_pos, rear_pos, lanes) -> bool:
    """
    Return True when the two ArUco markers of the *same* physical car are
    detected on *different* lanes — i.e. one marker reads lane 1 and the
    other reads lane 2.

    This indicates the car is physically spanning the middle boundary (or a
    detection anomaly that would cause unsafe lane changes).  In both cases
    the correct response is an immediate emergency stop.

    Only meaningful when both front and rear positions are available.
    Returns False if either position is None or lane 2 is not yet fitted.
    """
    if front_pos is None or rear_pos is None:
        return False
    middle = lanes.get("middle_curve")
    if middle is None or not lanes.get("lane2_ready", False):
        return False

    hyst = LANE_HYSTERESIS
    arr  = np.array(middle, dtype=float)

    def _side(pos):
        p   = np.array(pos, dtype=float)
        idx = int(np.argmin(np.linalg.norm(arr - p, axis=1)))
        _, _, ln = local_path_frame(middle, idx)
        d = float(np.dot(p - arr[idx], ln))
        if d > hyst:  return 1
        if d < -hyst: return 2
        return 0   # ambiguous / on the boundary

    s_front = _side(front_pos)
    s_rear  = _side(rear_pos)
    # Straddle = both sides resolved AND they disagree
    return (s_front != 0 and s_rear != 0 and s_front != s_rear)


CAR_CLEARANCE_PX = 90  # px — longitudinal clearance past the obstacle point required
                       # (~1.3 car-lengths) before the rear marker is considered "past it"


def _rear_passed_obstacle(rear_pos, obstacle_point, tangent) -> bool:
    """
    True once the car's REAR marker has physically travelled past the
    obstacle it is overtaking, measured along the lane tangent captured at
    the moment the overtake decision was made.

    This replaces the old purely time-based LANE_CHANGE_HOLD gate for
    returning to the start lane: an overtake should merge back only after
    the rear of the car has cleared the obstacle, not just after a fixed
    number of seconds (which could fire too early for a slow car or too
    late for a fast one).

    Returns True (never blocks) when there isn't enough information to
    judge — i.e. no rear position or no stored obstacle point yet.
    """
    if rear_pos is None or obstacle_point is None or tangent is None:
        return True
    t = np.array(tangent, dtype=float)
    nrm = np.linalg.norm(t)
    if nrm < 1e-9:
        return True
    t = t / nrm
    along = float(np.dot(np.array(rear_pos, dtype=float) -
                          np.array(obstacle_point, dtype=float), t))
    return along > CAR_CLEARANCE_PX


def _decide_lane_cooperative(car_id, car_pos, lanes, obstacle_info, cars, now,
                              rear_pos=None) -> int:
    st = _lane_state[car_id]

    # Lock start_lane on first valid classification (once only).
    if st["start_lane"] is None:
        st["start_lane"] = st["lane"]

    obs_bad = obstacle_info["state"] in ("blocking", "near")

    # Emergency stop: both lanes blocked — nowhere safe to go.
    if _both_lanes_blocked(car_pos, lanes):
        st["emergency_stop"] = True
        return st["lane"]
    st["emergency_stop"] = st.get("emergency_stop", False) and obs_bad

    start  = st["start_lane"]
    adjacent = 2 if st["lane"] == 1 else 1

    # ── Obstacle encounter: make a random decision once per encounter ─────────
    # Decision is always a lane_switch into the adjacent lane first; what
    # differs is whether the car later merges back (overtake) or permanently
    # adopts the new lane (lane_switch). The obstacle point + lane tangent at
    # the moment of the decision are stashed so the return-to-start-lane gate
    # can later test "has the REAR marker physically passed the obstacle yet"
    # instead of relying on a fixed timer.
    if obs_bad and st["obstacle_decision"] is None:
        if lanes.get(f"lane{adjacent}_ready", False):
            adj_cars = _adjacent_lane_cars(cars, adjacent)
            if _lane_obstacle_free(car_pos, lanes, adjacent) and _coop_gap_ok(car_pos, adj_cars, cars):
                _apply_implicit_coop(adj_cars)
                decision = random.choice(["overtake", "lane_switch"])
                st["obstacle_decision"] = decision
                st["lane"], st["timer"], st["overtaking"] = adjacent, now, True
                st["pass_point"]   = obstacle_info.get("point")
                st["pass_tangent"] = obstacle_info.get("tangent")
                if decision == "lane_switch":
                    st["start_lane"] = adjacent   # permanently adopt new lane

    # ── Return to start_lane after overtake (when decision was "overtake") ────
    # Gated on the REAR marker having cleared the obstacle by CAR_CLEARANCE_PX —
    # NOT on a per-frame obs_bad read. obs_bad can glitch to False for a single
    # frame purely from occlusion re-acquisition noise (EKF predict-only pose
    # jump, transient lane misclassification) even though the car has not
    # physically passed the obstacle yet. Resetting obstacle_decision on that
    # transient would let a brand-new random decision fire on the very next
    # frame once obs_bad flips back to True — the observed "reinterprets and
    # keeps changing" behaviour. The decision is now generated exactly ONCE
    # per obstacle encounter and only cleared once the physical pass-by test
    # below confirms the car is actually clear of it.
    # ── Clear a completed lane_switch — checked FIRST and independent of the
    # start_lane comparison below. A lane_switch permanently sets
    # start_lane = adjacent at decision time, so st["lane"] == start
    # immediately on the very next call — the old "st['lane'] != start" guard
    # then NEVER fires again for this car, obstacle_decision is never reset,
    # and every future obstacle encounter is silently ignored (the reported
    # "lane_switch not updating" bug). Testing this branch unconditionally
    # fixes that.
    if st.get("obstacle_decision") == "lane_switch":
        if _rear_passed_obstacle(rear_pos, st.get("pass_point"), st.get("pass_tangent")):
            st["obstacle_decision"] = None
            st["pass_point"] = None
            st["pass_tangent"] = None
    # ── Return to start_lane after overtake (when decision was "overtake") ────
    elif st["overtaking"] and st["lane"] != start:
        if _rear_passed_obstacle(rear_pos, st.get("pass_point"), st.get("pass_tangent")):
            if _lane_obstacle_free(car_pos, lanes, start):
                st["lane"], st["timer"], st["overtaking"] = start, now, False
                st["returned_to_startlane"] = True
                st["obstacle_decision"] = None   # reset ONLY after physically clearing it
                st["pass_point"] = None
                st["pass_tangent"] = None

    return st["lane"]


def _decide_lane_egocentric(car_id, car_pos, lanes, obstacle_info, now,
                             rear_pos=None) -> int:
    st = _lane_state[car_id]

    # Lock start_lane on first valid classification (once only).
    if st["start_lane"] is None:
        st["start_lane"] = st["lane"]

    obs_bad = obstacle_info["state"] in ("blocking", "near")

    # Emergency stop: both lanes blocked — nowhere safe to go.
    if _both_lanes_blocked(car_pos, lanes):
        st["emergency_stop"] = True
        return st["lane"]
    st["emergency_stop"] = st.get("emergency_stop", False) and obs_bad

    start    = st["start_lane"]
    adjacent = 2 if st["lane"] == 1 else 1

    # ── Obstacle encounter: make a random decision once per encounter ─────────
    if obs_bad and st["obstacle_decision"] is None:
        if lanes.get(f"lane{adjacent}_ready", False):
            if _lane_obstacle_free(car_pos, lanes, adjacent):
                decision = random.choice(["overtake", "lane_switch"])
                st["obstacle_decision"] = decision
                st["lane"], st["timer"], st["overtaking"] = adjacent, now, True
                st["pass_point"]   = obstacle_info.get("point")
                st["pass_tangent"] = obstacle_info.get("tangent")
                if decision == "lane_switch":
                    st["start_lane"] = adjacent   # permanently adopt new lane

    # ── Return to start_lane after overtake (when decision was "overtake") ────
    # Gated on the REAR marker having cleared the obstacle by CAR_CLEARANCE_PX —
    # NOT on a per-frame obs_bad read (see _decide_lane_cooperative for the
    # full explanation of why that caused the decision to reinterpret/keep
    # changing after an occlusion event). The decision is generated exactly
    # ONCE per obstacle encounter and only cleared once physically clear of it.
    # ── Clear a completed lane_switch — checked FIRST, independent of the
    # start_lane comparison (see _decide_lane_cooperative for why: start_lane
    # is updated to `adjacent` at decision time, so st["lane"] == start on
    # every subsequent call and the old guard below never re-fired).
    if st.get("obstacle_decision") == "lane_switch":
        if _rear_passed_obstacle(rear_pos, st.get("pass_point"), st.get("pass_tangent")):
            st["obstacle_decision"] = None
            st["pass_point"] = None
            st["pass_tangent"] = None
    elif st["overtaking"] and st["lane"] != start:
        if _rear_passed_obstacle(rear_pos, st.get("pass_point"), st.get("pass_tangent")):
            if _lane_obstacle_free(car_pos, lanes, start):
                st["lane"], st["timer"], st["overtaking"] = start, now, False
                st["returned_to_startlane"] = True
                st["obstacle_decision"] = None
                st["pass_point"] = None
                st["pass_tangent"] = None

    return st["lane"]


def _decide_lane(car_id, car_pos, lanes, obstacle_info, cars, now, rear_pos=None) -> int:
    policy = DRIVING_POLICY.get(car_id, DEFAULT_POLICY)
    if policy == "non_cooperative":
        return _decide_lane_egocentric(car_id, car_pos, lanes, obstacle_info, now, rear_pos=rear_pos)
    return _decide_lane_cooperative(car_id, car_pos, lanes, obstacle_info, cars, now, rear_pos=rear_pos)


# ══════════════════════════════════════════════════════════════════════════════
# OBSTACLE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _classify_obstacle_distance(d: float) -> str:
    """Plain step-function distance rule — no fuzzy blending, no drawn zones.
    Mirrors the car-to-car D_COL / D_WARN / D_SAFE principle exactly.

    RENAMED (methodology alignment, request 2): OBSTACLE_D_WARN is now the
    stricter/smaller hard-stop radius and OBSTACLE_D_SAFE is the larger,
    more permissive yield radius — this function's comparison order was
    updated to match the new constant values (170 vs 70 swapped names,
    same physical thresholds)."""
    if d <= OBSTACLE_D_WARN:
        return "blocking"
    if d <= OBSTACLE_D_SAFE:
        return "near"
    return "clear"


def nearest_relevant_obstacle(car_pos, tangent, left_normal, obstacles,
                               half_width: float = OBSTACLE_TRACK_HALF_WIDTH) -> dict:
    """
    half_width is the lateral corridor (px) considered "in this lane" ahead
    of the car. Callers should pass the lane-specific half-width (see
    _lane_obstacle_state) so an obstacle sitting in one lane is never picked
    up as relevant to the other lane's corridor — the two lanes' obstacle
    corridors must not overlap.
    """
    CLEAR = {"distance": float("inf"), "sideoffset": 0.0, "state": "clear", "point": None}
    if not obstacles: return CLEAR
    t  = tangent     / (np.linalg.norm(tangent)     + 1e-9)
    n  = left_normal / (np.linalg.norm(left_normal) + 1e-9)
    p0 = np.array(car_pos, dtype=float)
    best_d, best = float("inf"), None
    for obs in obstacles:
        vec  = np.array(obs, dtype=float) - p0
        along = float(np.dot(vec, t))
        side  = float(np.dot(vec, n))
        eucl  = float(np.linalg.norm(vec))
        if along <= 0 or along > OBSTACLE_LOOKAHEAD: continue
        if abs(side) > half_width:                   continue
        if eucl < best_d:
            best_d, best = eucl, (obs, side)
    if best is None: return CLEAR
    return {"distance": best_d, "sideoffset": best[1],
            "state": _classify_obstacle_distance(best_d),
            "point": best[0]}


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED COORDINATION
# ══════════════════════════════════════════════════════════════════════════════

def _apply_coordination(cars, base_speed) -> dict:
    """
    Distance-threshold speed arbitration (methodology §Coordination Logic).

    RENAMED (methodology alignment, request 2): D_WARN is now the stricter/
    smaller hard-stop boundary and D_SAFE is the larger/more permissive
    yield boundary (previously inverted vs. the methodology's d_col < d_warn
    < d_safe ordering). The physical stop/slow distances are unchanged —
    only the constant names and this comparison's ordering were swapped.

      d ≤ D_WARN : yielder stops
      d ≤ D_SAFE : yielder slows
      d > D_SAFE : continue
    Cooperative cars yield; non-cooperative cars do not.
    """
    speeds = {cid: base_speed for cid in cars}
    ids    = sorted(cars.keys())
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            pa = np.array(cars[a]["midpoint"], float)
            pb = np.array(cars[b]["midpoint"], float)
            d  = float(np.linalg.norm(pa - pb))
            yd = max(a, b)
            if DRIVING_POLICY.get(yd, DEFAULT_POLICY) == "non_cooperative":
                continue
            if d <= D_WARN:
                speeds[yd] = STOP_SPEED;            _pp_waiting[yd] = True
            elif d <= D_SAFE:
                speeds[yd] = min(speeds[yd], SLOW_SPEED); _pp_waiting[yd] = True
            else:
                _pp_waiting.setdefault(yd, False)
    return speeds


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def draw_lanes(frame, lanes):
    """Render lane fills, boundary polylines, and dashed reference centrelines.

    Lane 1 (inner–middle) tinted green; lane 2 (middle–outer) tinted blue.
    Both fills use a 20% opacity blend so marker overlays stay visible.
    Reference centrelines are dashed (every 8 samples).
    """
    overlay = frame.copy()
    if "inner_curve" in lanes and "middle_curve" in lanes:
        cv2.fillPoly(overlay,
                     [np.vstack([np.array(lanes["inner_curve"],  np.int32),
                                 np.array(lanes["middle_curve"], np.int32)[::-1]])],
                     LANE1_FILL_COLOR)
    if "middle_curve" in lanes and "outer_curve" in lanes:
        cv2.fillPoly(overlay,
                     [np.vstack([np.array(lanes["middle_curve"], np.int32),
                                 np.array(lanes["outer_curve"],  np.int32)[::-1]])],
                     LANE2_FILL_COLOR)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    for key, color in [("inner_curve",  INNER_LINE_COLOR),
                       ("middle_curve", MIDDLE_LINE_COLOR),
                       ("outer_curve",  OUTER_LINE_COLOR)]:
        pts = lanes.get(key, [])
        if pts:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32).reshape(-1,1,2)],
                          True, color, 2, cv2.LINE_AA)

    # Dashed reference centrelines — one tick every 8 samples
    vis_centres = []
    if "inner_curve" in lanes and "middle_curve" in lanes:
        mid = (np.array(lanes["inner_curve"], float) +
               np.array(lanes["middle_curve"], float)) / 2
        vis_centres.append([tuple(p) for p in mid.astype(int).tolist()])
    if "middle_curve" in lanes and "outer_curve" in lanes:
        mid = (np.array(lanes["middle_curve"], float) +
               np.array(lanes["outer_curve"],  float)) / 2
        vis_centres.append([tuple(p) for p in mid.astype(int).tolist()])
    for pts in vis_centres:
        nn = len(pts)
        for i in range(0, nn, 8):
            cv2.line(frame, pts[i], pts[(i + 4) % nn], REF_LINE_COLOR, 1)


def draw_car(frame, car, car_id):
    """Visualise a car.
    Gold dot = raw_rear = marker_center() = centroid of the 4 ArUco corners.
    Never a corner vertex, never the midpoint between two markers.
    Tier colour: cyan=BOTH_VISIBLE/REAR_ONLY, orange=FRONT_ONLY, yellow=BOTH_OCCLUDED.
    """
    rear = car["rear"]
    if rear is None:
        return
    tier = car.get("occlusion_tier", "BOTH_VISIBLE")
    tier_colour = {
        "BOTH_VISIBLE":  (0, 255, 255),
        "FRONT_ONLY":    (0, 165, 255),
        "REAR_ONLY":     (0, 255, 255),
        "BOTH_OCCLUDED": (0, 220, 255),
    }.get(tier, (0, 255, 255))
    cv2.circle(frame, rear, 5, tier_colour, -1)

    front = car.get("front")
    if front and front != rear and car["heading"] != (0, 0):
        cv2.arrowedLine(frame, rear, front, (0, 255, 0), 2, tipLength=0.3)

    # Gold dot: marker_center centroid — pinned to the physical marker square.
    raw_rear = car.get("raw_rear", rear)
    cv2.circle(frame, raw_rear, 4, (255, 255, 255), -1)
    label = str(car_id) + ("F!" if car.get("using_front_fallback") else "")  # car_id == rear marker ID
    cv2.putText(frame, label, (raw_rear[0] + 6, raw_rear[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_guide_line(frame, ref_pose, target, obstacle_close):
    """Draw the pure-pursuit guide line from the car midpoint to the lookahead target.

    Red when an obstacle is blocking/near, white otherwise.
    """
    color = (0, 0, 220) if obstacle_close else REF_LINE_COLOR
    cv2.circle(frame, target, 9, color, 2)
    cv2.circle(frame, target, 3, (255, 255, 255), -1)
    cv2.line(frame, ref_pose, target, (160, 34, 201), 2)

# Interaction-zone colours (BGR) by obstacle state
_IZ_COLOURS = {
    "clear":    (  0, 180,  60),   # green  — path ahead free
    "near":     (  0, 190, 210),   # amber  — obstacle within warn distance
    "blocking": (  0,  40, 220),   # red    — obstacle within safe distance
}

def draw_interaction_zone(frame, car_pos, tangent, left_normal, obs_state: str):
    """
    Draw the obstacle-detection interaction zone as a semi-transparent
    oriented rectangle ahead of the car.

    The zone is OBSTACLE_LOOKAHEAD pixels deep and
    OBSTACLE_TRACK_HALF_WIDTH pixels wide on each side (same geometry used
    by nearest_relevant_obstacle).  Colour reflects the current obstacle state:
        clear    → green
        near     → amber
        blocking → red

    Parameters
    ----------
    car_pos     : (x, y) pixel position of the car reference point
    tangent     : unit forward vector (path tangent, same as used in NRO)
    left_normal : unit left-perpendicular vector
    obs_state   : "clear" | "near" | "blocking"
    """
    t  = np.array(tangent,     dtype=float)
    t  = t  / (np.linalg.norm(t)  + 1e-9)
    n  = np.array(left_normal, dtype=float)
    n  = n  / (np.linalg.norm(n)  + 1e-9)
    p0 = np.array(car_pos, dtype=float)

    # IZ is 50 % deeper and 40 % wider than the bare detection footprint
    # so it reads clearly on screen as a distinct visualisation zone.
    depth = float(OBSTACLE_LOOKAHEAD)        * 1.5
    half  = float(OBSTACLE_TRACK_HALF_WIDTH) * 1.4

    # Four corners of the oriented rectangle
    corners = np.array([
        p0 + n * half,
        p0 + t * depth + n * half,
        p0 + t * depth - n * half,
        p0 - n * half,
    ], dtype=np.int32)

    colour = _IZ_COLOURS.get(obs_state, _IZ_COLOURS["clear"])

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [corners], colour)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # Solid border
    cv2.polylines(frame, [corners], isClosed=True, color=colour, thickness=1,
                  lineType=cv2.LINE_AA)

    # Small arrow showing forward direction
    tip = (p0 + t * depth * 0.55).astype(int)
    cv2.arrowedLine(frame, tuple(p0.astype(int)), tuple(tip),
                    colour, 1, tipLength=0.15, line_type=cv2.LINE_AA)


def draw_obstacles(frame, obstacles):
    """Draw each detected obstacle as a translucent filled circle with an OBS label."""
    for obs in obstacles:
        ox, oy = int(obs[0]), int(obs[1])
        overlay = frame.copy()
        cv2.circle(overlay, (ox, oy), OBSTACLE_RADIUS, (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.circle(frame, (ox, oy), OBSTACLE_RADIUS, (0, 0, 255), 2)
        cv2.circle(frame, (ox, oy), 6,                (0, 0, 255), -1)
        cv2.putText(frame, "OBS", (ox + 8, oy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

def _car_color(car_id: int) -> tuple:
    """Deterministic per-car accent colour (BGR) for guide line / HUD border.

    Cycles through a small fixed palette so N connected cars each get a
    visually distinct colour in the display thread.
    """
    palette = [
        (0, 255, 255),   # cyan/yellow
        (255, 160, 0),   # orange-blue
        (0, 220, 120),   # green
        (200, 120, 255), # violet
        (60, 60, 255),   # red
    ]
    return palette[car_id % len(palette)]


def _build_hud_lines(car_id, servo, motor, current_lane, policy,
                      lane_info, obstacle_info, steer_state, speed_state) -> list:
    """Return the HUD text lines for ONE car — no drawing.

    This is the pure-data twin of the old _draw_hud(): identical line
    content, but returned as a list of strings instead of being blitted
    onto a shared canvas. The display thread renders each car's lines
    into its own bordered HUD box (see player_launcher._draw_hud_box).
    """
    lc_st  = _lane_state.get(car_id, {})
    lctag  = "OVT" if lc_st.get("overtaking") else "NRM"
    cooptag = "C!" if speed_state.get("coop_hint") else ""
    obsd   = obstacle_info["distance"]
    obsstr = f"{obsd:.0f}px" if obsd < float("inf") else "---"
    waittag = "W" if _pp_waiting.get(car_id, False) else ""
    tier    = _car_occlusion_state.get(car_id, {}).get("tier", "??")
    _sl     = lc_st.get("start_lane")
    _od     = lc_st.get("obstacle_decision")
    _od_tag = {"overtake": "OVT", "lane_switch": "SWT"}.get(_od, "---")
    lines = [
        f"Car {car_id} Ln{current_lane} {lctag} {policy[:4].upper()}{waittag} {tier[:2]}",
        f"Start: lane{_sl}  ObsDec:{_od_tag}",
        f"Srv {servo:.2f}  Mtr {motor:.2f}{cooptag}",
        f"Lat {lane_info['lateral_error']:.0f}px {steer_state.get('lateral','-')[:4]}",
        f"Hd  {lane_info['heading_error']:.0f}°  {steer_state.get('heading','-')[:4]}",
        f"Obs {obsstr} {obstacle_info['state'][:4]}",
        f"Spd {speed_state.get('zone','-')[:4]} {steer_state.get('obstacle','-')[:4]}",
    ]
    if steer_state.get("emergency"):
        lines.append("!! EMERGENCY STOP !!")
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(frame: np.ndarray, car_id: int):
    """
    Execute one full sense → plan → act cycle.

    Parameters
    ----------
    frame       : raw (undecorated) camera frame — used for ArUco detection.
    car_id      : which car to sense/plan/act for.

    All rendering has been moved out of this function. Instead of returning
    an annotated frame, run() returns a lightweight `draw_data` dict describing
    everything the display thread needs to draw for this car (marker points,
    guide line, lookahead target, heading arrow, HUD text, interaction zone).
    The main/display thread owns ALL drawing and renders every car's
    draw_data onto one shared canvas in a fixed order — no per-thread canvas,
    no diff-mask merge.

    Returns (servo, motor, draw_data, car_log_data).
    """
    global _obstacle_positions, _cycle_counter, _car_marker_cache
    global _last_track_markers, _last_obs_positions

    _cycle_counter += 1
    # Detection always runs on the raw (undecorated) frame. No canvas is
    # touched here anymore — run() returns geometry only; the display
    # thread (player_launcher.py) does all drawing in one pass.
    detect_src = frame                              # clean pixels for ArUco
    now        = time.time()
    draw_data: dict = {
        "car_id": car_id,
        "color": _car_color(car_id),
        "rear_pt": None, "front_pt": None, "raw_rear_pt": None,
        "heading": (0.0, 0.0),
        "using_front_fallback": False, "occlusion_tier": "BOTH_OCCLUDED",
        "guide_line": None, "lookahead_pt": None, "obstacle_close": False,
        "interaction_zone": None, "hud_lines": [], "hud_emergency": False,
        "path_point": None, "obstacle_point": None,
    }

    # ── 1. Sense ─────────────────────────────────────────────────────────────
    # detect_all_markers() no longer draws anything itself — it only detects
    # and caches raw ArUco corner/id geometry into the module-level
    # `_last_marker_debug`, which the display thread renders exactly once
    # per frame via draw_marker_debug() (see player_launcher.py), instead of
    # once per connected car as the old draw_target-based approach did.
    car_markers, track_markers, obs_markers = detect_all_markers(detect_src)
    cars  = identify_cars(car_markers, car_id)
    lanes = detect_lanes(track_markers)
    global _last_lanes
    if lanes:
        _last_lanes = lanes
    # Merge live 6x6-marker obstacles with parked/unregistered-car marker
    # positions (built by identify_cars() into _inactive_car_obstacles this
    # same call) into ONE list — a single obstacle pipeline shared by both
    # source types, so an unregistered car (e.g. ID=0) sitting across both
    # lanes is checked, lane-assigned, and drawn exactly like a static
    # 6x6 obstacle.
    global _obstacle_marker_ids
    _obstacle_positions = []
    _obstacle_marker_ids = {}
    for _mid, _corner in obs_markers.items():
        _pos = marker_center(_corner)
        _obstacle_positions.append(_pos)
        _obstacle_marker_ids[_pos] = _mid
    for _mid, _pos in _inactive_car_obstacles:
        if _pos not in _obstacle_positions:
            _obstacle_positions.append(_pos)
            _obstacle_marker_ids[_pos] = _mid
    # Lane fills and obstacle circles are drawn ONCE by the display thread
    # (player_launcher.py), not here. run() no longer draws anything — it
    # only reports geometry via draw_data for the display thread to render.

    # ── 2. Pose source: raw rear/front marker centroid (no EKF) ─────────────
    # identify_cars() has already resolved car["rear"], car["midpoint"],
    # car["raw_rear"] and car["heading"] per occlusion tier:
    #   BOTH_VISIBLE / REAR_ONLY -> rear marker centroid
    #   FRONT_ONLY               -> front marker centroid (rear-fallback)
    #   BOTH_OCCLUDED            -> car_id absent from `cars`
    # No filtering/estimation stage sits between the marker detection and
    # these fields — Pure Pursuit consumes car["rear"] directly below.
    ostate   = _car_occlusion_state.get(car_id, {"tier": "BOTH_OCCLUDED",
                                                   "frames_occluded": 999,
                                                   "prev_tier": "BOTH_OCCLUDED"})

    if car_id in cars:
        car = cars[car_id]
        _update_velocity_cache(car_id, car["raw_rear"], now)

    else:
        # BOTH_OCCLUDED: both markers unavailable — perform no steering
        # computation. Hold position at the last known raw centroid (no
        # motion extrapolation, since that required the EKF's bicycle-model
        # predictor) and decay the last commanded motor output toward stop.
        n_occ = ostate["frames_occluded"]
        _last_known = _vel_cache.get(car_id, {}).get("pos")
        if n_occ > _CAR_OCCLUDE_HARD_STOP_FRAMES or _last_known is None:
            draw_data["occlusion_tier"] = "BOTH_OCCLUDED"
            return 0.0, STOP_SPEED, draw_data, {
                "policy":        DRIVING_POLICY.get(car_id, DEFAULT_POLICY),
                "pose":          [float(_last_known[0]), float(_last_known[1]), 0.0]
                                 if _last_known is not None else [0.0, 0.0, 0.0],
                "lane":          _lane_state.get(car_id, {}).get("lane", 1),
                "segment":       "occluded_hardstop",
                "curvature":     0.0,
                "servo":         0.0,
                "motor":         0.0,
                "waiting":       False,
                "lateral_error": 0.0,
                "heading_error": 0.0,
                "obstacle_info": {"state": "unknown", "distance": float("inf")},
                "minicar_events": ["occluded_hardstop"],
                "safety_events":  {"car": ["no_collision"], "object": ["no_collision"]},
                "occlusion_tier": "BOTH_OCCLUDED",
                "emergency_stop": _lane_state.get(car_id, {}).get("emergency_stop", False),
            }
        ghost_pos = (int(_last_known[0]), int(_last_known[1]))
        cars[car_id] = {
            "front": None, "rear": ghost_pos, "midpoint": ghost_pos,
            "raw_rear": ghost_pos,
            "heading": (0.0, 0.0),
            "using_front_fallback": False,
            "occlusion_tier": "BOTH_OCCLUDED",
        }
        _lc    = _last_cmd.get(car_id, (0.0, 0.0))
        _decay = _OCCLUDE_VEL_DECAY ** n_occ
        _occ_log = {
            "policy":        DRIVING_POLICY.get(car_id, DEFAULT_POLICY),
            "pose":          [float(ghost_pos[0]), float(ghost_pos[1]), 0.0],
            "lane":          _lane_state.get(car_id, {}).get("lane", 1),
            "segment":       "occluded",
            "curvature":     0.0,
            "servo":         round(_lc[0], 2),
            "motor":         round(_lc[1] * _decay, 3),
            "waiting":       False,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "obstacle_info": {"state": "unknown", "distance": float("inf")},
            "minicar_events": ["occluded"],
            "safety_events":  {"car": ["no_collision"], "object": ["no_collision"]},
            "occlusion_tier": "BOTH_OCCLUDED",
            "emergency_stop": _lane_state.get(car_id, {}).get("emergency_stop", False),
        }
        draw_data["occlusion_tier"] = "BOTH_OCCLUDED"
        return _lc[0], round(_lc[1] * _decay, 3), draw_data, _occ_log


    car     = cars[car_id]
    car_ref = car["rear"]
    # Obstacle-distance checks use the FRONT marker when visible (it is the
    # part of the car that actually reaches an obstacle first); fall back to
    # the rear-marker centroid when the front marker is not currently visible.
    obstacle_ref_pos = car.get("front") if car.get("front") else car_ref

    # Marker-point geometry for THIS car only — the display thread iterates
    # over every car's own draw_data to render all of them, so run() no
    # longer needs to loop over `cars` and draw every other car here too.
    draw_data["rear_pt"]  = car.get("rear")
    draw_data["front_pt"] = car.get("front")
    draw_data["raw_rear_pt"] = car.get("raw_rear", car.get("rear"))
    draw_data["heading"] = car.get("heading", (0.0, 0.0))
    draw_data["using_front_fallback"] = bool(car.get("using_front_fallback", False))
    draw_data["occlusion_tier"] = car.get("occlusion_tier", "BOTH_VISIBLE")


    # ── 3. Reference path & lane ─────────────────────────────────────────────
    current_lane = _lane_state.get(car_id, {}).get("lane", 1)
    # Tier-aware lane classification: only pass positions confirmed live for this tier.
    _tier_cl  = car.get("occlusion_tier", "BOTH_VISIBLE")
    _front_cl = car.get("front")    if _tier_cl in ("BOTH_VISIBLE", "FRONT_ONLY") else None
    _rear_cl  = car.get("raw_rear") if _tier_cl in ("BOTH_VISIBLE", "REAR_ONLY")  else None
    current_lane = _classify_lane(_front_cl, _rear_cl, lanes, current_lane)

    # ── Emergency stop: car straddles both lanes ─────────────────────────────
    # If the car's front marker and rear marker are each in different lanes the
    # car is physically spanning the middle boundary.  Stop immediately.
    _straddle_front = car.get("front") if _tier_cl in ("BOTH_VISIBLE", "FRONT_ONLY") else None
    _straddle_rear  = car.get("raw_rear") if _tier_cl in ("BOTH_VISIBLE", "REAR_ONLY")  else None
    if car_id not in ACTIVE_CAR_IDS and _car_straddles_lanes(_straddle_front, _straddle_rear, lanes):
        _lane_state.setdefault(car_id, {})["emergency_stop"] = True

        return 0.0, STOP_SPEED, draw_data, {
            "policy": DRIVING_POLICY.get(car_id, DEFAULT_POLICY),
            "pose": [float(car_ref[0]), float(car_ref[1]),
                     round(math.degrees(aruco_heading_rad(car)) % 360, 2)],
            "lane": current_lane, "segment": "straddle", "curvature": 0.0,
            "servo": 0.0, "motor": STOP_SPEED, "waiting": True,
            "lateral_error": 0.0, "heading_error": 0.0,
            "obstacle_info": {"state": "emergency_straddle", "distance": float("inf")},
            "minicar_events": ["emergency_stop_straddle"],
            "safety_events":  {"car": ["emergency_straddle"], "object": ["no_collision"]},
            "occlusion_tier": _tier_cl,
        }

    # ref_curve for lookahead follows start_lane in steady state, but must
    # follow the physically active lane (st["lane"]) while an "overtake" is
    # in progress — otherwise the car keeps steering toward start_lane's
    # centreline even after having logically switched lanes, and never
    # actually drives alongside the obstacle in the adjacent lane.
    # A "lane_switch" decision is permanent and already updates start_lane
    # elsewhere, so it is unaffected by this branch.
    _st_nav = _lane_state.get(car_id, {})
    _nav_lane = (_st_nav.get("lane") if _st_nav.get("overtaking")
                 else (_st_nav.get("start_lane") or current_lane))
    ref_curve = lanes.get(f"lane{_nav_lane}_ref")
    if not ref_curve:
        # Lanes not fitted yet — emit a minimal log entry so the car is
        # visible in the log from frame 1 instead of disappearing silently.
        _theta_no_ref = round(math.degrees(aruco_heading_rad(car)) % 360, 2)
        return 0.0, STOP_SPEED, draw_data, {
            "policy":        DRIVING_POLICY.get(car_id, DEFAULT_POLICY),
            "pose":          [float(car_ref[0]), float(car_ref[1]), _theta_no_ref],
            "lane":          current_lane,
            "segment":       "no_lane_ref",
            "curvature":     0.0,
            "servo":         0.0,
            "motor":         STOP_SPEED,
            "waiting":       False,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "obstacle_info": {"state": "unknown", "distance": float("inf")},
            "minicar_events": ["no_lane_ref"],
            "safety_events":  {"car": ["no_collision"], "object": ["no_collision"]},
            "occlusion_tier": _tier_cl,
            "emergency_stop": _lane_state.get(car_id, {}).get("emergency_stop", False),
        }

    lane_info     = compute_lane_measurements(car_ref, car["heading"], ref_curve)

    obstinfo     = nearest_relevant_obstacle(obstacle_ref_pos, lane_info["tangent"],
                                              lane_info["left_normal"],
                                              _obstacle_positions)
    # Attach the lane tangent used for this obstacle check so _decide_lane_*
    # can stash it as "pass_tangent" — needed by _rear_passed_obstacle() to
    # measure how far the rear marker has travelled past the obstacle point.
    obstinfo["tangent"] = lane_info["tangent"]

    policy = DRIVING_POLICY.get(car_id, DEFAULT_POLICY)

    # Ensure _lane_state[car_id] always carries ALL required keys BEFORE any
    # branch below touches it — including the emergency-stop branch, which
    # must never create a partial dict via setdefault(). Without this, a car
    # whose very first frame is both-lanes-blocked would get a bare
    # {"emergency_stop": ...} entry, and _decide_lane_* would later KeyError
    # on "start_lane" once the obstacle clears and the else-branch init
    # check saw car_id already present (with missing keys) and skipped init.
    if car_id not in _lane_state:
        _lane_state[car_id] = {
            "lane": current_lane, "timer": now, "overtaking": False,
            "start_lane": None, "obstacle_decision": None,
        }
    # else:
    #     _lane_state[car_id]["lane"] = current_lane

    # ── Emergency stop: BOTH lanes blocked (checked FIRST, exclusively) ──────
    # Evaluated fresh every frame from the CURRENT both_blocking reading only —
    # it must never latch on the previous frame's own flag, otherwise the car
    # stays hard-stopped forever after a single both-lanes-blocking frame, even
    # once the obstacle clears or recedes into the "near" (slow-down) band.
    #
    # This check now happens BEFORE _decide_lane(), compute_lookahead(),
    # _curve_speed() and _apply_coordination() are ever called — when both
    # lanes are blocked there is nowhere safe to drive (overtaking would
    # require leaving the track's inner/outer boundaries, which is not
    # allowed), so none of that planning work is meaningful and none of it
    # is executed. No other obstacle/lane/coordination event (safety_stop,
    # obstacle_overtake, obstacle_lane_switch, follow_start_lane,
    # both_lanes_near_slowdown, lane_change, etc.) can be appended or take
    # effect on a frame where both_blocking is True — the car simply cannot
    # advance, so nothing about "how" it would advance applies.
    both_blocking = _both_lanes_blocked(obstacle_ref_pos, lanes)
    _lane_state[car_id]["emergency_stop"] = both_blocking

    if both_blocking:
        current_lane = _lane_state.get(car_id, {}).get("lane", 1)
        minicar_events = ["emergency_stop_both_lanes_blocked"]
        car_safety_events = ["no_collision"]
        object_safety_events = ["emergency_break"]
        obs_state = obstinfo["state"]   # used later by steer_state/HUD logging
        servo     = 0.0
        motor     = 0.0
        cte       = 0.0
        raw_delta = 0.0
        coop_hint = False
        segment, curvature = _classify_segment(car_ref, lanes, current_lane,
                                                ref_idx=lane_info["idx"])
        # Hard stop: force motor to 0 directly instead of letting the PID
        # controller ease toward STOP_SPEED — an emergency stop must not
        # leave a nonzero motor command on the wire.
        _spd_pid[car_id] = {"i": 0.0, "e_prev": 0.0, "t_prev": now,
                             "pos_prev": _spd_pid.get(car_id, {}).get("pos_prev"),
                             "v_prev": 0.0, "cmd_prev": 0.0}
        if car_id in _vel_cache:
            _vel_cache[car_id]["v"] = 0.0
        new_lane = current_lane
    else:
        # _lane_state[car_id] is already fully initialised (see above, run
        # unconditionally before the both_blocking check), so it is safe to
        # read "start_lane" etc. here without any additional init.
        prev_lane = _lane_state.get(car_id, {}).get("lane", 1)

        # ── 4. Lane-change decision ───────────────────────────────────────────
        new_lane    = _decide_lane(car_id, car_ref, lanes, obstinfo, cars, now,
                                    rear_pos=car.get("raw_rear"))
        
        # Persist the DECIDED lane, not the raw geometric classification —
        # _decide_lane() is the source of truth for lane changes/switches/
        # overtakes. Overwriting this with the geometric _classify_lane()
        # result on the next frame's setup (above) would silently discard
        # any decision made here, causing prev_lane/new_lane to disagree
        # every frame and the lane_change event to never clear.
        _lane_state[car_id]["lane"] = new_lane

        _st         = _lane_state.get(car_id, {})
        _start_lane = _st.get("start_lane") or new_lane
        # Read the real decision made in _decide_lane_* (was previously
        # hardcoded to "lane_switch", which silently discarded the actual
        # random overtake/lane_switch choice and broke the overtake branch).
        _obs_dec    = _st.get("obstacle_decision")   # "overtake" | "lane_switch" | None

        _returned = _lane_state.get(car_id, {}).pop("returned_to_startlane", False)

        # ── Corner-point, lane-aware safety classification ─────────────────
        # near_miss/collision must only ever be evaluated against obstacles
        # and other cars sharing THIS car's current lane. Cars on different
        # lanes — even if geometrically close (e.g. inner/outer at a bend) —
        # must never register as near_miss/collision and must never trigger
        # emergency_stop. Uses the actual ArUco corner points (front + rear
        # markers) rather than a single centroid, so a corner clipping the
        # obstacle/car is caught even if the marker centers are still clear.
        _own_corners = [c for c in (_car_marker_cache.get(car_id, {}).get("front_corner"),
                                     _car_marker_cache.get(car_id, {}).get("rear_corner"))
                        if c is not None]
        # Split car-vs-car and car-vs-object safety events (request: keep
        # collision/near_miss/decision distinct by source, not merged into
        # one flat list).
        car_safety_events = ["no_collision"]
        object_safety_events = ["no_collision"]

        # A car counts as "maneuvering" only while actively overtaking or
        # mid lane_switch (i.e. physically occupying/crossing into a lane
        # other than its locked start_lane for this obstacle encounter).
        # Cross-lane car-to-car thresholds are evaluated ONLY when at
        # least one of the two cars involved is maneuvering — a car sitting
        # still (or cruising) in its own lane must never be flagged against
        # a car in the other lane just because they are geometrically close
        # at a bend.
        def _is_maneuvering(_cid):
            _s = _lane_state.get(_cid, {})
            return bool(_s.get("overtaking")) or (_s.get("obstacle_decision") in ("overtake", "lane_switch"))

        _self_maneuvering = _is_maneuvering(car_id)

        def _min_corner_dist_to_point(corners, point):
            best = float("inf")
            p = np.array(point, dtype=float)
            for c in corners:
                pts = np.array(c, dtype=float).reshape(-1, 2)
                d = float(np.min(np.linalg.norm(pts - p, axis=1)))
                if d < best:
                    best = d
            return best

        if _own_corners:
            _rel_obs = [o for o in _obstacle_positions
                        if _obstacle_lane(o, lanes) == new_lane]
            for _opos in _rel_obs:
                _od = _min_corner_dist_to_point(_own_corners, _opos)
                if _od <= OBSTACLE_D_COL:
                    object_safety_events = ["collision"]
                    _lane_state[car_id]["emergency_stop"] = True
                    break
                elif _od <= OBSTACLE_D_WARN:
                    object_safety_events = ["near_miss"]
                elif _od <= OBSTACLE_D_SAFE and object_safety_events[0] not in ("collision", "near_miss"):
                    object_safety_events = ["decision"]

            if object_safety_events[0] != "collision":
                for _oc_id, _oc in cars.items():
                    if _oc_id == car_id:
                        continue
                    _oc_lane = _lane_state.get(_oc_id, {}).get("lane")
                    _oc_maneuvering = _is_maneuvering(_oc_id)
                    _same_lane = (_oc_lane == new_lane)
                    # Cars on different lanes are only distance-checked
                    # against each other while one of the two is actively
                    # overtaking or switching lanes. Otherwise (both static
                    # in their own lanes) different-lane pairs are skipped,
                    # exactly as before.
                    if not _same_lane and not (_self_maneuvering or _oc_maneuvering):
                        continue
                    _oc_corners = [c for c in (_car_marker_cache.get(_oc_id, {}).get("front_corner"),
                                                _car_marker_cache.get(_oc_id, {}).get("rear_corner"))
                                   if c is not None]
                    if not _oc_corners:
                        continue
                    _cd = min(_min_corner_dist_to_point(_own_corners, marker_center(c))
                              for c in _oc_corners)
                    if _cd <= D_COL:
                        car_safety_events = ["collision"]
                        _lane_state[car_id]["emergency_stop"] = True
                        break
                    elif _cd <= D_WARN:
                        car_safety_events = ["near_miss"]
                    elif _cd <= D_SAFE and car_safety_events[0] not in ("collision", "near_miss"):
                        car_safety_events = ["decision"]

        if new_lane != prev_lane:
            # Physical lane change: pure-pursuit must always track the lane
            # the car is now actually on, for every decision type (overtake,
            # lane_switch, return_start_lane, or plain lane_change) —
            # otherwise steering keeps referencing the lane it just left and
            # the car can never physically get there.
            ref_curve = lanes.get(f"lane{new_lane}_ref", ref_curve)
            lane_info  = compute_lane_measurements(car_ref, car["heading"], ref_curve)
            obstinfo = nearest_relevant_obstacle(obstacle_ref_pos, lane_info["tangent"],
                                                  lane_info["left_normal"],
                                                  _obstacle_positions)
            obstinfo["tangent"] = lane_info["tangent"]

            if _returned:
                minicar_events = ["return_start_lane"]   # ref_curve now back on start_lane
            elif _obs_dec == "overtake":
                minicar_events = ["lane_change"]         # ref_curve now on adjacent lane (temporary)
            elif _obs_dec == "lane_switch":
                minicar_events = ["follow_switch_lane"]  # ref_curve now on adjacent lane (permanent)
            else:
                minicar_events = ["lane_change"]
        else:
            # Steady-state: following start_lane (may physically straddle, that is normal)
            if new_lane == _start_lane:
                minicar_events = ["follow_start_lane"]
            elif _st.get("overtaking"):
                # Mid-maneuver: still alongside the obstacle in the adjacent
                # lane, waiting for _rear_passed_obstacle() to clear before
                # merging back. Completes the lane_change -> alongside ->
                # return_start_lane -> follow_start_lane sequence in the log.
                minicar_events = ["lane_change"]
            else:
                # Holding a permanently-adopted switched lane, steady state.
                minicar_events = ["follow_switch_lane"]
        current_lane = new_lane

        # ── 5. Pure-pursuit steering ─────────────────────────────────────────
        if abs(float(np.dot(
                np.array(car_ref, float) - np.array(ref_curve[0], float),
                np.array([0.0, 1.0])))) > 120.0 and car_id in _car_fwd_step:
            del _car_fwd_step[car_id]

        raw_delta, cte, _, target_pt = compute_lookahead(
            car_ref, car["heading"], ref_curve, car_id)


        # Report lookahead target geometry (guide line: car_ref -> target_pt).
        # obstacle_close controls the guide-line colour (red vs. neutral) —
        # the actual cv2.line/circle calls now live in the display thread.
        if target_pt:
            draw_data["guide_line"] = [tuple(car_ref), tuple(target_pt)]
            draw_data["lookahead_pt"] = tuple(target_pt)
            draw_data["obstacle_close"] = bool(obstinfo["state"] in ("blocking", "near"))

        # ── Interaction zone geometry — S4 scenario only ──────────────────
        if LOG_SCENARIO == "S4":
            draw_data["interaction_zone"] = {
                "car_pos": tuple(car_ref),
                "tangent": tuple(lane_info["tangent"]),
                "left_normal": tuple(lane_info["left_normal"]),
                "obs_state": obstinfo["state"],
            }

        # ── 6. Speed (curve + CTE penalty + obstacle + coordination + PID) ──
        base_speed = _curve_speed(car_ref, lanes, current_lane)

        # CTE penalty: reduce setpoint when lateral error is large (S1 stability,
        # S2 car-following smoothness — prevents overshoot mid-correction)
        base_speed = cte_speed_penalty(base_speed, cte)

        coop_hint = False
        if car_id in _coop_slowdown_until and now < _coop_slowdown_until[car_id]:
            base_speed = min(base_speed, SLOW_SPEED)
            coop_hint  = True

        speeds    = _apply_coordination(cars, base_speed)
        v_desired = speeds.get(car_id, base_speed)

        # ── Per-lane distance-threshold rule (front marker vs. obstacle) ────
        # Same three-state rule as car-to-car coordination — no drawn zones,
        # just a plain step function on distance, evaluated at the car's own
        # (single) lane obstacle reading.
        obs_state = obstinfo["state"]
        if obs_state == "blocking":
            v_desired = STOP_SPEED
            # object_safety_events.append("safety_stop")
        elif obs_state == "near":
            v_desired = min(v_desired, SLOW_SPEED)
            # object_safety_events.append("obstacle_near_slowdown")

        # Both lanes in the yield/near band (but not yet both blocking) —
        # early speed reduction only, never a stop.
        if _both_lanes_near(obstacle_ref_pos, lanes):
            v_desired = min(v_desired, SLOW_SPEED)
            # object_safety_events.append("both_lanes_near_slowdown")

        # PID longitudinal controller: smooth continuous motor command that
        # tracks v_desired using a raw-pose finite-difference velocity estimate.
        motor = pid_speed(car_id, v_desired, now)

    # ── 7. Map raw_delta (rad) → continuous servo [-0.5, 0.5] ───────────────
    # δ ∈ [-π/4, +π/4] → servo = δ × _RAD_TO_SERVO, clamped to [-MAX_SERVO, +MAX_SERVO]
    # No /2 fudge factor: PI/servo-smoothing gains handle tuning instead.
    # Servo: negative = LEFT turn, positive = RIGHT turn, 0.0 = straight.
    _raw_servo = float(np.clip(raw_delta * _RAD_TO_SERVO, -MAX_SERVO, MAX_SERVO))
    servo      = _smooth_servo(car_id, _raw_servo)  # EMA low-pass: damps ArUco noise

    # ── 8. (Former "update EKF inputs" step — no longer needed; the PI
    # speed controller reads velocity from the raw-pose cache updated in
    # step 2, and there is no bicycle-model predictor left to feed.) ────────

    # ── 9. Segment / HUD / logging ────────────────────────────────────────────
    # Reuse the ref index already computed by compute_lane_measurements so that
    # segment type and heading_error are anchored to the exact same curve point.
    segment, curvature = _classify_segment(car_ref, lanes, current_lane,
                                            ref_idx=lane_info["idx"])

    _emerg = _lane_state.get(car_id, {}).get("emergency_stop", False)
    steer_state = {
        "lateral":  "ok"      if abs(cte) < 15 else "high",
        "heading":  "aligned" if abs(lane_info["heading_error"]) < 10 else "error",
        "obstacle": obs_state,
        "emergency": _emerg,
    }
    speed_state = {
        "zone":      segment,
        "coop_hint": coop_hint,
    }
    draw_data["hud_lines"] = _build_hud_lines(
        car_id, servo, motor, current_lane, policy,
        lane_info, obstinfo, steer_state, speed_state)
    draw_data["hud_emergency"] = bool(steer_state.get("emergency"))
    draw_data["path_point"] = tuple(lane_info["path_point"])
    draw_data["obstacle_point"] = (
        tuple(map(int, obstinfo["point"])) if obstinfo["point"] is not None else None
    )

    theta_deg = math.degrees(aruco_heading_rad(car)) % 360
    # Enrich cars snapshot with lane + segment_idx so _build_log_entry
    # can report same_lane / seg_delta for every pair without mutating live state.
    _cars_snap = {}
    for _cid, _cd in cars.items():
        _cl   = _lane_state.get(_cid, {}).get("lane", 1)
        _cref_curve = lanes.get(f"lane{_cl}_ref", [])
        _sidx = project_onto_curve(_cd["midpoint"], _cref_curve) if _cref_curve else None
        _cars_snap[_cid] = dict(_cd, lane=_cl, segment_idx=_sidx)
    # Build per-car log data — collected by the main loop into one frame entry.
    # Car-to-object safety classification — every currently tracked obstacle,
    # not just the single nearest one used for control decisions above.
    # Reuses the same _obstacle_positions list and obstacle_ref_pos already
    # computed for obstinfo; only adds the D_COL/D_SAFE bucket per obstacle.
    obs_interactions = []
    for _opos in _obstacle_positions:
        # Corner-point distance (front/rear marker corners), not just the
        # single reference centroid, so a corner clipping the obstacle is
        # caught even while the marker centers are still technically clear.
        _corners_now = [c for c in (_car_marker_cache.get(car_id, {}).get("front_corner"),
                                     _car_marker_cache.get(car_id, {}).get("rear_corner"))
                        if c is not None]
        if _corners_now:
            _od = min(float(np.min(np.linalg.norm(
                        np.array(c, float).reshape(-1, 2) - np.array(_opos, float), axis=1)))
                      for c in _corners_now)
        else:
            _od = float(np.linalg.norm(
                np.array(obstacle_ref_pos, float) - np.array(_opos, float)))
        # 4-way (request 1 + 2): pass thresholds in (col, warn, safe) order,
        # matching the renamed OBSTACLE_D_COL < OBSTACLE_D_WARN < OBSTACLE_D_SAFE.
        _ostate = _classify_safety(_od, OBSTACLE_D_COL, OBSTACLE_D_WARN, OBSTACLE_D_SAFE)
        obs_interactions.append({
            "marker_id": _obstacle_marker_ids.get(_opos),
            "distance_px": round(_od, 2),
            "state": _ostate,
        })
        # Priority: collision > near_miss > decision > no_collision — a
        # "decision" reading is now recorded as its own safety_event
        # (request 1) instead of being silently dropped/overwritten.
        # These are OBJECT (car-to-obstacle) events, kept separate from
        # car-to-car events (car_safety_events) so the log can distinguish
        # the two interaction types.
        if _ostate == "collision":
            object_safety_events = ["collision"]
            _lane_state[car_id]["emergency_stop"] = True
        elif _ostate == "near_miss" and object_safety_events[0] != "collision":
            object_safety_events = ["near_miss"]
        elif _ostate == "decision" and object_safety_events[0] not in ("collision", "near_miss"):
            object_safety_events = ["decision"]

    car_log_data = {
        "policy":        policy,
        "pose":          [car_ref[0], car_ref[1], theta_deg],
        "lane":          current_lane,
        "segment":       segment,
        "curvature":     curvature,
        "servo":         round(servo, 2),
        "motor":         round(motor, 2),
        "waiting":       _pp_waiting.get(car_id, False),
        "lateral_error": lane_info["lateral_error"],
        "heading_error": lane_info["heading_error"],
        "obstacle_info": obstinfo,
        "obstacle_interactions": obs_interactions,
        "minicar_events": minicar_events,
        "safety_events":  {"car": car_safety_events, "object": object_safety_events},
        "maneuvering":    _lane_state.get(car_id, {}).get("overtaking", False) or
                           (_lane_state.get(car_id, {}).get("obstacle_decision") in ("overtake", "lane_switch")),
        "occlusion_tier": cars.get(car_id, {}).get("occlusion_tier", "BOTH_VISIBLE"),
        "emergency_stop": _lane_state.get(car_id, {}).get("emergency_stop", False),
        "start_lane":     _lane_state.get(car_id, {}).get("start_lane", new_lane),
        "obstacle_decision": _lane_state.get(car_id, {}).get("obstacle_decision"),
    }

    _last_cmd[car_id] = (round(servo, 2), round(motor, 2))
    return round(servo, 2), round(motor, 2), draw_data, car_log_data


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def load_calibration(path: str) -> None:
    global _CAM_K, _CAM_D, _CALIBRATED, _DFOV, _CAM_RVEC, _CAM_TVEC, _CALIB_PATH
    data = np.load(path)
    with _calib_lock:
        _CAM_K, _CAM_D = data["K"], data["D"]
        if "rvec" in data and "tvec" in data:
            _CAM_RVEC = data["rvec"].reshape(3, 1)
            _CAM_TVEC = data["tvec"].reshape(3, 1)
        _CALIBRATED = True
    _CALIB_PATH = path
    # Infer diagonal FOV from filename convention, e.g. "calib-90deg.npz" → "90"
    stem = os.path.basename(path)
    for _tag in ("90", "78"):
        if _tag in stem:
            _DFOV = _tag
            break
    # Camera height (request 4): only set when a known DFOV tag was detected
    # above — measured heights are fixed per rig (90fov mount = 170 cm,
    # 78fov mount = 220 cm). Left as None for unrecognised/untagged files
    # rather than guessing.
    camera_height = _DFOV_HEIGHT_CM.get(_DFOV)
    print(f"[calib] loaded {path}  K={_CAM_K}  D={_CAM_D.ravel()}  dFOV={_DFOV}  height_cm={camera_height}")


def estimate_extrinsics(world_pts: np.ndarray, image_pts: np.ndarray) -> bool:
    """
    Estimate camera extrinsics (rvec, tvec) from N≥4 point correspondences.

    Called automatically once per session after the track geometry is fitted,
    using all 12 track-boundary markers as world↔pixel pairs derived from
    the stadium geometry and TRACK_LONG_AXIS_CM.

    Parameters
    ----------
    world_pts : (N, 3) float64  3-D world points in cm, origin = track centroid
    image_pts : (N, 2) float64  corresponding pixel coordinates

    Returns True on success, False otherwise.
    """
    global _CAM_RVEC, _CAM_TVEC, _extrinsics_done
    if not _CALIBRATED:
        print("[calib] estimate_extrinsics: intrinsics not loaded yet.")
        return False
    if world_pts.shape[0] < 4:
        print("[calib] estimate_extrinsics: need ≥ 4 point pairs.")
        return False
    with _calib_lock:
        K = _CAM_K.copy()
        D = _CAM_D.copy()
    ok, rvec, tvec = cv2.solvePnP(
        world_pts.astype(np.float64),
        image_pts.astype(np.float64),
        K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("[calib] estimate_extrinsics: solvePnP failed.")
        return False
    with _calib_lock:
        _CAM_RVEC = rvec.reshape(3, 1)
        _CAM_TVEC = tvec.reshape(3, 1)
    _extrinsics_done = True
    print(f"[calib] Extrinsics estimated from {world_pts.shape[0]} markers  "
          f"rvec={rvec.ravel()}  tvec={tvec.ravel()}")
    return True


def world_to_pixel(x_cm: float, y_cm: float, z_cm: float = 0.0) -> tuple:
    """
    Project a track-centred world point (x, y, z) in cm → pixel (u, v).
    Requires both intrinsics (_CALIBRATED) and extrinsics (_extrinsics_done).
    Returns (0, 0) when either is not available.
    """
    if not (_CALIBRATED and _extrinsics_done):
        return (0, 0)
    with _calib_lock:
        K  = _CAM_K.copy()
        D  = _CAM_D.copy()
        rv = _CAM_RVEC.copy()
        tv = _CAM_TVEC.copy()
    if np.allclose(rv, 0) and np.allclose(tv, 0):
        return (0, 0)
    pts = np.array([[float(x_cm), float(y_cm), float(z_cm)]], dtype=np.float64)
    projected, _ = cv2.projectPoints(pts, rv, tv, K, D)
    u, v = projected[0][0]
    return (int(round(u)), int(round(v)))


def save_calibration(path: str, mtx, dist,
                     rvec: np.ndarray = None,
                     tvec: np.ndarray = None) -> None:
    global _CAM_K, _CAM_D, _CALIBRATED, _CAM_RVEC, _CAM_TVEC
    save_kwargs = dict(K=mtx, D=dist)
    if rvec is not None: save_kwargs["rvec"] = rvec
    if tvec is not None: save_kwargs["tvec"] = tvec
    np.savez(path, **save_kwargs)
    with _calib_lock:
        _CAM_K, _CAM_D = mtx, dist
        if rvec is not None: _CAM_RVEC = rvec.reshape(3, 1)
        if tvec is not None: _CAM_TVEC = tvec.reshape(3, 1)
        _CALIBRATED = True
    print(f"[calib] saved → {path}")


def undistort_frame(frame: np.ndarray) -> np.ndarray:
    if not _CALIBRATED: return frame
    with _calib_lock:
        k, d = _CAM_K.copy(), _CAM_D.copy()
    return cv2.undistort(frame, k, d)


def _calib_mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = True


def wizard_chess(cap, chess_size=(8,6), square_mm=38.0) -> tuple:
    cols, rows = chess_size
    objp = np.zeros((cols*rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    objpts, imgpts, captured = [], [], 0
    click = [False]
    win   = "Calibration — click to capture (ESC=done)"
    cv2.namedWindow(win); cv2.setMouseCallback(win, _calib_mouse_cb, click)
    print(f"[calib] Chessboard {cols}×{rows}, {square_mm}mm — LEFT CLICK to capture, ESC to finish")
    while True:
        ok, frame = cap.read()
        if not ok: continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        disp  = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, (cols, rows), corners, found)
            cv2.putText(disp, "Board detected — click to capture", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(disp, "No board found", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(disp, f"Captured: {captured}  ESC=done", (20,75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.imshow(win, disp)
        cv2.waitKey(1)
        if click[0]:
            click[0] = False
            if found:
                c2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), CALIB_CRITERIA)
                objpts.append(objp); imgpts.append(c2); captured += 1
                print(f"[calib] frame {captured} captured")
            else:
                print("[calib] click ignored — no board in this frame")
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow(win)
    if captured < CALIB_MIN_FRAMES:
        print(f"[calib] only {captured} frames, need {CALIB_MIN_FRAMES} — aborted")
        return None, None
    h, w = gray.shape
    rms, mtx, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, (w, h), None, None)
    print(f"[calib] RMS = {rms:.4f} px")
    return mtx, dist


def wizard_charuco(cap, grid_size=(8,6), square_mm=38.0, marker_mm=18.0) -> tuple:
    from cv2 import aruco as _ar
    cols, rows  = grid_size
    a_dict      = _ar.getPredefinedDictionary(_ar.DICT_4X4_250)
    board       = _ar.CharucoBoard(cols, rows, square_mm/1000, marker_mm/1000, a_dict)
    detector    = _ar.CharucoDetector(board, _ar.CharucoParameters(), _ar.DetectorParameters())
    all_corners, all_ids, captured = [], [], 0
    click = [False]
    win   = "Calibration ChArUco — click to capture (ESC=done)"
    cv2.namedWindow(win); cv2.setMouseCallback(win, _calib_mouse_cb, click)
    print(f"[calib] ChArUco {cols}×{rows}, sq={square_mm}mm mk={marker_mm}mm — LEFT CLICK, ESC=done")
    gray = None
    while True:
        ok, frame = cap.read()
        if not ok: continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp  = frame.copy()
        cc, ci, _, _ = detector.detectBoard(gray)
        if ci is not None and len(ci) >= 4:
            cv2.aruco.drawDetectedCornersCharuco(disp, cc, ci)
            cv2.putText(disp, f"ChArUco {len(ci)} corners — click", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(disp, "No ChArUco board found", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(disp, f"Captured: {captured}  ESC=done", (20,75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.imshow(win, disp)
        cv2.waitKey(1)
        if click[0]:
            click[0] = False
            if ci is not None and len(ci) >= 4:
                all_corners.append(cc); all_ids.append(ci); captured += 1
                print(f"[calib] ChArUco frame {captured} captured")
            else:
                print("[calib] click ignored — board not detected")
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow(win)
    if captured < CALIB_MIN_FRAMES:
        print(f"[calib] only {captured} frames — aborted"); return None, None
    ok2, f2 = cap.read()
    h, w = (f2.shape[:2] if ok2 else gray.shape)
    rms, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, w, h, None, None)
    print(f"[calib] RMS = {rms:.4f} px")
    return mtx, dist


def run_calibration_wizard(cap, board_type="chess", chess_size=(8,6),
                           square_mm=38.0, marker_mm=18.0, save_path=None):
    out = save_path or CALIB_DEFAULT_OUT
    mtx, dist = (wizard_charuco(cap, chess_size, square_mm, marker_mm)
                 if board_type == "charuco"
                 else wizard_chess(cap, chess_size, square_mm))
    if mtx is not None:
        save_calibration(out, mtx, dist)


def _startup_calibration(args, cap) -> None:
    """
    Three exclusive calibration flows (call from __main__ after argparse):

      --calib-file PATH              Load existing .npz → activate undistortion.
      --calibrate                    Run wizard → save to calib.npz.
      --calibrate --calib-file PATH  Run wizard → save to PATH.
      (neither)                      RAW mode, no undistortion.

    FOV note: call this function BEFORE applying --fov, so the CLI flag
    can always override the filename-inferred _DFOV value.
    """
    if args.calib_file and not args.calibrate:
        if not os.path.isfile(args.calib_file):
            print(f"[calib] ERROR: {args.calib_file} not found. Run --calibrate first.")
            sys.exit(1)
        load_calibration(args.calib_file)
        print(f"[calib] Mode: LOAD  → {args.calib_file}")
    elif args.calibrate:
        save_path = args.calib_file or CALIB_DEFAULT_OUT
        print(f"[calib] Mode: WIZARD → saving to {save_path}")
        run_calibration_wizard(cap, board_type=args.board,
                               chess_size=tuple(args.chess_size),
                               square_mm=args.square_mm,
                               marker_mm=args.marker_mm,
                               save_path=save_path)
    else:
        print("[calib] Mode: RAW (no undistortion)")

def init_camera(cam_index: int = 0,
                video_name: str | None = None) -> tuple[cv2.VideoCapture, cv2.VideoWriter]:
    """Open the overhead camera and create a VideoWriter for optional recording.

    Mirrors auto_control.init_camera exactly so both entry-points behave
    identically.  Pass --video-name on the CLI to enable recording; omit it
    and only the VideoCapture is useful (the writer is created but never written
    to unless `out` is used in the display loop).
    """

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(cam_index, backend)
    if not cap.isOpened():
        sys.exit(f"[error] Camera {cam_index} not found.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_rate   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Width : {frame_width}")
    print(f"Height: {frame_height}")
    print(f"FPS   : {frame_rate}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _vname = (video_name.strip() if video_name else None)
    if _vname is None:
        print("[video] No --video-name given, skipping recording.")
        return cap, None
    _vpath = _vname if _vname.lower().endswith(".mp4") else _vname + ".mp4"
    print(f"[video] Recording to: {_vpath}")
    out = cv2.VideoWriter(_vpath, fourcc, 30.0,
                          (int(frame_width), int(frame_height)))
    return cap, out

# ══════════════════════════════════════════════════════════════════════════════
# REMOTE BOOT
# ══════════════════════════════════════════════════════════════════════════════

_SSH_USERNAME    = "cpslab1"
_SSH_PASSWORD    = "cpslab1"
_LAUNCH_SCRIPT   = "./player/launch.txt"


def _car_ip(car_id: int) -> str:
    """Derive Raspberry Pi IP from car ID, matching player_launcher.py logic."""
    if car_id in range(0, 10):
        return f"192.168.0.20{car_id}"
    return f"192.168.0.2{car_id}"


def boot(ip: str, username: str, password: str,
         launch_script: str = "./player/launch.txt") -> None:
    """SSH into a minicar Pi and launch the on-board player via PuTTY.

    Runs in its own daemon thread — one per car.  Blocks until the remote
    command exits (or the thread is killed at process shutdown).

    Parameters
    ----------
    ip            : IP address of the minicar Raspberry Pi.
    username      : SSH username (e.g. 'cpslab1').
    password      : SSH password (e.g. 'cpslab1').
    launch_script : Local path to the PuTTY command file (launch.txt).
    """
    cmd = f'putty -ssh {username}@{ip} -pw {password} -m "{launch_script}"'
    print(f"[boot] Launching player on {ip}  →  {cmd}")
    os.system(cmd)
    print(f"[boot] Player session ended for {ip}")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     import argparse, socket, struct

#     parser = argparse.ArgumentParser(description="Autonomous minicar controller — multi-car")
#     parser.add_argument("-n", "--cars", nargs="+", type=int, default=None,
#                         help="Car IDs to control, e.g. -n 1 2 3")
#     parser.add_argument("--no-boot", action="store_true",
#                         help="Skip the SSH boot step (useful when players are already running)")
#     parser.add_argument("--port", type=int, default=6789,
#                         help="UDP port on each minicar (default 6789)")
#     parser.add_argument("--cam", type=int, default=0)
#     parser.add_argument("--calibrate", action="store_true")
#     parser.add_argument("--calib-file", type=str, default=None, metavar="PATH")


#     parser.add_argument("--board", type=str, default="chess",
#                         choices=["chess", "charuco"])
#     parser.add_argument("--chess-size", type=int, nargs=2, default=[8, 6],
#                         metavar=("COLS", "ROWS"))
#     parser.add_argument("--square-mm", type=float, default=38.0)
#     parser.add_argument("--marker-mm", type=float, default=18.0)
#     parser.add_argument("--scenario", type=str, default="S1",
#                         help="Experiment scenario label (default: S1)")
#     parser.add_argument("--policy", type=str, default="cooperative",
#                         help="Driving policy label (default: cooperative)")
#     parser.add_argument("--video-name", type=str, default=None, metavar="NAME",
#                         help="Base name for the recorded video (without .mp4). Default: output.mp4. " \
#                         "Example: --video-name exp1_run3")
#     parser.add_argument("--run-name", type=str, default=None, metavar="NAME",
#                         help="Optional sub-folder under .exp/{scenario}/. "
#                              "Log saved to .exp/{scenario}/{run-name}/file.json. "
#                              "Omit to place the log directly in .exp/{scenario}/.")
#     parser.add_argument("--repetition", type=int, default=1, metavar="N",
#                         help="Repetition index used in the log filename "
#                              "(e.g. --repetition 3 -> exp-log-S1-r3-fov-calib-policy.json). "
#                              "No counter file is read or written.")
#     parser.add_argument("--fov", type=str, default=None, metavar="DEG",
#                         help="Diagonal FOV of the overhead camera in degrees, e.g. --fov 90 or "
#                              "--fov 78.  Sets the dfov tag used in log filenames and the JSON "
#                              "meta block.  When --calib-file is given the FOV is also inferred "
#                              "from the filename (e.g. calib-90deg.npz → 90), but --fov always "
#                              "takes precedence and works even without a calibration file.")
#     args = parser.parse_args()

#     if not args.cars:
#         print("No minicars selected. Use -n <id1> <id2> ..."); sys.exit(0)

#     # Derive IP for every car from its ID (matches player_launcher.py logic)
#     car_ips = {cid: _car_ip(cid) for cid in args.cars}

#     cap, out = init_camera(args.cam, video_name=args.video_name)
#     if not cap.isOpened():
#         print("[error] Cannot open camera."); sys.exit(1)

#     _startup_calibration(args, cap)

#     # ── Apply CLI FOV override ────────────────────────────────────────────────
#     # --fov always wins over the filename-inferred _DFOV so it can be
#     # used independently of calibration (e.g. RAW mode, or when the .npz
#     # filename does not follow the calib-{N}deg.npz naming convention).
#     if args.fov is not None:
#         _DFOV = args.fov.strip().rstrip("°").rstrip("deg")
#         print(f"[fov] Diagonal FOV set to {_DFOV}° via --fov flag")
#     elif _DFOV is None:
#         _DFOV = "unknown"   # ensures log filename is always well-formed
#         print("[fov] No FOV known — using 'unknown' tag in log filename. "
#               "Pass --fov <deg> to set it explicitly.")

#     # Apply CLI experiment metadata
#     LOG_SCENARIO   = args.scenario
#     LOG_POLICY     = args.policy
#     DEFAULT_POLICY = args.policy
#     LOG_RUN_NAME   = args.run_name   # optional sub-folder name
#     LOG_REPETITION = args.repetition  # naming only — no counter file written

#     # ── Boot each minicar in its own daemon thread ────────────────────────────
#     import threading as _threading
#     boot_threads: dict = {}
#     if not args.no_boot:
#         for cid in args.cars:
#             _ip = car_ips[cid]
#             t = _threading.Thread(
#                 target=boot,
#                 args=(_ip, _SSH_USERNAME, _SSH_PASSWORD, _LAUNCH_SCRIPT),
#                 daemon=True,
#                 name=f"boot-car{cid}"
#             )
#             t.start()
#             boot_threads[cid] = t
#             print(f"[boot] Car {cid} boot thread started ({_SSH_USERNAME}@{_ip})")
#     else:
#         print("[boot] --no-boot set, skipping SSH launch")

#     # ── One UDP socket per car ────────────────────────────────────────────────
#     sockets: dict = {}
#     _clean_flags: dict = {cid: False for cid in args.cars}  # per-car clean/stop flag
#     for cid in args.cars:
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         sockets[cid] = s
#         print(f"[net] Car {cid} → {car_ips.get(cid, '(no IP)')}:{args.port}")

#     def _send(car_id: int, servo: float, motor: float,
#               clean: bool = False) -> None:
#         """Pack and send one actuation packet to a minicar.

#         Packet layout (little-endian):
#             float  motor   — normalised motor speed  [0, 1]
#             float  servo   — normalised servo angle  [-1, 1]
#             float  0.0     — reserved / padding
#             bool   clean   — True signals the car to stop/clean up
#         """
#         _ip = car_ips.get(car_id)
#         if _ip is None:
#             return
#         pkt = bytearray(struct.pack('<fff?',
#                                     float(motor),
#                                     float(servo),
#                                     0.0,
#                                     bool(clean)))
#         try:
#             sockets[car_id].sendto(pkt, (_ip, args.port))
#         except OSError as exc:
#             print(f"[net] Car {car_id} send error: {exc}")

#     # ── HUD panel for multi-car status bar (bottom strip) ────────────────────
#     def _draw_status_bar(frame: np.ndarray,
#                          results: dict,   # car_id → (servo, motor)
#                          h: int, w: int) -> None:
#         """
#         Draw a translucent status bar at the bottom of the frame.
#         One column per car: Car ID | Srv | Mtr | Ln | Lat | Policy
#         """
#         FONT   = cv2.FONT_HERSHEY_SIMPLEX
#         SCALE  = 0.38
#         THICK  = 1
#         LH     = 14
#         PAD    = 5
#         n_cars = len(results)
#         if n_cars == 0:
#             return

#         col_w  = max(140, w // max(n_cars, 1))
#         bar_h  = LH * 4 + PAD * 2
#         y0     = h - bar_h

#         # Semi-transparent dark background
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (0, y0), (w, h), (15, 15, 15), -1)
#         cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

#         for col, (cid, (servo, motor)) in enumerate(sorted(results.items())):
#             x0 = col * col_w + PAD
#             ls  = _lane_state.get(cid, {})
#             vel = _vel_cache.get(cid, {})
#             pid = _spd_pid.get(cid, {})
#             lat = round(pid.get("e_prev", 0.0) * 100, 1)   # speed error proxy
#             lane_n = ls.get("lane", "?")
#             ovt    = "OVT" if ls.get("overtaking") else "NRM"
#             wait   = "W" if _pp_waiting.get(cid, False) else " "
#             policy = DRIVING_POLICY.get(cid, DEFAULT_POLICY)[:4].upper()
#             tier   = _car_occlusion_state.get(cid, {}).get("tier", "??")

#             lines = [
#                 f"-- Car {cid} {policy}{wait} [{tier[:2]}] --",
#                 f"Srv {servo:+.2f}  Mtr {motor:.2f}",
#                 f"Ln{lane_n} {ovt}",
#                 f"EKF v={ekf.get('v', 0.0):.1f}px/s",
#             ]
#             for row, txt in enumerate(lines):
#                 cy = y0 + PAD + (row + 1) * LH - 2
#                 cv2.putText(frame, txt, (x0, cy),
#                             FONT, SCALE, (220, 220, 220), THICK, cv2.LINE_AA)

#     # ── Main loop ─────────────────────────────────────────────────────────────
#     # Architecture:
#     #   1. Capture one frame from the overhead camera.
#     #   2. Call run(frame, cid) for each car in sequence.
#     #      run() draws its own annotations (ArUco, EKF ellipse, lookahead,
#     #      per-car HUD) onto its own copy of the frame, then the annotated
#     #      copies are merged back into a single shared canvas by overlaying
#     #      the car-specific marks onto the base frame.
#     #   3. A shared status bar is drawn at the bottom.
#     #   4. Only ONE window is shown: "Minicar Testbed".
#     #
#     # Note: run() internally calls detect_all_markers() which re-detects on
#     # each call. To avoid redundant detection per car, we perform the shared
#     # perception step once (detect + EKF for all cars) before the per-car
#     # control step. The per-car HUD and lookahead overlays are drawn onto
#     # a single shared canvas so all annotations are visible at once.

#     WINDOW = "Minicar Testbed"
#     cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

#     print(f"[ctrl] Controlling cars: {args.cars}  |  press ESC to quit")

#     try:
#         while True:
#             ok, raw_frame = cap.read()
#             if not ok:
#                 continue
#             raw_frame = undistort_frame(raw_frame)
#             h, w = raw_frame.shape[:2]

#             # ── One-shot extrinsics estimation ────────────────────────────
#             # Runs on the first frame where enough track markers are visible.
#             # Pixel positions come from _last_track_markers (all three rings
#             # — inner 0-3, middle 4-7, outer 8-11) already cached by
#             # detect_all_markers(). No pre-measured image coords needed.
#             # JSON format (--world-pts file): all marker IDs you have measured.
#             #   {"0":[x,y,0],"1":[x,y,0],...,"11":[x,y,0]}
#             # Values are real-world positions in metres (z=0, flat track plane).
#             # More markers → better accuracy. Minimum 4 required.

#             # ── Shared canvas: starts as the raw undistorted frame ────────────
#             # Each run() call annotates onto a copy, then we merge the
#             # car-specific overlay pixels (non-background marks) back in.
#             canvas = raw_frame.copy()

#             # ── Shared perception: draw lanes + obstacles ONCE per frame ─────
#             # detect_lanes uses the marker cache populated by the first
#             # detect_all_markers() call inside run().  We call it here too so
#             # lanes are visible even before the first run() call completes,
#             # and — critically — so that draw_lanes() is not called inside each
#             # run() call (which would cause the diff-mask merge to zero out
#             # the guide lines drawn by earlier cars in the sequence).
#             _shared_lanes = _last_lanes  # cached from the previous frame
#             draw_lanes(canvas, _shared_lanes)

#             results:    dict = {}   # car_id → (servo, motor)
#             cars_data:  dict = {}   # car_id → car_log_data for this frame

#             for cid in args.cars:
#                 # Pass raw_frame for ArUco detection (undecorated pixels) but
#                 # seed the drawing buffer from canvas so per-car annotations
#                 # are layered on top of the already-drawn lane fills.
#                 servo, motor, annotated, car_log_data = run(raw_frame, cid, draw_frame=canvas.copy())
#                 _send(cid, servo, motor)
#                 _send(cid, servo, motor)
#                 results[cid] = (servo, motor)
#                 if car_log_data is not None:
#                     cars_data[cid] = car_log_data

#                 # Merge annotated frame back into canvas:
#                 # Pixels that changed from the input → this car's annotations.
#                 # Use a simple diff-based overlay so all cars appear on one frame.
#                 diff_mask = np.any(annotated != canvas, axis=2)
#                 canvas[diff_mask] = annotated[diff_mask]

#             # One log entry per frame covering all cars
#             if cars_data:
#                 # Build a merged cars dict for pairwise distance computation.
#                 # Prefer live identify_cars output; fall back to EKF ghost pose.
#                 _merged_cars: dict = {}
#                 for _cid, _cd in cars_data.items():
#                     _px, _py = _cd["pose"][0], _cd["pose"][1]
#                     _cl   = _cd["lane"]
#                     _cref = _last_lanes.get(f"lane{_cl}_ref", [])
#                     _sidx = project_onto_curve((_px, _py), _cref) if _cref else None
#                     _merged_cars[_cid] = {
#                         "midpoint":    (_px, _py),
#                         "lane":        _cl,
#                         "segment_idx": _sidx,
#                     }
#                 _log_entries.append(
#                     _build_log_entry(time.time(), _cycle_counter,
#                                      cars_data, _merged_cars))

#             # ── Shared status bar at the bottom ──────────────────────────────
#             _draw_status_bar(canvas, results, h, w)

#             # ── FPS counter (top-right corner) ────────────────────────────────
#             fps_text = f"{len(args.cars)} car(s)  |  k={_cycle_counter}"
#             tw, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0], None
#             cv2.putText(canvas, fps_text, (w - tw[0] - 8, 18),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

#             cv2.imshow(WINDOW, canvas)
#             # out.write(canvas) # to save the annotated video, uncomment this line
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break

#     finally:
#         # Send a clean=True stop packet to every car before closing sockets
#         for cid in args.cars:
#             _send(cid, servo=0.0, motor=0.0, clean=True)
#         for s in sockets.values():
#             s.close()
#         out.release()
#         cap.release()
#         cv2.destroyAllWindows()
#         save_log()
#         print("[ctrl] Shutdown complete.")
