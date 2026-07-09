"""
player_launcher.py — Player-mode minicar launcher.

Architecture
────────────
• One camera producer thread fills `latest_frame` continuously.
• One worker thread per car (Player.transfer_data) calls the controller's
  listen() at ~100 Hz, which delegates to auto_control.run() for
  compute-only perception/planning (no drawing), and fires UDP packets
  to the minicar.
• The main (display) thread runs at ~50 Hz and owns ALL drawing:
    – lane-fill drawing (once per frame)
    – obstacle circle drawing (once per frame)
    – per-car overlay drawing (guide line, lookahead target, heading
      arrow, marker dots) drawn in a fixed car-id order from each car's
      draw_data dict — no diff-mask merge, no per-thread canvas
    – per-car HUD box, one bordered box per connected car
    – log entry assembly  (_build_log_entry / save_log)
    – optional video recording via --video-name

CLI interface is a superset of auto_control.py:
  -n / --cars          car IDs
  --no-boot            skip SSH launch
  --port               UDP port
  --cam                camera index
  --calibrate          run calibration wizard
  --calib-file PATH    load existing calibration
  --board / --chess-size / --square-mm / --marker-mm
  --scenario           experiment label (default S1)
  --policy             driving policy (default cooperative)
  --video-name NAME    record annotated video to NAME.mp4
  --run-name NAME      log sub-folder under .exp/{scenario}/
  --fov DEG            diagonal FOV tag for log filenames
  -c / --controller    keyboard | joystick
  -m / --mode          0=manual  1=semi-autonomous  2=autonomous
"""

import argparse
import os
import socket
import struct
import sys
import time

import cv2
import keyboard
import numpy as np
from threading import Thread, Lock

from player import binds
from player import controllers
import auto_control as _ac

# ── Shared state ──────────────────────────────────────────────────────────────
frame_lock      = Lock()
latest_frame    = None           # raw camera frame      (producer → workers)

result_lock     = Lock()
car_results     = {}             # {car_id: draw_data dict or None}
car_log_results = {}             # {car_id: (servo, motor, car_log_data)}


# ── Camera producer ───────────────────────────────────────────────────────────
def camera_producer(capture):
    global latest_frame
    while True:
        ret, frame = capture.read()
        if not ret or frame is None:
            continue
        with frame_lock:
            latest_frame = frame.copy()


def get_latest_frame():
    with frame_lock:
        return latest_frame.copy() if latest_frame is not None else None


# ── Player ────────────────────────────────────────────────────────────────────
class Player:
    """One minicar player — holds the controller and the UDP sender."""

    def __init__(self, number: int, controller: str, mode: int,
                 ideal_speed: float = 0.55, max_angle: float = 0.5):
        from auto_control import _car_ip          # canonical IP derivation

        self.car_number  = number
        self.ip          = _car_ip(number)
        # last 1 or 2 digits of the IP — used for log-file copy
        self.copy_file_nb = self.ip[-1:] if number in range(0, 10) else self.ip[-2:]
        self.username    = 'cpslab1'
        self.password    = 'cpslab1'
        self.remote_port = 6789

        if controller == 'keyboard':
            print('Keyboard connected!')
            self.controller = controllers.Keyboard(mode, number, ideal_speed, max_angle)
        elif controller == 'joystick':
            print('Joystick connected!')
            self.controller = controllers.Joystick(mode, number, ideal_speed, max_angle)
        else:
            raise ValueError(f'Unknown controller type: {controller!r}')

    def boot(self):
        from auto_control import boot as _boot, _SSH_USERNAME, _SSH_PASSWORD, _LAUNCH_SCRIPT
        _boot(self.ip, _SSH_USERNAME, _SSH_PASSWORD, _LAUNCH_SCRIPT)

    def copy_file(self):
        os.system(
            f'pscp -pw {self.password} {self.username}@{self.ip}:'
            f'car-{self.copy_file_nb}-log.txt ./firmware/console_prints')

    def ping(self):
        return os.system(f'ping -n 1 -w 200 {self.ip} | find "Reply"') if os.name == 'nt' \
            else os.system(f'ping -c 1 -W 1 {self.ip} | grep -q "1 received"')

    def transfer_data(self, sock: socket.socket, port: int):
        """Worker thread: compute-only listen() → send UDP at ~100 Hz.

        This thread NEVER draws. It calls the controller's listen(), which
        delegates to auto_control.run() for perception/planning and returns
        lightweight overlay geometry in ctrl.draw_data. The display thread
        (main loop below) is the only place that ever calls cv2.* drawing
        functions, using ctrl.draw_data from every connected car.
        """
        global car_results, car_log_results

        Thread(target=self.boot, daemon=True).start()
        print(f'[car {self.car_number}] Sending...')

        ctrl = self.controller
        cid  = self.car_number

        while ctrl.running:
            raw = get_latest_frame()
            if raw is None:
                time.sleep(0.01)   # camera not ready yet — spin
                continue           # ← skip this entire tick

            ctrl.listen(captured_frame=raw)

            with result_lock:
                car_results[cid]     = getattr(ctrl, "draw_data", None)
                car_log_results[cid] = (ctrl.speed, ctrl.angle,
                                        ctrl.last_log_data)

            pkt = bytearray(struct.pack(
                '<fff?', ctrl.speed, ctrl.angle, ctrl.brightness, ctrl.clean))
            try:
                sock.sendto(pkt, (self.ip, port))
            except OSError as exc:
                print(f'[car {cid}] UDP send error: {exc}')

            time.sleep(1. / 100.)

        time.sleep(0.85)
        print(f'[car {self.car_number}] Copying log file...')
        self.copy_file()


# ── Player-run orchestrator ───────────────────────────────────────────────────
def players_run(car_list: list, sock: socket.socket, args,
                cap: cv2.VideoCapture, out: cv2.VideoWriter):
    """Initialise, ping, and start all cars; then own the display + log loop.

    Parameters
    ----------
    car_list : list of car IDs (ints)
    sock     : UDP socket (one shared socket is fine for player mode)
    args     : parsed argparse namespace
    cap      : VideoCapture returned by init_camera
    out      : VideoWriter returned by init_camera (written when --video-name given)
    """
    # Import shared perception + log helpers from auto_control.
    # Using auto_control's own module-level state means the player launcher
    # shares the same EKF, lane-cache, log-entries list, and save_log() as the
    # standalone auto_control runner — zero duplication.
    from auto_control import (
        draw_lanes,
        draw_obstacles,
        draw_marker_debug,
        _build_log_entry,
        _log_entries,
        project_onto_curve,
        save_log,
        undistort_frame,
        _startup_calibration,
    )

    # ── Apply CLI metadata into auto_control module state ────────────────────
    _ac.LOG_SCENARIO   = args.scenario
    _ac.LOG_POLICY     = args.policy
    _ac.DEFAULT_POLICY = args.policy
    _ac.LOG_RUN_NAME   = args.run_name
    _ac.LOG_REPETITION = args.repetition   # naming only — no counter file written

    # ── Calibration (--calibrate / --calib-file) ──────────────────────────────
    _startup_calibration(args, cap)

    # ── FOV tag ──────────────────────────────────────────────────────────────
    if args.fov is not None:
        _ac._DFOV = args.fov.strip().rstrip("°").rstrip("deg")
        print(f"[fov] Diagonal FOV set to {_ac._DFOV}° via --fov flag")
    elif _ac._DFOV is None:
        _ac._DFOV = "unknown"
        print("[fov] No FOV known — using 'unknown' tag. Pass --fov <deg> to set it.")

    # ── Ping & boot cars ─────────────────────────────────────────────────────
    key_bind     = binds.KeyboardBinds()
    players      = {}
    threads      = {}
    unresponsive = []

    # --no-boot: register all requested car IDs immediately so their ArUco
    # markers are never injected as obstacles (visualisation-only mode).
    # We do this before the ping loop so even cars that do not respond are
    # treated as active vehicles on the track, not as static obstacles.
    if args.no_boot:
        for _nb_cid in car_list:
            _ac.ACTIVE_CAR_IDS.add(int(_nb_cid))
        print("[no-boot] All requested car IDs pre-registered as active "
              f"({sorted(_ac.ACTIVE_CAR_IDS)}) — markers not treated as obstacles")

    print('Initializing…')
    for cid in car_list:
        cid = int(cid)
        p   = Player(cid, args.controller, args.mode)
        if args.no_boot:
            print("[boot] --no-boot set, skipping SSH launch")
            p.boot = lambda: None    # replace with no-op

        if p.ping() == 0 or args.no_boot:
            print(f'[Minicar {cid}] RESPONSIVE')
            players[cid] = p
            _ac.ACTIVE_CAR_IDS.add(cid)  # register before worker starts
            t = Thread(target=p.transfer_data,
                    args=(sock, args.port), daemon=True)
            threads[cid] = t
            t.start()      
        else:
            print(f'[Minicar {cid}] UNRESPONSIVE')
            unresponsive.append(cid)

    if unresponsive and not args.no_boot:
        print(f'Minicar(s) {unresponsive} unresponsive — press R to retry or Q to quit.')
        while True:
            if keyboard.is_pressed(key_bind.stop):
                print("Quitting the connection!")
                sock.close(); sys.exit(0)
            elif keyboard.is_pressed(key_bind.retry):
                print("\nRetry connecting to unresponsive minicar(s)...")
                # Recursive retry for the unresponsive subset only
                players_run(unresponsive, sock, args, cap, out)
                return
            time.sleep(0.1)
    else:
        # if unresponsive and args.no_boot:
        #     print("[boot] --no-boot set, skipping SSH launch")
        #     for p in players.values():
        #         p.boot = lambda: None    # replace with no-op
        print('All minicars ready!')

    # # ── Boot non-boot flag handled per-car inside Player.boot() ──────────────
    # # If --no-boot was set, skip SSH launch by not calling boot() at all.
    # # We achieve this by patching the boot method on already-started Players.
    # if args.no_boot:
    #     print("[boot] --no-boot set, skipping SSH launch")
    #     for p in players.values():
    #         p.boot = lambda: None    # replace with no-op

    # ── Camera producer thread ────────────────────────────────────────────────
    Thread(target=camera_producer, args=(cap,), daemon=True).start()
    print('\nWaiting for first camera frame...')
    while get_latest_frame() is None:
        time.sleep(0.05)
    print('Camera streaming!\n')

    # ── Display + log loop (~50 Hz) ───────────────────────────────────────────
    record = args.video_name is not None      # only write frames when requested
    WINDOW = "Minicar Player View"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(f"[ctrl] Controlling cars: {args.cars}  |  press ESC to quit"
          + (f"  |  recording → {args.video_name}.mp4" if record else ""))

    try:
        while True:
            raw = get_latest_frame()
            if raw is None:
                time.sleep(0.02)
                continue

            # Undistort when a calibration file was loaded
            raw = undistort_frame(raw)
            h, w = raw.shape[:2]

            # 1. Start canvas from undistorted raw frame
            canvas = raw.copy()

            # 2. Draw lane fills ONCE — avoids repeated semi-transparent stacking
            from auto_control import _last_lanes as _ll
            draw_lanes(canvas, _ll)

            # 3. Draw shared obstacle circles ONCE, from auto_control's own
            # obstacle cache — no per-car obstacle drawing anymore since
            # run() no longer draws at all.
            draw_obstacles(canvas, getattr(_ac, "_obstacle_positions", []))

            # 3b. Draw ArUco debug rectangles ONCE per frame. detect_all_markers()
            # runs once per connected car inside run(), but always against the
            # same physical camera frame, so its cached geometry is identical
            # across cars within a frame — drawing it here (instead of inside
            # run()) avoids redundant/overlapping rectangle draws per car.
            draw_marker_debug(canvas)

            # 4. Snapshot both per-car draw_data and log payloads atomically
            with result_lock:
                draws_snap = dict(car_results)
                log_snap   = dict(car_log_results)

            # 5. Draw every connected car's overlay in a fixed, deterministic
            # order (sorted by car id) — replaces the old diff-mask merge.
            # Each worker thread never touches the canvas; only this thread
            # ever calls cv2.* drawing functions.
            active_ids = sorted(cid for cid, dd in draws_snap.items() if dd is not None)
            for cid in active_ids:
                _draw_car_overlay(canvas, draws_snap[cid])
            # HUD boxes drawn in a second pass so they always sit on top of
            # every car's guide line / markers, regardless of draw order above.
            for idx, cid in enumerate(active_ids):
                _draw_hud_box(canvas, draws_snap[cid], idx, h)

            # 6. Assemble one multi-car log entry per display frame
            #    Mirrors the auto_control main loop log block exactly.
            entry = _ac.build_frame_log_entry(log_snap, _ll, _ac._cycle_counter)
            if entry is not None:
                _log_entries.append(entry)

            # 7. FPS / counter overlay (top-right, same style as auto_control)
            fps_text = f"{len(args.cars)} car(s)  |  k={_ac._cycle_counter}"
            tw = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            cv2.putText(canvas, fps_text, (w - tw[0] - 8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
                        cv2.LINE_AA)

            # 8. Optional video recording
            if record:
                out.write(canvas)

            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) & 0xFF == key_bind.exit_ASCII:   # ESC
                print('[player] ESC — shutting down.')
                break

            time.sleep(0.02)   # ~50 Hz

    finally:
        # Send clean=True stop packet to every car
        for cid in list(players.keys()):
            pkt = bytearray(struct.pack('<fff?', 0.0, 0.0, 0.0, True))
            try:
                sock.sendto(pkt, (players[cid].ip, args.port))
            except OSError:
                pass
        if record:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
        save_log()
        print('[player] Shutdown complete.')

    
def _draw_car_overlay(canvas, dd):
    """Draw one car's marker dots, heading arrow, and guide line/lookahead.

    `dd` is the draw_data dict returned by auto_control.run() for this car.
    Mirrors the pixel output of the old draw_car()/draw_guide_line() calls
    that used to run inside run() itself.
    """
    if dd is None:
        return
    rear = dd.get("rear_pt")
    if rear is None:
        return
    tier = dd.get("occlusion_tier", "BOTH_VISIBLE")
    tier_colour = {
        "BOTH_VISIBLE":  (0, 255, 255),
        "FRONT_ONLY":    (0, 165, 255),
        "REAR_ONLY":     (0, 255, 255),
        "BOTH_OCCLUDED": (0, 220, 255),
    }.get(tier, (0, 255, 255))
    cv2.circle(canvas, rear, 5, tier_colour, -1)

    front = dd.get("front_pt")
    heading = dd.get("heading", (0.0, 0.0))
    if front and front != rear and heading != (0.0, 0.0):
        cv2.arrowedLine(canvas, rear, front, (0, 255, 0), 2, tipLength=0.3)

    raw_rear = dd.get("raw_rear_pt", rear)
    cv2.circle(canvas, raw_rear, 4, (255, 255, 255), -1)
    label = str(dd["car_id"]) + ("F!" if dd.get("using_front_fallback") else "")
    cv2.putText(canvas, label, (raw_rear[0] + 6, raw_rear[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    guide = dd.get("guide_line")
    if guide:
        ref_pose, target = guide
        colour = (0, 0, 220) if dd.get("obstacle_close") else (255, 255, 255)
        cv2.circle(canvas, target, 9, colour, 2)
        cv2.circle(canvas, target, 3, (255, 255, 255), -1)
        cv2.line(canvas, ref_pose, target, (160, 34, 201), 2)

    iz = dd.get("interaction_zone")
    if iz is not None:
        from auto_control import draw_interaction_zone
        draw_interaction_zone(canvas, iz["car_pos"], iz["tangent"],
                              iz["left_normal"], iz["obs_state"])

    p = dd.get("path_point")
    if p is not None:
        cv2.circle(canvas, p, 4, (255, 255, 0), -1)
        op = dd.get("obstacle_point")
        if op is not None:
            cv2.line(canvas, p, op, (0, 0, 255), 1)


def _draw_hud_box(canvas, dd, idx, frame_h):
    """Draw one bordered HUD box for one car, offset horizontally by `idx`.

    Multiple connected cars each get their own box side-by-side, so no
    car's status text can overwrite or crowd another's.
    """
    if dd is None or not dd.get("hud_lines"):
        return
    FONT, SCALE, THICK, LH, PAD, GAP = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1, 14, 6, 10
    lines = dd["hud_lines"]
    maxw  = max(cv2.getTextSize(l, FONT, SCALE, THICK)[0][0] for l in lines)
    box_w, box_h = maxw + PAD * 2, LH * len(lines) + PAD * 2
    x0 = 8 + idx * (box_w + GAP)
    y0 = 8
    ov = canvas.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, canvas)
    border_colour = dd.get("color", (200, 200, 200))
    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), border_colour, 1)
    for i, txt in enumerate(lines):
        _txt_colour = (0, 60, 230) if txt.startswith("!!") else (230, 230, 230)
        cv2.putText(canvas, txt, (x0 + PAD, y0 + PAD + (i + 1) * LH - 2),
                    FONT, SCALE, _txt_colour, THICK, cv2.LINE_AA)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Player-mode minicar launcher — multi-car")

    # ── car selection & network ───────────────────────────────────────────────
    parser.add_argument("-n", "--cars", nargs="+", type=int, default=None,
                        help="Car IDs to control, e.g. -n 1 2 3")
    parser.add_argument("--no-boot", action="store_true",
                        help="Skip the SSH boot step (useful when players are already running)")
    parser.add_argument("--port", type=int, default=6789,
                        help="UDP port on each minicar (default 6789)")

    # ── camera ────────────────────────────────────────────────────────────────
    parser.add_argument("--cam", type=int, default=0)

    # ── calibration ───────────────────────────────────────────────────────────
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calib-file", type=str, default=None, metavar="PATH")
    parser.add_argument("--board", type=str, default="chess",
                        choices=["chess", "charuco"])
    parser.add_argument("--chess-size", type=int, nargs=2, default=[8, 6],
                        metavar=("COLS", "ROWS"))
    parser.add_argument("--square-mm", type=float, default=38.0)
    parser.add_argument("--marker-mm", type=float, default=18.0)

    # ── experiment metadata ───────────────────────────────────────────────────
    parser.add_argument("--scenario", type=str, default="S1",
                        help="Experiment scenario label (default: S1)")
    parser.add_argument("--policy", type=str, default="cooperative",
                        help="Driving policy label (default: cooperative)")
    parser.add_argument("--fov", type=str, default=None, metavar="DEG",
                        help="Diagonal FOV of the overhead camera in degrees, e.g. --fov 90 or "
                             "--fov 78.  Sets the dfov tag used in log filenames and the JSON "
                             "meta block.  When --calib-file is given the FOV is also inferred "
                             "from the filename (e.g. calib-90deg.npz → 90), but --fov always "
                             "takes precedence and works even without a calibration file.")
    parser.add_argument("--run-name", type=str, default=None, metavar="NAME",
                        help="Optional sub-folder under .exp/{scenario}/. "
                             "Log saved to .exp/{scenario}/{run-name}/file.json. "
                             "Omit to place the log directly in .exp/{scenario}/.",
                        dest="run_name")
    parser.add_argument("--repetition", type=int, default=1, metavar="N",
                        help="Repetition index used in the log filename "
                             "(e.g. --repetition 3 → exp-log-S1-r3-…json). "
                             "No counter file is read or written.")

    # ── recording ─────────────────────────────────────────────────────────────
    parser.add_argument("--video-name", type=str, default=None, metavar="NAME",
                        help="Base name for the recorded video (without .mp4). "
                             "Default: no recording.  Example: --video-name exp1_run3",
                        dest="video_name")

    # ── controller / mode (player-specific) ──────────────────────────────────
    parser.add_argument("-c", "--controller", type=str, default="keyboard",
                        help="Controller type: keyboard | joystick")
    parser.add_argument("-m", "--mode", type=int, default=2,
                        help="Operation mode: 0=manual  1=semi-autonomous  2=autonomous")

    args = parser.parse_args()

    if not args.cars:
        print("No minicar selected. Use -n <id> …")
        sys.exit(0)

    cap, out = _ac.init_camera(args.cam, video_name=args.video_name)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        players_run(args.cars, sock, args, cap, out)
    finally:
        _ac.ACTIVE_CAR_IDS.clear()   # deregister all cars on exit
        sock.close()
