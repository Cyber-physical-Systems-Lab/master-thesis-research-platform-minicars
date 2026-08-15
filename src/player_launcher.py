"""player_launcher.py -- Player-mode minicar launcher.

Threading model:
  - One camera producer thread fills `latest_frame` continuously.
  - One worker thread per car (Player.transfer_data) calls the controller's
    listen() at ~100 Hz (delegates to auto_control.run() for compute-only
    perception/planning, no drawing) and sends TCP packets to the minicar.
  - The main thread runs the ~50 Hz display/log loop and owns ALL drawing
    (lane fills, obstacles, per-car overlays/HUD) plus log-entry assembly
    and optional video recording.

CLI is a superset of auto_control.py: -n/--cars, --no-boot, --port, --cam,
--calibrate, --calib-file, --board/--chess-size/--square-mm/--marker-mm,
--scenario, --policy, --fov, --run-name, --repetition, --video-name,
-c/--controller, -m/--mode.
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

from src.player import binds, controllers
import src.auto_control as _ac

# ── Shared state ─────────────────────────────────────────────────────────
frame_lock = Lock()
latest_frame = None  # raw camera frame (producer -> workers)

result_lock = Lock()
car_results = {}      # {car_id: draw_data dict or None}
car_log_results = {}  # {car_id: (servo, motor, car_log_data)}

PKT_FMT = "<fff?"  # servo/motor commands over TCP: speed, angle, brightness, clean

# ── Camera producer ──────────────────────────────────────────────────────

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

# ── Player ────────────────────────────────────────────────────────────────

class Player:
    """One minicar player -- holds the controller and the TCP sender."""

    def __init__(self, number: int, controller: str, mode: int,
                 ideal_speed: float = 0.55, max_angle: float = 0.5):
        from src.auto_control import _car_ip
        self.car_number = number
        self.ip = _car_ip(number)
        self.copy_file_nb = self.ip[-1:] if number in range(0, 10) else self.ip[-2:]
        self.username = "cpslab1"
        self.password = "cpslab1"
        self.remote_port = 6789

        if controller == "keyboard":
            print("Keyboard connected!")
            self.controller = controllers.Keyboard(mode, number, ideal_speed, max_angle)
        elif controller == "joystick":
            print("Joystick connected!")
            self.controller = controllers.Joystick(mode, number, ideal_speed, max_angle)
        else:
            raise ValueError(f"Unknown controller type: {controller!r}")

    def boot(self):
        from src.auto_control import boot as _boot, _SSH_USERNAME, _SSH_PASSWORD, _LAUNCH_SCRIPT
        _boot(self.ip, _SSH_USERNAME, _SSH_PASSWORD, _LAUNCH_SCRIPT)

    def copy_file(self):
        os.system(f"pscp -pw {self.password} {self.username}@{self.ip}:"
                  f"car-{self.copy_file_nb}-log.txt ./src/minicar_firmware/console_prints")

    def ping(self):
        return (os.system(f'ping -n 1 -w 200 {self.ip} | find "Reply"') if os.name == "nt"
                else os.system(f'ping -c 1 -W 1 {self.ip} | grep -q "1 received"'))

    def transfer_data(self, sock: socket.socket, port: int):
        """Worker thread: compute-only listen() -> send TCP at ~100 Hz. Never draws."""
        global car_results, car_log_results
        Thread(target=self.boot, daemon=True).start()
        print(f"[car {self.car_number}] Sending...")

        ctrl, cid = self.controller, self.car_number
        while ctrl.running:
            raw = get_latest_frame()
            if raw is None:
                time.sleep(0.01)  # camera not ready yet
                continue

            ctrl.listen(captured_frame=raw)
            with result_lock:
                car_results[cid] = getattr(ctrl, "draw_data", None)
                car_log_results[cid] = (ctrl.speed, ctrl.angle, ctrl.last_log_data)

            pkt = bytearray(struct.pack(PKT_FMT, ctrl.speed, ctrl.angle, ctrl.brightness, ctrl.clean))
            try:
                sock.sendto(pkt, (self.ip, port))
            except OSError as exc:
                print(f"[car {cid}] TCP send error: {exc}")
            time.sleep(1.0 / 100.0)

        time.sleep(0.85)
        print(f"[car {self.car_number}] Copying log file...")
        self.copy_file()

# ── Overlay drawing (main thread only) ───────────────────────────────────

_TIER_COLOUR = {
    "BOTH_VISIBLE": (0, 255, 255), "FRONT_ONLY": (0, 165, 255),
    "REAR_ONLY": (0, 255, 255), "BOTH_OCCLUDED": (0, 220, 255),
}

def _draw_car_overlay(canvas, dd):
    """Draw one car's marker dots, heading arrow, and guide line/lookahead from its draw_data dict."""
    if dd is None:
        return
    rear = dd.get("rear_pt")
    if rear is None:
        return

    tier = dd.get("occlusion_tier", "BOTH_VISIBLE")
    cv2.circle(canvas, rear, 5, _TIER_COLOUR.get(tier, (0, 255, 255)), -1)

    front = dd.get("front_pt")
    heading = dd.get("heading", (0.0, 0.0))
    if front and front != rear and heading != (0.0, 0.0):
        cv2.arrowedLine(canvas, rear, front, (0, 255, 0), 2, tipLength=0.3)

    raw_rear = dd.get("raw_rear_pt", rear)
    cv2.circle(canvas, raw_rear, 4, (255, 255, 255), -1)
    if dd.get("using_front_fallback"):
        label = f"{dd['car_id']}F!"
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
        from src.auto_control import draw_interaction_zone
        draw_interaction_zone(canvas, iz["car_pos"], iz["tangent"], iz["left_normal"], iz["obs_state"])

    p = dd.get("path_point")
    if p is not None:
        cv2.circle(canvas, p, 4, (255, 255, 0), -1)
        op = dd.get("obstacle_point")
        if op is not None:
            cv2.line(canvas, p, op, (0, 0, 255), 1)

def _draw_hud_box(canvas, dd, idx, frame_h):
    """Draw one bordered HUD box for a car, offset horizontally by idx so multiple cars don't overlap."""
    if dd is None or not dd.get("hud_lines"):
        return
    FONT, SCALE, THICK, LH, PAD, GAP = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1, 14, 6, 10
    lines = dd["hud_lines"]
    max_w = max(cv2.getTextSize(l, FONT, SCALE, THICK)[0][0] for l in lines)
    box_w, box_h = max_w + PAD * 2, LH * len(lines) + PAD * 2
    x0, y0 = 8 + idx * (box_w + GAP), 8

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), dd.get("color", (200, 200, 200)), 1)
    for i, txt in enumerate(lines):
        colour = (0, 60, 230) if txt.startswith("!!") else (230, 230, 230)
        cv2.putText(canvas, txt, (x0 + PAD, y0 + PAD + (i + 1) * LH - 2), FONT, SCALE, colour, THICK, cv2.LINE_AA)

# ── Orchestration ─────────────────────────────────────────────────────────

def players_run(car_list: list, sock: socket.socket, args, cap: cv2.VideoCapture, out: cv2.VideoWriter):
    """Ping/boot all requested cars, then own the display + log loop until ESC."""
    from src.auto_control import (draw_lanes, draw_obstacles, draw_marker_debug,
                                   _log_entries, save_log, undistort_frame, startup_calibration)

    _ac.LOG_SCENARIO = args.scenario
    _ac.LOG_POLICY = args.policy
    _ac.DEFAULT_POLICY = args.policy
    _ac.LOG_RUN_NAME = args.run_name
    _ac.LOG_REPETITION = args.repetition

    startup_calibration(args, cap)

    if args.fov is not None:
        _ac._DFOV = args.fov.strip().rstrip().rstrip("deg")
        print(f"[fov] Diagonal FOV set to {_ac._DFOV} via --fov flag")
    elif _ac._DFOV is None:
        _ac._DFOV = "unknown"
        print("[fov] No FOV known -- using 'unknown' tag. Pass --fov <deg> to set it.")

    key_bind = binds.KeyboardBinds()
    players, threads, unresponsive = {}, {}, []

    if args.no_boot:
        for cid in car_list:
            _ac.ACTIVE_CAR_IDS.add(int(cid))
        print(f"[no-boot] All requested car IDs pre-registered as active "
              f"({sorted(_ac.ACTIVE_CAR_IDS)}) -- markers not treated as obstacles")

    print("Initializing...")
    for cid in car_list:
        cid = int(cid)
        p = Player(cid, args.controller, args.mode)
        if args.no_boot:
            print("[boot] --no-boot set, skipping SSH launch")
            p.boot = lambda: None

        if p.ping() == 0 or args.no_boot:
            print(f"[Minicar {cid}] RESPONSIVE")
            players[cid] = p
            _ac.ACTIVE_CAR_IDS.add(cid)
            t = Thread(target=p.transfer_data, args=(sock, args.port), daemon=True)
            threads[cid] = t
            t.start()
        else:
            print(f"[Minicar {cid}] UNRESPONSIVE")
            unresponsive.append(cid)

    if unresponsive and not args.no_boot:
        print(f"Minicar(s) {unresponsive} unresponsive -- press R to retry or Q to quit.")
        while True:
            if keyboard.is_pressed(key_bind.stop):
                print("Quitting the connection!")
                sock.close(); sys.exit(0)
            elif keyboard.is_pressed(key_bind.retry):
                print("\nRetry connecting to unresponsive minicar(s)...")
                players_run(unresponsive, sock, args, cap, out)
                return
            time.sleep(0.1)
    else:
        print("All minicars ready!")

    Thread(target=camera_producer, args=(cap,), daemon=True).start()
    print("\nWaiting for first camera frame...")
    while get_latest_frame() is None:
        time.sleep(0.05)
    print("Camera streaming!\n")

    record = args.video_name is not None
    WINDOW = "Minicar Player View"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(f"[ctrl] Controlling cars: {args.cars} | press ESC to quit"
          + (f" | recording -> {args.video_name}.mp4" if record else ""))

    try:
        while True:
            raw = get_latest_frame()
            if raw is None:
                time.sleep(0.02)
                continue

            raw = undistort_frame(raw)
            h, w = raw.shape[:2]
            canvas = raw.copy()

            from src.auto_control import _last_lanes as _ll
            draw_lanes(canvas, _ll)
            draw_obstacles(canvas, getattr(_ac, "_obstacle_positions", []))
            draw_marker_debug(canvas)  # once per frame: identical across cars for the same physical frame

            with result_lock:
                draws_snap = dict(car_results)
                log_snap = dict(car_log_results)

            active_ids = sorted(cid for cid, dd in draws_snap.items() if dd is not None)
            for cid in active_ids:
                _draw_car_overlay(canvas, draws_snap[cid])
            for idx, cid in enumerate(active_ids):
                _draw_hud_box(canvas, draws_snap[cid], idx, h)

            entry = _ac.build_frame_log_entry(log_snap, _ll, _ac._cycle_counter)
            if entry is not None:
                _log_entries.append(entry)

            fps_text = f"{len(args.cars)} car(s) | k={_ac._cycle_counter}"
            tw = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            cv2.putText(canvas, fps_text, (w - tw[0] - 8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

            if record:
                out.write(canvas)

            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) & 0xFF == key_bind.exit_ASCII:  # ESC
                print("[player] ESC -- shutting down.")
                break

            time.sleep(0.02)  # ~50 Hz
    finally:
        for cid in list(players.keys()):
            pkt = bytearray(struct.pack(PKT_FMT, 0.0, 0.0, 0.0, True))
            try:
                sock.sendto(pkt, (players[cid].ip, args.port))
            except OSError:
                pass
        if record:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
        save_log()
        print("[player] Shutdown complete.")

# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Player-mode minicar launcher (multi-car)")
    parser.add_argument("-n", "--cars", nargs="+", type=int, default=None,
                         help="Car IDs to control, e.g. -n 1 2 3")
    parser.add_argument("--no-boot", action="store_true",
                         help="Skip the SSH boot step (useful when players are already running)")
    parser.add_argument("--port", type=int, default=6789, help="TCP port on each minicar (default 6789)")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calib-file", type=str, default=None, metavar="PATH")
    parser.add_argument("--board", type=str, default="chess", choices=["chess", "charuco"])
    parser.add_argument("--chess-size", type=int, nargs=2, default=[8, 6], metavar=("COLS", "ROWS"))
    parser.add_argument("--square-mm", type=float, default=38.0)
    parser.add_argument("--marker-mm", type=float, default=18.0)
    parser.add_argument("--scenario", type=str, default="S1", help="Experiment scenario label (default S1)")
    parser.add_argument("--policy", type=str, default="cooperative", help="Driving policy label (default cooperative)")
    parser.add_argument("--fov", type=str, default=None, metavar="DEG",
                         help="Diagonal FOV of the overhead camera in degrees, e.g. --fov 90. "
                              "Sets the dfov tag in log filenames/meta; overrides any FOV inferred from --calib-file.")
    parser.add_argument("--run-name", type=str, default=None, metavar="NAME", dest="run_name",
                         help="Optional sub-folder under .exp/{scenario}/. Omit to log directly there.")
    parser.add_argument("--repetition", type=int, default=1, metavar="N",
                         help="Repetition index used in the log filename, e.g. --repetition 3")
    parser.add_argument("--video-name", type=str, default=None, metavar="NAME", dest="video_name",
                         help="Base name for the recorded video (without .mp4). Default: no recording.")
    parser.add_argument("-c", "--controller", type=str, default="keyboard", help="Controller type: keyboard|joystick")
    parser.add_argument("-m", "--mode", type=int, default=2, help="Operation mode: 0=manual 1=semi-autonomous 2=autonomous")
    args = parser.parse_args()

    if not args.cars:
        print("No minicar selected. Use -n <id>")
        sys.exit(0)

    cap, out = _ac.init_camera(args.cam, video_name=args.video_name)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        players_run(args.cars, sock, args, cap, out)
    finally:
        _ac.ACTIVE_CAR_IDS.clear()
        sock.close()

if __name__ == "__main__":
    main()
