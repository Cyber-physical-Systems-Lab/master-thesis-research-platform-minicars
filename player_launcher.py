"""
This file is the player control.
It enables the player to control one minicar in one of three modes using a controller.

The <username> and <password> on line 44 must be replaced by the username and password of the Raspberry Pi before
running this code.
"""

import argparse
import socket
import struct
import time
import os
import sys
import keyboard
import cv2

from threading import Thread, Lock

from player import controllers, binds

# Shared camera state
frame_lock   = Lock()
latest_frame = None          # raw camera frame (producer writes, consumers read)
result_lock  = Lock()
car_results  = {}            # {car_id: annotated_frame}  (consumers write, display reads)


# Producer: one camera thread
def camera_producer(capture):
    """Continuously reads frames from the camera and stores the latest one."""
    global latest_frame
    while True:
        ret, frame = capture.read()
        if not ret or frame is None:
            continue
        with frame_lock:
            latest_frame = frame.copy()

# Consumer helper: get the latest raw frame
def get_latest_frame():
    with frame_lock:
        return latest_frame.copy() if latest_frame is not None else None
    
def init_camera(cam_index=0):
    if os.name == "nt":
        cap = cv2.VideoCapture(cam_index)
    elif os.name == "posix":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        sys.exit("Camera not found!\n")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("Camera detected successfully!")
    return cap


class Player(object):
    """ 
        Defines the player object to be controlled externally by keyboard/controller to run the car in
        manual, semi-autonomous or autonomous mode
    """

    def __init__(self, number, controller, mode, ideal_speed=0.55, max_angle=0.5):
        # Import variables
        self.car_number = number

        # Calculate IP and slicing car_number from IP to copy the log file
        if self.car_number in range(0,10):
            self.ip = f"192.168.0.20{self.car_number}"
            self.copy_file_nb = self.ip[-1:]
        else:
            self.ip = f"192.168.0.2{self.car_number}"
            self.copy_file_nb = self.ip[-2:]
        self.username = 'cpslab1'
        self.password = 'cpslab1'

        # Assign controller
        if controller == 'keyboard':
            print("Keyboard connected!")
            self.controller = controllers.Keyboard(mode, self.car_number, ideal_speed, max_angle)
        elif controller == 'joystick':
            print("Joystick connected!")
            self.controller = controllers.Joystick(mode, self.car_number, ideal_speed, max_angle)
        else:
            raise ValueError('Invalid controller!')

        # Define variables
        self.remote_port = 6789

    def boot(self):
        """Launches car.py in the Pi board"""
        os.system('putty -ssh {}@{} -pw {} -m "./player/launch.txt"'.format(self.username, self.ip, self.password))

    def copy_file(self):
        os.system('pscp -pw {} {}@{}:car-{}-log.txt ./firmware/console_prints'.format(self.password, self.username, self.ip, self.copy_file_nb))

    def ping(self):
        """Tests for response"""
        return os.system('ping -n 1 -w 200 {} | find "Reply"'.format(self.ip))

    def transfer_data(self):
        global car_results
        """Initiates the Pi and transfers incoming data to the Pi board at 100Hz"""
        th = Thread(target=self.boot)
        th.start()
        print('Sending...\n')

        while self.controller.running:
            # Give the controller the latest shared frame instead of reading itself
            # Listening for controller commands
            shared_frame = get_latest_frame()
            self.controller.listen(captured_frame=shared_frame)      # pass frame explicitly

            # Store annotated result for this car (for the compositor)
            if self.controller.frame is not None:
                with result_lock:
                    car_results[self.car_number] = self.controller.frame.copy()

            # Send data at 100Hz
            buffer = bytearray(
                struct.pack('fff?', self.controller.speed, self.controller.angle, self.controller.brightness,
                            self.controller.clean))
            s.sendto(buffer, (self.ip, self.remote_port))
            time.sleep(1. / 100.)
        
        time.sleep(0.85)
        print(f"Copying log file from the minicar(s)...")
        self.copy_file()

        s.close()
        sys.exit(0)


def players_run(car_list):
    """Start and run the listed cars"""
    unresponsive = []
    capture = init_camera(1)
    key_bind = binds.KeyboardBinds()

    # Start the single camera producer thread
    cam_thread = Thread(target=camera_producer, args=(capture,), daemon=True)
    cam_thread.start()

    # Wait until at least one frame is available before proceeding
    print('Waiting for camera connection...')
    while get_latest_frame() is None:
        time.sleep(0.05)
    print('Camera ready!\n')

    print('Initializing...')
    for car_number in car_list:
        car_number = int(car_number)
        players[car_number] = Player(car_number, args.controller, args.mode)

        response = players[car_number].ping()
        if response == 0:
            print('Minicar {} responsive and ready to drive!\n'.format(car_number))

            players_thread[car_number] = Thread(target=players[car_number].transfer_data)
            players_thread[car_number].start()
        elif response == 1:
            print('Minicar {} unresponsive!'.format(car_number))
            
            del players[car_number]
            unresponsive.append(car_number)
    
    if unresponsive:
        print('Minicar(s) {} unresponsive!\nPress R to retry connecting or Q the connection!'.format([*unresponsive]))
        
        while True:
            if keyboard.is_pressed(key_bind.stop):
                print("Quitting the connection!")
                s.close()
                sys.exit(0)
            elif keyboard.is_pressed(key_bind.retry):
                print("\nRetry connecting to unresponsive minicar(s)...")
                players_run(unresponsive)
                break

        time.sleep(0.1)
    else:
        print('Initializing minicar(s) finished!')

    # Main display loop: composite all car frames into one
    while True:
        # Start from the latest raw camera frame as the base
        base = get_latest_frame()
        if base is None:
            time.sleep(0.02)
            continue

        # Overlay each car's annotated result on top of the base frame
        # (Each car thread writes its own annotated frame into car_results)
        with result_lock:
            results = dict(car_results)   # snapshot, don't hold lock during drawing

        display_frame = base.copy()
        for _, annotated in results.items():
            if annotated is not None and annotated.shape == display_frame.shape:
                # Blend annotated overlay (lane lines, circles, telemetry) onto base
                cv2.addWeighted(annotated, 0.85, display_frame, 0.15, 0, display_frame)

        cv2.imshow("Autonomous Minicars View", display_frame)
        if cv2.waitKey(1) & 0xFF == key_bind.exit_ASCII:
            sys.exit("Exiting visualisation!")
            break

        time.sleep(0.02)   # ~50Hz display refresh

    capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start the control of the minicar(s)")
    parser.add_argument('-n', '--cars', nargs='+', type=int, default=None, help='Manual cars: input the ID of each one')
    parser.add_argument('-c', '--controller', type=str, default='keyboard', help='Controller type: keyboard or joystick')
    #TODO change the default --mode to 2 ONLY after the car is fully functional in the autonomous mode (fully operational)
    parser.add_argument('-m', '--mode', type=int, default='0', help='Operation mode: 0 - manual, 1 - semi-autonomous, 2 - autonomous')
    args = parser.parse_args()

    if args.cars is None:
        print('No minicar selected!')
        sys.exit(0)

    # Setup socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    players = {}
    players_thread = {}

    players_run(args.cars)
