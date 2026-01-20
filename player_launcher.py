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
from auto_control_diag import init_camera

#TODO see if here some changes are needed also
class Player(object):
    """ 
        Defines the player object to be controlled externally by keyboard/controller to run the car in
        manual, semi-autonomous or autonomous mode
    """

    def __init__(self, number, controller, mode, capture, ideal_speed=0.55, max_angle=0.5):
        # Import variables
        self.car_number = number
        self.capture = capture

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
            self.controller = controllers.Keyboard(mode, self.car_number, self.capture, ideal_speed, max_angle)
        elif controller == 'joystick':
            print("Joystick connected!")
            self.controller = controllers.Joystick(mode, self.car_number, self.capture, ideal_speed, max_angle)
        else:
            raise ValueError('Invalid controller')

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
        global latest_frame
        """Initiates the Pi and transfers incoming data to the Pi board at 100Hz"""
        th = Thread(target=self.boot)
        th.start()
        print('Listening...')

        while self.controller.running:
            # listening for controller commands
            self.controller.listen()

            if self.controller.frame is not None:
                with frame_lock:
                    latest_frame = self.controller.frame

            # Send data at 100Hz
            buffer = bytearray(
                struct.pack('fff?', self.controller.speed, self.controller.angle, self.controller.brightness,
                            self.controller.clean))
            s.sendto(buffer, (self.ip, self.remote_port))
            time.sleep(1. / 100.)
        
        time.sleep(0.85)
        print("Copying log file from the Pi board...")
        self.copy_file()

        s.close()
        sys.exit(0)


def players_run(car_list):
    """Start and run the listed cars"""
    unresponsive = []
    capture = init_camera(1)
    global latest_frame
    key_bind = binds.KeyboardBinds()

    print('Initializing...')
    for car_number in car_list:
        car_number = int(car_number)
        players[car_number] = Player(car_number, args.controller, args.mode, capture)

        response = players[car_number].ping()
        if response == 0:
            print('Player {} responsive and ready to drive!'.format(car_number))

            players_thread[car_number] = Thread(target=players[car_number].transfer_data)
            players_thread[car_number].start()
        elif response == 1:
            print('Player {} unresponsive!'.format(car_number))
            
            del players[car_number]
            unresponsive.append(car_number)
    
    if len(unresponsive) == 0:
        print('Initializing finished!\nSending...')
    else:
        print('Cars {} unresponsive!\nPress R to retry connecting or Q the connection!'.format([*unresponsive]))

        while True:
            if keyboard.is_pressed(key_bind.stop):
                print("Quitting the connection!")
                s.close()
                sys.exit(0)
                break
            elif keyboard.is_pressed(key_bind.retry):
                print("\n\nRetry connecting to unresponsive car(s)...")
                players_run(unresponsive)
                break

        time.sleep(0.1)

    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is not None:
            cv2.imshow("Autonomous Cars View", frame)
        else:
            sys.exit("No frame available for visualisation.")
            break
        
        if cv2.waitKey(1) & 0xFF == key_bind.exit_ASCII:
            sys.exit("Exiting visualisation.")
            break

        time.sleep(0.02)  # ~50Hz display refresh
    
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
        print('No player selected')
        sys.exit(0)

    # Setup socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    players = {}
    players_thread = {}
    
    frame_lock = Lock()
    latest_frame = None

    players_run(args.cars)
