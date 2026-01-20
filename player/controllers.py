"""class Autonomous(object):
This file provides the keyboard and Joystick control for the player.
Along with the semi and autonomous control of the car.
"""

import time
import keyboard
import pygame
import sys

from auto_control_diag import run_a
from auto_control_follow import run_f
from player import binds

def safe_write(file, msg):
    """Displays on console and writes to a .txt file safely"""
    if file and not file.closed:
        print(msg)
        file.write(msg)
        file.flush()


class Keyboard(object):
    """Defines the Keyboard object to transfer keyboard inputs into player instructions"""

    def __init__(self, mode, number, capture, ideal_speed, max_angle):
        # Import variables
        self.ideal_speed = ideal_speed
        self.max_angle = max_angle
        self.car_number = number
        self.capture = capture

        # Define variables
        self.speed = 0.
        self.angle = 0.
        self.frame = None
        self.brightness = 0.
        self.turned_on = False

        self.control_type = mode
        self.clean = False
        self.running = True

        self.binds = binds.KeyboardBinds()
        self.file = open('./player/console_prints/keyboard-console-log.txt', 'w')

        safe_write(self.file, "Turning on the control functionality...\n")

        if self.control_type == 0:
            safe_write(self.file,"Launching in mode {} - Manual Control...\r\n".format(self.control_type))
        elif self.control_type == 1:
            safe_write(self.file,"Launching in mode {} - Semi-Autonomous Control...\r\n".format(self.control_type))
        elif self.control_type == 2:
            safe_write(self.file,"Launching in mode {} - Autonomous Control...\r\n".format(self.control_type))

    def break_until_stop(self):
        """Gradually brakes until the car's motor stops"""
        brake_force = 0.1  # units to gradually break

        while abs(self.speed) > 0.:
            if self.speed > 0.:
                self.speed = max(0., self.speed - brake_force)
            else:
                self.speed = min(0., self.speed + brake_force)
            time.sleep(0.05)

    def motor_stop(self):
        """Full brakes and stops the car's motor instantly"""
        self.speed = 0.

    def neutral_steering(self):
        """Straight steering position of the car's servo"""
        self.angle = 0.

    def stop(self):
        """Shutdowns the motor, resets the servo's angle to stop the car"""
        if keyboard.is_pressed(self.binds.stop):
            self.motor_stop()
            self.neutral_steering()
            
            safe_write(self.file,"Car is stopping...\nNeutral steering position...\n")

    def accelerate(self, accel):
        """Increases/decreases the speed up to the max speed"""
        self.speed = min(self.ideal_speed, max(self.speed + accel, -self.ideal_speed))

    def turn(self, turn):
        """Increases/decreases the turning angle up to the max angle"""
        self.angle = min(self.max_angle, max(self.angle + turn, -self.max_angle))

    def listen(self):
        """Interprets bound inputs"""

        forward_pressed = keyboard.is_pressed(self.binds.forwards)
        backward_pressed = keyboard.is_pressed(self.binds.backwards)

        # Quit
        if keyboard.is_pressed(self.binds.exit):
            safe_write(self.file,"\nTurning off the control functionality...")
            self.file.close()
            print("Keyboard/print logs file generated from the console's outputs...")
            
            self.running = False
            self.clean = True

            return

        # Set control type
        if keyboard.is_pressed(self.binds.manual) and self.control_type != 0:
            self.stop()
            self.control_type = 0
            time.sleep(0.1)

            safe_write(self.file,'Switching to Manual Control...\n')
        elif keyboard.is_pressed(self.binds.semiautonomous) and self.control_type != 1:
            self.stop()
            self.control_type = 1
            time.sleep(0.1)

            safe_write(self.file,'Switching to Semi-Autonomous Control...\n')
        elif keyboard.is_pressed(self.binds.autonomous) and self.control_type != 2:
            self.stop()
            self.control_type = 2
            time.sleep(0.1)
            
            safe_write(self.file,'Switching to Autonomous Control...\n')

        # Manual control
        if self.control_type == 0:
            moved = False
            turned = False

            if forward_pressed and backward_pressed:
                self.motor_stop()
                safe_write(self.file,"W and S - breaking with {}!\n".format(self.speed))

                moved = True
            elif forward_pressed:
                self.accelerate(self.ideal_speed)
                safe_write(self.file,"W - acceleration with {}!\n".format(self.speed))

                moved = True
            elif backward_pressed:
                self.accelerate(-self.ideal_speed)
                safe_write(self.file,"S - deceleration with {}!\n".format(self.speed))
                
                moved = True

            if keyboard.is_pressed(self.binds.turn_left):
                self.turn(-self.max_angle)
                safe_write(self.file,"A - turn left with {}!\n".format(self.angle))

                turned = True
            elif keyboard.is_pressed(self.binds.turn_right):
                self.turn(self.max_angle)
                safe_write(self.file,"D - turn right with {}!\n".format(self.angle))

                turned = True
            
            if not moved and not turned:
                self.break_until_stop()
                self.neutral_steering()
            elif not moved and turned:
                self.break_until_stop()

        # Semi-autonomous control
        elif self.control_type == 1:
            moved = False

            if forward_pressed and backward_pressed:
                self.motor_stop()
                safe_write(self.file,"W and S - breaking with {}!\n".format(self.speed))

                moved = True
            elif forward_pressed:
                self.accelerate(self.ideal_speed)
                safe_write(self.file,"W - acceleration with {}!\n".format(self.speed))

                moved = True
            elif backward_pressed:
                self.accelerate(-self.ideal_speed)
                safe_write(self.file,"S - deceleration with {}!\n".format(self.speed))
                
                moved = True

            if not moved:
                self.break_until_stop()      

            self.angle, _, self.frame = run_a(self.capture, self.car_number)
            # self.angle, _, self.frame = run_f(self.capture, self.car_number)
            safe_write(self.file, "Autonomous turning with {}!\n".format(self.angle))

        # Autonomous control
        elif self.control_type == 2:
            # self.angle, self.speed, self.frame = run_a(self.capture, self.car_number)
            self.angle, self.speed, self.frame = run_f(self.capture, self.car_number)
            safe_write(self.file, "Autonomous speed of {} and turning with {}!\n".format(self.speed, self.angle))

        # LEDs brightness control
        if keyboard.is_pressed(self.binds.lights_off):
            self.turned_on = False
            self.brightness = 0.
            
            safe_write(self.file,"All lights full off!\n")
        elif keyboard.is_pressed(self.binds.lights_on):
            self.turned_on = True
            self.brightness = 1.
            
            safe_write(self.file,"All lights full on!\n")

        if self.turned_on:
            if keyboard.is_pressed(self.binds.decrease_brightness) and not keyboard.is_pressed(self.binds.lights_on):
                self.brightness = round(max(0., self.brightness - 0.05), 2)
                
                if self.brightness > 0.:
                    safe_write(self.file,"Decrease brightness - LEDs value: {}!\n".format(self.brightness))
            elif keyboard.is_pressed(self.binds.increase_brightness) and not keyboard.is_pressed(self.binds.lights_on):
                self.brightness = round(min(1., self.brightness + 0.05), 2)
                
                if self.brightness < 1.:
                    safe_write(self.file,"Increase brightness - LEDs value: {}!\n".format(self.brightness))


class Joystick(object):
    """Defines the Joystick object to transfer joystick inputs into player instructions"""

    def __init__(self, number, capture, ideal_speed, max_angle):
        # Import variables
        self.ideal_speed = ideal_speed
        self.max_angle = max_angle
        self.car_number = number
        self.capture = capture

        # Define variables
        self.speed = 0.
        self.angle = 90.
        self.brightness = 0.

        self.control_type = 0
        self.clean = False
        self.running = True
        self.joystick = None
        
        self.binds = binds.JoystickBinds()
        self.file = open('./player/console_prints/joystick-console-log.txt', 'w')

        # Initialise joystick
        pygame.display.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()

        if joystick_count == 0:
            print('No joystick connected!')
            sys.exit()

        elif joystick_count == 1:
            self.joystick = pygame.joystick.Joystick(0)

        elif joystick_count > 1:
            safe_write(self.file,'Number of joysticks: {}'.format(joystick_count))

            # Print options
            for i in range(joystick_count):
                joystick = pygame.joystick.Joystick(i)
                safe_write(self.file,'ID: {}   |   Name: {}\n'.format(i, joystick.get_name()))

            # Select options
            if joystick_count > 1:
                while self.joystick is None:
                    for i in range(joystick_count):
                        if keyboard.is_pressed(str(i)):
                            self.joystick = pygame.joystick.Joystick(i)

        self.joystick.init()
        if self.joystick.get_init():
            """
            safe_write(self.file,'Name: {}\n'.format(self.joystick.get_name()))
            safe_write(self.file,'Axes: {}\n'.format(self.joystick.get_numaxes()))
            safe_write(self.file,'Trackballs: {}\n'.format(self.joystick.get_numballs()))
            safe_write(self.file,'Buttons: {}\n'.format(self.joystick.get_numbuttons()))
            safe_write(self.file,'Hats: {}\n'.format(self.joystick.get_numhats()))
            """
            pass
        else:
            safe_write(self.file,'Error initialising joystick!')

    def __del__(self):
        try:
            self.joystick.quit()
        except (AttributeError, pygame.error):
            pass

    def check(self, bind):
        """Read data from bound location"""
        input_type = bind[0]
        input_location = bind[1]

        if input_type == 'axis':
            return round(self.joystick.get_axis(input_location), 2)
        elif input_type == 'button':
            return self.joystick.get_button(input_location)
        elif input_type == 'hat':
            return self.joystick.get_hat(input_location[0])[input_location[1]]

    def accelerate(self, accel):
        """Increases/decreases the speed up to the max speed"""
        self.speed = min(self.ideal_speed, max(self.speed + accel, -self.ideal_speed))

    def turn(self, turn):
        """Increases/decreases the turning angle up to the max angle"""
        self.angle = min(self.max_angle, max(self.angle + turn, -self.max_angle))

    def motor_stop(self):
        """Full brakes and stops the car's motor instantly"""
        self.speed = 0.

    def neutral_steering(self):
        """Straight steering position of the car's servo"""
        self.angle = 0.

    def stop(self):
        """Shutdowns the motor, resets the servo's angle to stop the car"""
        if keyboard.is_pressed(self.binds.stop):
            self.motor_stop()
            self.neutral_steering()
            
            safe_write(self.file,"Car is stopping...\nNeutral steering position...\n")

    def listen(self):
        """Interprets bound inputs"""
        pygame.event.pump()

        # Quit
        if self.check(self.binds.exit) == 1:
            safe_write(self.file,"\nTurning off the control functionality...")
            self.file.close()

            self.running = False
            self.clean = True

        # Set control type
        if keyboard.is_pressed(self.binds.manual) and self.control_type != 0:
            self.stop()
            self.control_type = 0
            time.sleep(0.1)
            
            safe_write(self.file,'Switching to Manual Control...\n')
        elif keyboard.is_pressed(self.binds.semiautonomous) and self.control_type != 1:
            self.stop()
            self.control_type = 1
            time.sleep(0.1)
            
            safe_write(self.file,'Switching to Semi-Autonomous Control...\n')
        elif keyboard.is_pressed(self.binds.autonomous) and self.control_type != 2:
            self.stop()          
            self.control_type = 2
            time.sleep(0.1)
            
            safe_write(self.file,'Switching to Autonomous Control...\n')

        # Manual control
        if self.control_type == 0:
            self.speed = self.ideal_speed * -self.check(self.binds.speed)
            self.angle = self.max_angle * -self.check(self.binds.angle)
            
            safe_write(self.file,"Manual moving forward with {} and manual turn with {}!\n".format(self.speed, self.angle))

        # Semi-autonomous control
        elif self.control_type == 1:
            self.accelerate(self.ideal_speed * self.check(self.binds.accelerate) / 10.)
            safe_write(self.file,"Manual moving forward with {}!\n".format(self.speed))

            self.angle, _, self.frame = run_a(self.capture, self.car_number)
            # self.angle, _, self.frame = run_f(self.capture, self.car_number)
            safe_write(self.file, "Autonomous turning with {}!\n".format(self.angle))

        # Autonomous control
        elif self.control_type == 2:
            self.angle, self.speed, self.frame = run_a(self.capture, self.car_number)
            # self.angle, self.speed, self.frame = run_f(self.capture, self.car_number)
            safe_write(self.file, "Autonomous speed of {} and turning with {}!\n".format(self.speed, self.angle))

        # LEDs brightness control
        if self.check(self.binds.brightness) > 0.5:
            self.brightness = self.check(self.binds.brightness)
            safe_write(self.file,"Adjust brightness - LEDs value: {}!\n".format(self.brightness))
        elif self.check(self.binds.lights_on) == 1:
            self.brightness = 1.
            safe_write(self.file,"All lights full on!\n")
        elif self.check(self.binds.lights_off) == 1:
            self.brightness = 0.
            safe_write(self.file,"All lights full off!\n")
