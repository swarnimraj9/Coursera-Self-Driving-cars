#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        self._kP                 = 0.8
        self._kI                 = 0.5
        self._kD                 = 0.0
        self._kStan              = 0.7

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def calculate_throttle(self,t,v,v_des):
        t_diff = t - self.vars.t_previous
        error = v_des - v
        p_term = self._kP * error
        i_term = self.vars.i_previous + self._kI * t_diff * error
        d_term = self._kD * error / t_diff
        self.vars.i_previous = i_term
        return p_term + i_term + d_term

    def distance(self, x, y, idx):
        d = np.sqrt((x-self._waypoints[idx][0])**2 + (y-self._waypoints[idx][1])**2)
        return d

    def closest_cal(self, x, y):
        min_idx       = 0
        min_dist      = float("inf")
        for i in range(len(self._waypoints)):
            dist = self.distance(x, y, i)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        min_idx = len(self._waypoints)-1 if (min_idx >= len(self._waypoints)-1) else min_idx
        return min_idx

    def head_calculator(self, idx):
        if idx==0:
            dx = self._waypoints[idx+1][0]-self._waypoints[idx][0]
            dy = self._waypoints[idx+1][1]-self._waypoints[idx][1]
        elif idx==len(self._waypoints)-1:
            dx = self._waypoints[idx-1][0]-self._waypoints[idx][0]
            dy = self._waypoints[idx-1][1]-self._waypoints[idx][1]
        else:
            dx = self._waypoints[idx+1][0]-self._waypoints[idx-1][0]
            dy = self._waypoints[idx+1][1]-self._waypoints[idx-1][1]
        h = np.arctan(dy/dx)
        #print(idx,",",dy,",",dx)
        return h

    def check_left(self, x, y, ind):
        x1 = self._waypoints[ind][0]
        y1 = self._waypoints[ind][1]
        x2 = self._waypoints[ind-1][0]
        y2 = self._waypoints[ind-1][1]
        k = (x-x1)*(y2-y1)-(y-y1)*(x2-x1)
        return True if k>0 else False

    def calculate_steer(self, x, y, yaw, v_f):
        self.vars.create_var('c', 0.0)
        self.vars.create_var('k', 0.0)
        self.vars.create_var('h_previous', 0.0)
        self.vars.create_var('yaw_prev', 0.0)
        closest_index = self.closest_cal(x,y)
        e = self.distance(x,y,closest_index)
        heading = self.head_calculator(closest_index)
        if self.vars.h_previous>1 and heading<=-1:
            self.vars.c += 1.0
        elif self.vars.h_previous<-1 and heading>=1:
            self.vars.c -= 1.0

        yaw = self._pi * yaw / abs(yaw) - yaw
        
        if self.vars.yaw_prev<=-2.5 and yaw>2.5:
            self.vars.k -= 1.0
        elif self.vars.yaw_prev>=2.5 and yaw<-2.5:
            self.vars.k += 1.0
        
        print(self.vars.yaw_prev, " ", yaw, " ", self.vars.k, "\n")
        psi = (heading + self.vars.c*self._pi + yaw + self.vars.k*self._2pi)
        cons = -1 if self.check_left(x, y, closest_index) else 1
        delta = psi + cons * np.arctan(self._kStan * e / v_f)
        self.vars.h_previous = heading
        self.vars.yaw_prev = yaw
        return delta

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('t_previous', 0.0)
        self.vars.create_var('i_previous', 0.0)

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.
            throttle_output = self.calculate_throttle(t,v,v_desired)
            brake_output    = 0

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change the steer output with the lateral controller. 
            steer_output    = self.calculate_steer(x,y,yaw,v)

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v  # Store forward speed to be used in next step
        self.vars.t_previous = t