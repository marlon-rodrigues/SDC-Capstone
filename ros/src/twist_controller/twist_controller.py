import math
import time
import rospy

from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, 
    			 decel_limit, accel_limit, vehicle_total_mass, brake_deadband, wheel_radius):
        # TODO: Implement
        # pass

        # Variables for Yaw, LowPass and PID controller
        min_speed = 0.1
        Kp = 1.1
        Ki = 0.010
        Kd = 0.005
        pid_cmd_range = 4
        filter_tau = 0.0
        filter_ts = 0.8

        # YawController
        # def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle) 

        # LowPass Filter
        # def __init__(self, tau, ts)
        self.low_pass_filter = LowPassFilter(filter_tau, filter_ts)

        # PID Controller for the target velocity 
        #def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM)
        self.pid_controller = PID(Kp, Ki, Kd, decel_limit, accel_limit)

        self.vehicle_total_mass = vehicle_total_mass
        self.brake_deadband = brake_deadband
        self.wheel_radius = wheel_radius

        self.t_start = time.time()
        self.dt = 0.0
        self.feed_forward_brake_control = True


    def control(self, current_velocity, linear_velocity, angular_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        # return 1., 0., 0.

        # Check if all required parameters are set
        if not (self.vehicle_total_mass and self.brake_deadband and self.wheel_radius):
        	rospy.logerror('vehicle parameters not set')

        # Apply the filter to the angular velocity
        angular_velocity = self.low_pass_filter.filt(angular_velocity)

        # Compute the steering angle
        steer = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)

        # Compute the throttle command
        throttle_cmd = 0
        velocity_error = linear_velocity - current_velocity
        if self.dt:
        	throttle_cmd = self.pid_controller.step(velocity_error, self.dt)

        self.dt = time.time() - self.t_start
        self.t_start += self.dt

        throttle = throttle_cmd
        brake = 0.0

        rospy.logdebug('throttle = %.2f, T = %.2f, B = %.2f, S = %.2f (BAND: %.2f)', throttle_cmd, throttle, brake, steer, self.brake_deadband)

        return throttle, brake, steer

    def reset(self):
    	self.pid_controller.reset()