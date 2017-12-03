#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Pose
from styx_msgs.msg import Lane, Waypoint

import tf
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
ONE_MPH = 0.44704 # meter per seconds


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        
        # Current pose of the vehicle
        self.pose = None

        # Part of the complete waypoints retrieved from /waypoints
        self.waypoints = None

        # Velocity
        self.current_velocity = None

        # Base waypoints ahead
        self.waypoints_ahead = []

        # First waypoint ahead
        self.next_waypoint_id = None

        # Cycle time for the waypoint updater
        self.cycleTimeUpdater = 0.1

        # Maximum allowed velocity
        self.max_velocity = 20.0 * ONE_MPH # the slower the better

        # Minimum distance to the waypoint ahead
        self.min_distance_ahead = self.max_velocity * self.cycleTimeUpdater

        rospy.Timer(rospy.Duration(self.cycleTimeUpdater), self.process_waypoints)

        rospy.spin()

    def process_waypoints(self, event):
        if not self.pose:
            rospy.logwarn('no pose has been set')
            return None

        #if not self.current_velocity:
        #    rospy.logwarn('no velocity has been set')
        #    return None

        if not self.waypoints:
            rospy.logwarn('no waypoints has been set')
            return None

        # get nearest waypoint
        begin = self.get_nearest_waypoint()
        end = begin + LOOKAHEAD_WPS
        end = min(end, len(self.waypoints))
        rospy.logdebug("begin {}, end {}, waypoints len {}".format(begin, end, len(self.waypoints)))

        waypointsArr = []
        epsilon = 1.0
        velocity = 0
        previous_target_velocity = 0

        for i in range(begin, end):
            target_velocity = self.max_velocity

            # accelerate smoothly
            start_waypoint_velocity = self.get_waypoint_velocity(self.waypoints[begin])
            previous_target_velocity = start_waypoint_velocity if i == begin else previous_target_velocity
            curr_waypoint_velocity = self.get_waypoint_velocity(self.waypoints[i])

            rospy.logdebug("{} = start v {}, current v {}, prev v {}, target v {}".format(i, start_waypoint_velocity, curr_waypoint_velocity, previous_target_velocity, target_velocity ))

            if start_waypoint_velocity == 0 and curr_waypoint_velocity == 0 and previous_target_velocity == 0:
                target_velocity = (0.25 * target_velocity + 0.75 * previous_target_velocity)
            elif previous_target_velocity < target_velocity:
                target_velocity = (0.1 * target_velocity + 0.9 * previous_target_velocity)

            # make sure target_velocity doesn't suprass max allowed velocity
            if target_velocity > 4.0:
                target_velocity = self.max_velocity

            # make sure target velocity is within bounds 
            target_velocity = min(max(0, target_velocity), self.max_velocity)

            # store previous target velocity
            previous_target_velocity = target_velocity

            self.set_waypoint_velocity(self.waypoints, i, target_velocity)
            waypointsArr.append(self.waypoints[i])

        lane = Lane()
        lane.waypoints = waypointsArr
        self.final_waypoints_pub.publish(lane)

        #vd = self.get_waypoint_velocity(self.waypoints[begin])
            
    def get_nearest_waypoint(self):
        # get the pose of the vehicle
        curr_x = self.pose.position.x
        curr_y = self.pose.position.y
        curr_orientation = self.pose.orientation
        curr_quaternion = (curr_orientation.x, curr_orientation.y, curr_orientation.z, curr_orientation.w)
        _, _, curr_t = tf.transformations.euler_from_quaternion(curr_quaternion)

        nearest_waypoint = (float('inf'), -1)
        for waypoint_idx in range(len(self.waypoints)):
            # compute the euclidian distance of the waypoint from the current pose of the vehicle
            waypoint_x = self.waypoints[waypoint_idx].pose.pose.position.x
            waypoint_y = self.waypoints[waypoint_idx].pose.pose.position.y
            waypoint_ahead = ((waypoint_x - curr_x) * math.cos(curr_t) + 
                              (waypoint_y - curr_y) * math.sin(curr_t)) > self.min_distance_ahead

            if not waypoint_ahead:
                continue

            distance = math.sqrt((curr_x - waypoint_x)**2 + (curr_y - waypoint_y)**2)
            if distance < nearest_waypoint[0]:
                nearest_waypoint = (distance, waypoint_idx)

        return nearest_waypoint[1]

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg.pose

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
