#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import yaml
import cv2
import math
from datetime import datetime

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.stop_line_waypoints = []
        self.lights = []
        self.last_known_wp = 0
        self.state_count_threshold = 3

        self.light_classifier = TLClassifier()

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.last_known_wp = -1

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        # go through all traffic lights stop lines
        for stop_line in stop_line_positions:
            stop_line_pose = PoseStamped()
            stop_line_pose.pose.position.x = stop_line[0]
            stop_line_pose.pose.position.y = stop_line[1]
            stop_line_pose.pose.position.z = 0
            stop_line_pose.pose.orientation = 0

            # get nearest wp to light
            stop_line_wp = self.get_closest_waypoint(stop_line_pose)
            self.stop_line_waypoints.append(stop_line_wp)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        #self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= self.state_count_threshold:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        x = pose.pose.position.x
        y = pose.pose.position.y
        z = pose.pose.position.z
        wp_index = self.last_known_wp

        min_distance = 1e8
        closest_waypoint = -1

        while True:
            wx = self.waypoints.waypoints[wp_index].pose.pose.position.x
            wy = self.waypoints.waypoints[wp_index].pose.pose.position.y
            wz = self.waypoints.waypoints[wp_index].pose.pose.position.z
            distance = (wx - x)**2 + (wy - y)**2 + (wz - z) ** 2
            #distance = math.sqrt((wx - x)**2 + (wy - y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_waypoint = wp_index
            else:
                break

            if wp_index == (len(self.waypoints.waypoints)-1):
                wp_index = 0
            else:
                wp_index += 1

        rospy.logdebug("Closest waypoint distance is {}, which is waypoint {}".format(min_distance, closest_waypoint))

        self.last_know_wp = closest_waypoint

        return closest_waypoint


    def get_light_state(self, light_wp):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #if(not self.has_image):
        #    self.prev_light_loc = None
        #    return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        light_wp = -1
        traffic_light_state = TrafficLight.UNKNOWN

        if not self.pose:
            rospy.logwarn('no pose has been set')
            return light_wp, traffic_light_state

        if not self.waypoints:
            rospy.logwarn('no waypoints have been set')
            return light_wp, traffic_light_state

        if not self.lights:
            rospy.logwarn('no lights have been set')
            return light_wp, traffic_light_state

        if not self.light_classifier:
            rospy.logwarn('no classifier initialized')
            return light_wp, traffic_light_state


        vehicle_wp = self.get_closest_waypoint(self.pose)
        closest_distance = 1e8

        # Go through all traffic lights stop lines
        for stop_line_wp in self.stop_line_waypoints:

            if stop_line_wp < vehicle_wp:
                wp_distance = stop_line_wp + (len(self.waypoints.waypoints) - vehicle_wp)
            else:
                wp_distance = stop_line_wp - vehicle_wp

            # If wp index distance is less than current closest, set as closest
            if wp_distance < closest_distance:
                # Update closest waypoint distance
                closest_distance = wp_distance

                # Set to closest index
                light_wp = stop_line_wp

        if light_wp != -1:
            if closest_distance < 300:
                traffic_light_state = self.get_light_state(light_wp)

            rospy.logdebug("Traffic Light Ahead: waypoint={} state={} distance={}".format(light_wp, traffic_light_state, closest_distance))

        return light_wp, traffic_light_state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
