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
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

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
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
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
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose, isStopWP):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
            isStopWP (Boolean): if this is a stop waypoint

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement

        x = -1
        y = -1

        if(isStopWP):
            x = pose[0]
            y = pose[1]
        else:
            x = pose.position.x
            y = pose.position.y

        closest_waypoint = -1
        closest_distance = float('inf')

        for wp_index in range(len(self.waypoints)):
            wx = self.waypoints[wp_index].pose.pose.position.x
            wy = self.waypoints[wp_index].pose.pose.position.y
            distance = math.sqrt((x - wx)**2 + (y - wy)**2)
            if distance < closest_distance:
                closest_waypoint = wp_index

        rospy.logdebug("Closest waypoint distance is {}, which is waypoint {}".format(closest_distance, closest_waypoint))

        return closest_waypoint

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

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
        
        # prevent from running function if values are not set
        index, traffic_ligth_state = -1, TrafficLight.UNKNOWN

        if not self.pose:
            rospy.logwarn('no pose has been set')
            return index, traffic_ligth_state

        if not self.waypoints:
            rospy.logwarn('no waypoints have been set')
            return index, traffic_ligth_state

        if not self.lights:
            rospy.logwarn('no lights have been set')
            return index, traffic_ligth_state

        if not self.light_classifier:
            rospy.logwarn('no classifier initialized')
            return index, traffic_ligth_state


        # get current position and yaw of the car
        current_x = self.pose.position.x
        current_y = self.pose.position.y
        current_orientation = self.pose.orientation
        current_q = (current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w)
        _, _, current_w = tf.transformations.euler_from_quaternion(current_q)

        # find closest light ahead
        light = None
        light_wp = (float('inf'), -1, None)

        for i in range(len(self.lights)):
            light = self.lights[i]
            light_x = light.pose.pose.position.x
            light_y = light.pose.pose.position.y
            light_orientation = light.pose.pose.orientation
            light_q = (light_orientation.x, light_orientation.y, light_orientation.z, light_orientation.w)
            _, _, light_w = tf.transformations.euler_from_quaternion(light_q)

            # verify if light is ahead
            light_ahead = ((light_x - current_x) * math.cos(current_w) + 
                           (light_y - current_y) * math.sin(current_w)) > 0

            if not light_ahead:
                rospy.logdebug("light not ahead")
                continue
            rospy.logdebug("light ahead")

            # verify if light is facing the car
            light_facing_car = light_w * current_w > 0

            if not light_facing_car:
                rospy.logdebug("light not facing car")
            rospy.logdebug("light facing car")

            # calculate distance and store if closer than current distance
            light_distance = math.sqrt((current_x - light_x)**2 + (current_y - light_y)**2)
            rospy.logdebug("Store light {} with distance {} and position {}, {}".format(i, light_distance, light_x, light_y))

            if light_distance < light_wp[0]:
                light_wp = light_distance, self.get_closest_waypoint(light.pose.pose, False), light


        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        rospy.logdebug("Stop line positions ({})".format(stop_line_positions))
        
        stop_wp = -1
        stop_distance = float('inf')
        for stop_line in stop_line_positions:
            stop_x = stop_line[0]
            stop_y = stop_line[1]
            stop_d = math.sqrt((current_x - stop_x)**2 + (current_y - stop_y)**2)
            if stop_d <  stop_distance:
                stop_wp = self.get_closest_waypoint(stop_line, True)

        # Don't classify lights that are far away
        rospy.logdebug("Closest light is {} far and is {} waypoint".format(light_wp[0], light_wp[1]))
        rospy.logdebug("Closest stop line is {} far and is {} waypoint".format(stop_distance, stop_wp))

        if light_wp[0] > 50:
            return index, traffic_ligth_state

        state = self.get_light_state(light_wp[2])

        rospy.logdebug("Light state is " + str(state))

        #self.waypoints = None
        return stop_wp, state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
