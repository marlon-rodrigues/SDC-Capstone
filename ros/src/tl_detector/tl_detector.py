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
import numpy as np
import scipy.misc

STATE_COUNT_THRESHOLD = 3

dl = lambda a, b: np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.prev_distance = None
        self.prev_pos = None
        self.capture = False
        self.current_state = -1
        self.idx = 0

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

        self.next_light_pos = None
        self.prev_min_dist = None


        rospy.logwarn( "config: => %s" % self.config)


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

        self.nearest_traffic_light()

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def nearest_traffic_light(self):

        eucdist = lambda x1,y1, x2, y2: np.sqrt( (x1 - x2)**2 + (y1-y2)**2)

        min_dist = np.inf
        min_loc = None
        pose = self.pose.pose.position

        for x,y in self.config["stop_line_positions"]:
            dist = eucdist( x, y, pose.x, pose.y)
            if dist<min_dist:
                min_dist = dist
                min_loc = (x,y)

        if (self.next_light_pos is None) and (self.prev_min_dist is None):
            self.next_light_pos = min_loc
            self.prev_min_dist = min_dist


        delta_dist = min_dist - self.prev_min_dist
        if delta_dist < 0:
            self.next_light_pos  = min_loc
        else:
            self.next_light_pos = None


        rospy.logwarn( "delta = %s, next_light = %s" % ( delta_dist, self.next_light_pos) )

        self.prev_min_dist  = min_dist

    def traffic_cb(self, msg):
        self.lights = msg.lights
        self.capture = False
        return


        #roslaunch launch/styx.launch
        states = [item.state for item in msg.lights]
        positions = [item.pose.pose.position for item in msg.lights]

        car_pos = self.pose.pose.position

        distances = [ dl(p, car_pos) for p in positions ]
        nearest  = np.argmin( distances )

        distance = distances[nearest]

        if not self.prev_distance:
            self.prev_distance = distance


        d_diff =  distance  - self.prev_distance
        self.prev_distance = distance

        if (d_diff < 0) and (distance < 300):
            self.capture = True
            self.current_state = states[nearest]
            rospy.logwarn( "capturing for state %s " % states[nearest])
        elif (d_diff>0) and (distance > 100):
            rospy.logwarn( "capturing 'unkown'" )
            self.current_state = 4 #states[nearest]
            self.capture = True
        else:
            self.capture = False



        #cur_pos = "%s %s %s" % (car_pos.x, car_pos.y, car_pos.z)
        #rospy.logwarn( "pose: %s " % cur_pos )
        #rospy.logwarn( "nearest position :%s, distance %d " % ( nearest, distances[nearest] ))
        #rospy.logwarn( "traffic cb: %s " % str(states[nearest]) )


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


        if False: #self.has_image and self.capture:
            img = np.array(self.camera_image)

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            im_path = "/home/student/catkin_ws/src/SDC-Capstone/ros/img2/train_%s_%s.jpg" % (self.idx, self.current_state)
            cv2.imwrite(im_path, cv_image)
            self.idx+=1



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

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return 0
        if not self.waypoints:
            #rospy.logwarn("no waypoints")
            return 0

        pose = self.pose.pose.position

        distances = []
        for wp in self.waypoints.waypoints:
            wpos = wp.pose.pose.position
            dist = dl( pose, wpos )
            distances.append( dist )

        min_dist = np.argmin(distances)
        rospy.logwarn( "min dist idx is %s" % min_dist )

        return min_dist

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
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)
        #rospy.logwarn("Checking model")
        #light=True
        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
