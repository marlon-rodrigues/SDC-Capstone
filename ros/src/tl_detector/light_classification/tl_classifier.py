from styx_msgs.msg import TrafficLight
import keras

import cv2
import numpy as np
import rospy

import tensorflow as tf
from keras.models import load_model



WIDTH  = 64
HEIGHT = 64


class TLClassifier(object):


    def __init__(self):
        self.classifier = load_model("/home/student/catkin_ws/src/SDC-Capstone/ros/model/model_1.hd5")
        self.classifier._make_predict_function()
        self.graph = tf.get_default_graph()

    def get_classification(self, image):

        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #rospy.logwarn( "get get_classification")
        #return TrafficLight.UNKNOWN
        new_img = cv2.resize( image , (WIDTH, HEIGHT) ).reshape( 1, WIDTH, HEIGHT, 3)
        retval = TrafficLight.UNKNOWN
        with self.graph.as_default():
            ret = self.classifier.predict(new_img)
            tl = np.argmax(ret)
            rospy.logwarn( "Image classified as {}".format(tl))
            if tl==0:
                retval= TrafficLight.RED
            elif tl==1:
                retval = TrafficLight.YELLOW
            elif tl==2:
                retval = TrafficLight.GREEN
            else:
                retval = TrafficLight.UNKNOWN


        return retval
