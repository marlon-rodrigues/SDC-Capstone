from styx_msgs.msg import TrafficLight
import keras

import cv2
import numpy as np
import rospy

import tensorflow as tf
from keras.models import load_model
import yaml



light_label = {

    0 : "RED",
    1 : "YELLOW",
    2 : "GREEN"
}

class TLClassifier(object):


    def __init__(self):

        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        model_path = config["model_path"]
        self.model_width = config["model_width"]
        self.model_height = config["model_height"]


        rospy.logwarn( "Loading error model %s" % model_path)

        self.classifier = load_model(model_path)
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
        new_img = cv2.resize( image , (self.model_width, self.model_height) ).reshape( 1, self.model_height, self.model_width, 3)
        retval = TrafficLight.UNKNOWN
        with self.graph.as_default():
            ret = self.classifier.predict(new_img)
            tl = np.argmax(ret)

            rospy.logwarn( "Trafic Light Classified as {}".format( light_label[tl] ))
            if tl==0:
                retval= TrafficLight.RED
            elif tl==1:
                retval = TrafficLight.YELLOW
            elif tl==2:
                retval = TrafficLight.GREEN
            else:
                retval = TrafficLight.UNKNOWN

        return retval
