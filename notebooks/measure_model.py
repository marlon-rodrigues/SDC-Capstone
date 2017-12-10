import keras
import tensorflow as tf
from keras.models import load_model
import yaml
import glob
from time import time
import cv2

base_path = "/home/student/catkin_ws/src/SDC-Capstone/ros/src/tl_detector/"
config = yaml.load( open("/home/student/catkin_ws/src/SDC-Capstone/ros/src/tl_detector/sim_traffic_light_config.yaml"))
model_path = base_path + config["model_path"]
model_width = config["model_width"]
model_height = config["model_height"]


classifier = load_model(model_path)
classifier._make_predict_function()
graph = tf.get_default_graph()


def classify(image, classifier, model_width, model_height):
    new_img = cv2.resize( image , (model_width, model_height) ).reshape( 1, model_height, model_width, 3)
    with graph.as_default():
        ret = classifier.predict(new_img)



image_files = glob.glob( "/home/student/share/img4/*.jpg")
print( "Images: %d " % len(image_files))

for im_file in image_files:

    image = cv2.imread(im_file)

    t = time()
    classify( image, classifier, model_width, model_height)
    dt = time()  - t

    print( dt )
