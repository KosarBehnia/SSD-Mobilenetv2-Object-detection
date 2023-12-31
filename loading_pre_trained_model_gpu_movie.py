# -*- coding: utf-8 -*-
"""Loading_pre_trained_model_GPU_movie.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10TID_XCETI5i2S0Hi8CDJW4Y4YRzymxq
"""

!pip install tensorflow

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

if tf.__version__ < '1.14.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from PIL import Image, ImageDraw
import cv2

!pip install tensorflow-object-detection-api

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

!wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

!tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'  # change this path

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './mscoco_label_map.pbtxt'  # change this path

NUM_CLASSES = 90  # change this to the number of objects that you want to detect

!ls

from object_detection.utils import label_map_util
label_map_util.tf = tf.compat.v1
tf.gfile = tf.io.gfile

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

import tensorflow as tf
import cv2
import tensorflow.compat.v1 as tf
detection_threshold = .7
tf.disable_v2_behavior()
# Load the pre-trained model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    # Set up GPU acceleration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config, graph=detection_graph)

# Define the input and output tensors
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Create a VideoCapture object to read from the input video file
cap = cv2.VideoCapture('test1.mp4')

# Create a VideoWriter object to write to the output video file
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Read frames from the input video file and perform object detection on each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    frame_expanded = np.expand_dims(frame, axis=0)
    # Perform object detection
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    # Draw bounding boxes around the detected objects
    for i in range(int(num[0])):
        if scores[0,i] > detection_threshold:
            ymin, xmin, ymax, xmax = boxes[0,i]
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # Write the current frame to the output video file
    out.write(frame)

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()