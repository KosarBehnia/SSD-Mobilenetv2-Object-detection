## Object Detection using TensorFlow and YOLO

This repository contains pre-trained object detection models based on TensorFlow and YOLO, which can be used to detect objects in both images and videos.
TensorFlow Object Detection

The TensorFlow object detection model used in this repository is based on the pre-trained SSD MobileNet V2 model, and the label_map.pbtxt file used in the model is based on the MS COCO dataset. The script uses OpenCV to read video files and performs object detection on each frame of the video, drawing bounding boxes around the detected objects.

To use this script, you need to install TensorFlow and OpenCV on your system. The script has been tested with MP4 video files on Windows and Linux systems, but other video formats may not be compatible.

There is also a GPU version of the script available in the object_detection_video_gpu.py file, which uses the GPU to speed up object detection. This version requires a CUDA-enabled GPU and additional dependencies.

Performance comparison shows that the GPU version processes video at an average of 25 frames per second, while the CPU version processes video at an average of 6 frames per second. The exact performance may vary depending on your system configuration and the size of the input video file.

We compared the performance of the CPU and GPU versions of the script on a sample video file. The CPU version processed the video at an average of 6 frames per second, while the GPU version processed the video at 25 frames per second. Just so you know, the exact performance may vary depending on your system configuration and the size of the input video file.

The script is planned to have a multithreaded version that separates the video reading process from the object detection process using multiple threads to minimize frame missing. Another version of the script will accept a URL as input and will download the video file from the internet.
YOLO Object Detection

The YOLO object detection model used in this repository has been trained on the VisDrone dataset and can detect several classes, including 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', and 'motor'. The repository provides a Python script named yolo_detection.py, which demonstrates how to use the pre-trained model to perform object detection on images.

The yolo_detection.py script reads an image file and uses the pre-trained YOLO model to detect objects in the image. The detected objects are then drawn as bounding boxes on the image, and the image is displayed. The script uses the OpenCV library to load and display the image and to draw the bounding boxes.

When outliers are ignored in the training dataset, and the input images are adjusted to have a brightness of 0.89 and contrast of 0.79, the model achieves high training and validation metrics.

In addition to the pre-trained model, this repository also includes an implementation of the YOLO algorithm from scratch. The yolo.py file contains the implementation of the YOLO algorithm and demonstrates how to train the model on a custom dataset. This implementation uses PyTorch and has been tested on the COCO dataset.

Note that training a YOLO model from scratch can be a computationally intensive task and may require a powerful GPU.

The following diagrams illustrate the output of the YOLO algorithm on various images:

## Coming Soon

We are working on a multithreaded version of the TensorFlow script that separates the video reading process from the object detection process using multiple threads. This version will be able to read frames from the video file at a faster rate than the object detection process, which can help to minimize frame missing.

We are also working on a version of the TensorFlow script that can detect objects in videos that are hosted online. This version will be able to accept a URL as input and will download the video file from the internet.
## Acknowledgements

The TensorFlow object detection model used in this repository is based on the TensorFlow Object Detection API. The YOLO object detection model used in this repository is based on the VisDrone dataset. We want to acknowledge the creators of these datasets for their contributions to object detection.

We would also like to acknowledge the open-source community for their contributions to the development of TensorFlow and YOLO, as well as the many other libraries and tools that have made this project possible. Thank you to everyone who has helped to make this repository a valuable resource for object detection enthusiasts.

# SSD_Mobile net _Object_detection

Object Detection in Videos and pictures using TensorFlow

This is a Python script that demonstrates how to perform object detection in videos using TensorFlow. The script uses a pre-trained TensorFlow model to detect objects in each frame of a video and draws bounding boxes around the detected objects.
Installation

To run this script, you need to install the following dependencies:

    TensorFlow: Follow the instructions on the TensorFlow website ↗ to install TensorFlow on your system.
    OpenCV: Follow the instructions on the OpenCV website ↗ to install OpenCV on your system.

## Usage

To use this script, follow these steps:

1. Clone this repository to your local machine.
2. Open a terminal and navigate to the directory containing the `object_detection_video.py` file.
3. Run the following command to detect objects in a video file:

 
   ```

   Replace `<PATH_TO_CKPT>` with the path of the frozen model
    Replace `<PATH_TO_LABELS> with the label maps of the trained dataset 
 
  
   ````
   
   Note: This script has been tested with MP4 video files on Windows and Linux systems. Other video formats may not be compatible.
   
## GPU version

We have also created a GPU version of this script that uses the GPU to speed up object detection. The GPU version is available in the object_detection_video_gpu.py file. Note that this version requires a CUDA-enabled GPU and the installation of additional dependencies.

In the GPU version, we load the pre-trained model into a new Graph object and set up the GPU acceleration by creating a ConfigProto object and setting gpu_options.allow_growth to True. We then create a new Session object using the Graph object and the ConfigProto object.

We define the input and output tensors using the get_tensor_by_name() method on the Graph object. We then read frames from the input video file and perform object detection on each frame using sess.run(). The detected objects are then drawn as bounding boxes on the frame, and the frame is written to the output video file using a VideoWriter object.
## Performance Comparison
 
We compared the performance of the CPU and GPU versions of the script on a sample video file. The CPU version processed the video at an average of 6 frames per second, while the GPU version processed the video at an average of 25 frames per second. Note that the exact performance may vary depending on your system configuration and the size of the input video file.
Here is an example of the results that can be obtained by running the GPU version of the script on a suitable hardware configuration:

https://youtu.be/ynF5kOjCRIU

Here is an example of the results that can be obtained by running the CPU version of the script on a suitable hardware configuration:

https://youtu.be/Xt-0smWi3UM

It is apparent that the CPU version experiences a much greater number of missing frames.

## Coming Soon

 1- Multithreaded version: We are also working on a multithreaded version of this script that separates the video reading process from the object detection process using multiple threads. This version will be able to read frames from the video file at a faster rate than the object detection process, which can help to minimize frame missing. Stay tuned for updates!
   
 2- URL-based version: We are also working on a version of this script that can detect objects in videos that are hosted online. This version will be able to accept a URL as input and will download the video file from the internet. Stay tuned for updates!


## Acknowledgements

This script is based on the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and uses a pre-trained [SSD MobileNet V2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md#mobile-models) model for object detection. The `label_map.pbtxt` file used in this script is based on the [MS COCO dataset](http://cocodataset.org/).


