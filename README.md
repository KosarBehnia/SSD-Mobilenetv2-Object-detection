# YOLO_Object_detection

Object Detection in Videos and pictures using TensorFlow

This is a Python script that demonstrates how to perform object detection in videos using TensorFlow. The script uses a pre-trained TensorFlow model to detect objects in each frame of a video and draws bounding boxes around the detected objects.
Installation

To run this script, you need to install the following dependencies:

    TensorFlow: Follow the instructions on the TensorFlow website ↗ to install TensorFlow on your system.
    OpenCV: Follow the instructions on the OpenCV website ↗ to install OpenCV on your system.

Sure, here's an example of a README file for a GitHub repository that presents a Python script for object detection in videos using TensorFlow:

# Object Detection in Videos using TensorFlow

This is a Python script that demonstrates how to perform object detection in videos using TensorFlow. The script uses a pre-trained TensorFlow model to detect objects in each frame of a video and draws bounding boxes around the detected objects.

## Installation

To run this script, you need to install the following dependencies:

- TensorFlow: Follow the instructions on the [TensorFlow website ↗](https://www.tensorflow.org/install) to install TensorFlow on your system.
- OpenCV: Follow the instructions on the [OpenCV website ↗](https://opencv.org/) to install OpenCV on your system.

## Usage

To use this script, follow these steps:

1. Clone this repository to your local machine.
2. Open a terminal and navigate to the directory containing the `object_detection_video.py` file.
3. Run the following command to detect objects in a video file:

 
   ```

   Replace `<PATH_TO_CKPT>` with the path of the frozen model
    Replace `<PATH_TO_LABELS> with the label maps of the trained dataset 
 
   ````
   python object_detection_video.py --input <path_to_input_video> --output <path_to_output_video> --threshold 0.5
   ````

   The default detection threshold is 0.5.
   
   Note: This script has been tested with MP4 video files on Windows and Linux systems. Other video formats may not be compatible.
   Coming Soon

    GPU version: We are working on a version of this script that uses the GPU to speed up object detection. This version will require a CUDA-enabled GPU and the installation of additional dependencies. Stay tuned for updates!
    Multithreaded version: We are also working on a multithreaded version of this script that separates the video reading process from the object detection process using multiple threads. This version will be able to read frames from the video file at a faster rate than the object detection process, which can help to minimize frame missing. Stay tuned for updates!
    URL-based version: We are also working on a version of this script that can detect objects in videos that are hosted online. This version will be able to accept a URL as input and will download the video file from the internet. Stay tuned for updates!


## Acknowledgements

This script is based on the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and uses a pre-trained [SSD MobileNet V2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md#mobile-models) model for object detection. The `label_map.pbtxt` file used in this script is based on the [MS COCO dataset](http://cocodataset.org/).


