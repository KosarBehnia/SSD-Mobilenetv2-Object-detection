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


