## Target

* detect people in a video or from camera.
* can choose what detection method we are using.

## Project Structure

### From google
object label and detecion api
* data
* object_detection

### My module

* myutil: help function
* object_detection_tf.py: main module for detecting object


## Usage

run following command
```
python object_detection_tf.py -v /home/scott/Videos/S11E03.mp4
```
you need replace the video file with your choice

## Current Status

on my nvidia 1060 (6GB RAM ), the speed is ~25 frame per second, but gpu utilization
is below 30%, I will use multiprocessing to speed up.
