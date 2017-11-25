**Warning**

Here is the version of OS, main libs, other configuration is **not** tested and there is no plan for windows platform.
* ubuntu 16.04
* opencv 3.3.1
* tensorflow 1.4

## Target

* detect object in a video or from camera.

## Current Status

on my nvidia 1060 (6GB RAM ), the speed is ~20 frame per second, but gpu utilization
is below 30%, I will use multiprocessing to speed up.

Image0:
![image1](./images/bigbang3_fps.png)

Image1:
![image1](./images/bigbang1.png)

Image2:
![image1](./images/bigbang2.png)



## Project Structure

### From google tensorflow/models
object label and detecion api
#### data/
label data for evaluation.

#### object_detection/
object_detection API implemented by google
* commit: commit d710a97330152b9767f40f2554cda147d580cc5c
* Date:   Wed Nov 15 10:01:54 2017 -0800


### My module

#### myutil
* downloadutil.py: helper function for downloading model
* fps_measure.py: utility to measure fps
* queue_seq.py: a helper class to make sure the frame after processed by multi-process is in order.

#### object_detection_tf_multiprocessing.py
main module for detecting object


## Usage

run following command
```
python object_detection_tf_multiprocessing.py -v /home/scott/Videos/S11E03.mp4 -p 2
```

* you need replace the video file with your choice
* -p determine how many image detection process you want to run. For now, each detection process will comsune 1.3 GB
system RAM, ~1G Video RAM, 30% GPU(GT1060), you need decide this according to your hardware.
