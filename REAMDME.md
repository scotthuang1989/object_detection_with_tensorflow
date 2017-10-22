## Target

* detect object in a video or from camera.

## Current Status

on my nvidia 1060 (6GB RAM ), the speed is ~25 frame per second, but gpu utilization
is below 30%, I will use multiprocessing to speed up.

Image1:
![image1](./images/bigbang1.png)

Image2:
![image1](./images/bigbang2.png)



## Project Structure

### From google
object label and detecion api
#### data
label data

#### object_detection
object_detection API implemented by google

### My module

#### myutil
downloadutil.py: helper function for downloading model

#### object_detection_tf.py
main module for detecting object


## Usage

run following command
```
python object_detection_tf.py -v /home/scott/Videos/S11E03.mp4
```
you need replace the video file with your choice
