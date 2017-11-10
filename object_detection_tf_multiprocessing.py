import os
import tarfile
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue
import time
import argparse
import logging

import numpy as np
import cv2


from myutil import downloadutil, fps_measure, queue_seq
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-v', '--video', type=str, required=True,
                        help="video file for detection")
arg_parser.add_argument('-p', "--process", type=int, default=1,
                        help="# of detection process")

args = arg_parser.parse_args()

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_08'
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'


def load_graph(model_name=MODEL_NAME):
    MODEL_FILE = model_name + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = model_name + '/frozen_inference_graph.pb'
    downloadutil.maybe_download(os.getcwd(), MODEL_FILE,
                                DOWNLOAD_BASE+MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
          tar_file.extract(file, os.getcwd())
    # load graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return detection_graph

NUM_CLASSES = 90

def load_label_map(label_map_name, num_class):
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', label_map_name)
    # load label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                max_num_classes=num_class, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def detect_object(detection_graph, sess, image, category_index):
    with detection_graph.as_default():
        with sess.as_default() as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            #   image = Image.open(image_path)
              # the array based representation of the image will be used later in order to prepare the
              # result image with boxes and labels on it.
            # image_np = load_image_into_numpy_array(image)
            image_np = image
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8,
              min_score_thresh = 0.7)
            return image_np


detection_graph = load_graph(model_name=MODEL_NAME)
category_index = load_label_map(label_map_name='mscoco_label_map.pbtxt', num_class=NUM_CLASSES)

image_q = Queue(maxsize=200)
processed_q = queue_seq.Queue_Seq(maxsize=200)


#a process that put imge into image_q

def image_worker(image_q, video_file):
    print("image worker start")
    video_capture = cv2.VideoCapture(video_file)
    ret, frame = video_capture.read()
    frame_count = 0
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_q.put((frame_count, frame))
        ret, frame = video_capture.read()
        frame_count += 1
    video_capture.release()

input_process = multiprocessing.Process(target=image_worker, args=(image_q, args.video))

# a process to do the detection_graph

def object_detection_worker(image_q, processed_q, detection_graph, category_index, fps=None):
    print("detection worker start")
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(graph=detection_graph, config=config)
    while True:
        frame_count, frame = image_q.get()
        t = time.time()
        ann_image = detect_object(detection_graph, sess, frame, category_index)
        ann_image = cv2.cvtColor(ann_image, cv2.COLOR_RGB2BGR)
        if fps:
            fps.add_frame()
        processed_q.put((frame_count, ann_image))


def main():
    # configure logger
    logging.basicConfig(
        level=logging.INFO,
    )

    # setup fps counter
    fps = fps_measure.FPS()
    fps.start_count()
    detector_process = [multiprocessing.Process(target=object_detection_worker,
                        args=(image_q, processed_q, detection_graph, category_index, fps))
                        for i in range(args.process)]

    input_process.start()
    for p in detector_process:
        p.start()

    last_frame = -1
    while True:
        frame_count, ann_image = processed_q.get()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(ann_image, 'FPS:{}'.format(int(fps.get_fps())), (50, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # check frame order
        if last_frame != -1:
            if last_frame +1 != frame_count:
                cv2.putText(ann_image, "Frame order error", (100,100), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        last_frame = frame_count

        cv2.imshow('frame', ann_image)
        # print("fps is:", fps.get_fps())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    input_process.terminate()
    for p in detector_process:
        p.terminate()

    input_process.join()
    for p in detector_process:
        p.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
