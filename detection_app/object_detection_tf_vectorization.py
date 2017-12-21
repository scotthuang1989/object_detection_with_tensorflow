"""
Vectorization version.

---------------------------
use Vectorization to speed up.
"""

import os
import tarfile
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue
import argparse
import logging
import time

import numpy as np
import cv2


from myutil import downloadutil, fps_measure, queue_seq
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-v', '--video', type=str, required=True,
                        help="video file for detection")

args = arg_parser.parse_args()

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_08'
MODEL_NAME = 'my_exporter'


def load_graph(model_name=MODEL_NAME):
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = model_name + '/frozen_inference_graph.pb'
    # downloadutil.maybe_download(os.getcwd(), MODEL_FILE,
    #                             DOWNLOAD_BASE+MODEL_FILE)
    # tar_file = tarfile.open(MODEL_FILE)
    # for file in tar_file.getmembers():
    #   file_name = os.path.basename(file.name)
    #   if 'frozen_inference_graph.pb' in file_name:
    #       tar_file.extract(file, os.getcwd())
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
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image})
            # Visualization of the results of a detection.
            for i in range(image.shape[0]):
              vis_util.visualize_boxes_and_labels_on_image_array(
                image[0],
                np.squeeze(boxes[0]),
                np.squeeze(classes[0]).astype(np.int32),
                np.squeeze(scores[0]),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.7)
            return image


def image_worker(image_q, video_file):
    """Process that put imge into image_q."""
    logging.info("image worker start")
    video_capture = cv2.VideoCapture(video_file)
    ret, frame = video_capture.read()
    if not ret:
        logging.error("Can not read video file, please check!!!!")
    frame_count = 0
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_q.put((frame_count, frame))
        logging.debug("put image into queue")
        ret, frame = video_capture.read()
        frame_count += 1
    video_capture.release()


def get_n_frame(f_queue, n):
  multi_images = None
  counts = n
  b_time = time.time()
  while n > 0:
    # logging.debug("get image need: %d" % (n,))
    _, image = f_queue.get()
    image = np.expand_dims(image, axis=0)
    if isinstance(multi_images, np.ndarray):
      multi_images = np.concatenate([multi_images, image])
    else:
      multi_images = image
    n -= 1
  logging.info("get %d frames, internal: %f" % (counts, time.time() - b_time))
  return multi_images


def object_detection_worker(image_q, processed_q, detection_graph, category_index, fps=None):
    """a process to do the detection_graph."""
    logging.info("detection worker start")
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(graph=detection_graph, config=config)
    while True:
        frame = get_n_frame(image_q, 4)
        # print("image shape:", frame.shape)
        ann_image = detect_object(detection_graph, sess, frame, category_index)
        for img in ann_image:
          img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
          if fps:
              fps.add_frame()
          processed_q.put((0, img))


def main():
    # configure logger
    logging.basicConfig(
        level=logging.INFO,
    )
    image_q = Queue(maxsize=200)
    processed_q = Queue(maxsize=200)
    input_process = multiprocessing.Process(target=image_worker, args=(image_q, args.video))
    detection_graph = load_graph(model_name=MODEL_NAME)
    category_index = load_label_map(label_map_name='mscoco_label_map.pbtxt', num_class=NUM_CLASSES)

    # setup fps counter
    fps = fps_measure.FPS()
    fps.start_count()
    detector_process = multiprocessing.Process(
                        target=object_detection_worker,
                        args=(image_q, processed_q, detection_graph, category_index, fps))
    input_process.start()
    detector_process.start()

    while True:
        frame_count, ann_image = processed_q.get()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(ann_image, 'FPS:{}'.format(int(fps.get_fps())), (50, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', ann_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    input_process.terminate()
    detector_process.terminate()

    input_process.join()
    detector_process.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
