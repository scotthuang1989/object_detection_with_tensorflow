"""
Vectorization version.

---------------------------
use Vectorization to speed up.
"""

import os
import tensorflow as tf
import threading
import argparse
import logging
import time
from queue import Queue
import queue

import numpy as np
import cv2


from myutil import fps_measure
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

LABEL_FILE_NAME = "pascal_label_map.pbtxt"
NUM_CLASSES = 90
image_per_run = 8

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-v', '--video', type=str, required=True,
                        help="video file for detection")
arg_parser.add_argument('-m', '--model', type=str, default='my_exporter',
                        help='directory to find model')
arg_parser.add_argument('-s', '--save', type=bool, default=False,
                        help='directory to find model')

args = arg_parser.parse_args()


class SAFE_NUM():
  def __init__(self, num):
    self.num = num
    self.lock = threading.Lock()

  def get_num(self):
    self.lock.acquire()
    num = self.num
    self.lock.release()
    return num

  def set_num(self, num):
    self.lock.acquire()
    self.num = num
    self.lock.release()


is_quit = SAFE_NUM(0)
video_end = SAFE_NUM(0)


def load_graph(model_name=args.model):
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = model_name + '/frozen_inference_graph.pb'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_label_map(label_map_name, num_class):
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', label_map_name)
    # load label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
      label_map,
      max_num_classes=num_class, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def detect_object(detection_graph, sess, image, image_list, category_index):
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

            # build feed_dict
            feed_dict = {}
            for i in range(image_per_run):
              feed_dict.update({"image_ph%d:0" % i: image_list[i]})

            # Actual detection.
            feed_image = sess.run(image,
                                  feed_dict=feed_dict)

            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: feed_image})

            # Visualization of the results of a detection.
            for i in range(feed_image.shape[0]):
              vis_util.visualize_boxes_and_labels_on_image_array(
                feed_image[i],
                np.squeeze(boxes[i]),
                np.squeeze(classes[i]).astype(np.int32),
                np.squeeze(scores[i]),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.20)
            return feed_image


def image_worker(image_q, video_file):
    """Process that put imge into image_q."""
    logging.info("image worker start")
    video_capture = cv2.VideoCapture(video_file)
    ret, frame = video_capture.read()
    if not ret:
        logging.error("Can not read video file, please check!!!!")
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if is_quit.get_num():
          break
        image_q.put(frame)
        logging.debug("put image into queue")
        ret, frame = video_capture.read()
    video_end.set_num(1)
    video_capture.release()


def get_n_frame(f_queue, n):
  multi_images = []
  counts = n
  b_time = time.time()
  while n > 0:
    # logging.debug("get image need: %d" % (n,))
    if video_end.get_num():
      try:
        image = f_queue.get(block=False)
      except queue.Empty:
        # video file reached the end and image queue is empty, break
        break
    else:
      image = f_queue.get()
    # image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    multi_images.append(image)
    n -= 1
  logging.info("get %d frames, internal: %f" % (counts, time.time() - b_time))
  return multi_images


def object_detection_worker(image_q, processed_q, detection_graph, category_index, fps=None):
  """a process to do the detection_graph."""
  logging.info("detection worker start")
  gpu_options = tf.GPUOptions(allow_growth=True)
  config = tf.ConfigProto(gpu_options=gpu_options)
  sess = tf.Session(graph=detection_graph, config=config)

  with detection_graph.as_default():
    image_ph_list = [tf.placeholder(tf.uint8, shape=[], name="image_ph%d" % i)
                     for i in range(image_per_run)]

    frames = tf.stack(image_ph_list)

    while True:
      if is_quit.get_num() == 1:
        # before break, try to get some image, in case image worker is blocked.
        image_q.get(block=False)
        break
      images_list = get_n_frame(image_q, image_per_run)
      # print("image shape:", frame.shape)
      if len(images_list) == 0:
        break
      ann_image = detect_object(detection_graph, sess, frames, images_list, category_index)
      for img in ann_image:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if fps:
            fps.add_frame()
        processed_q.put(img)


def main():
  # configure logger
  logging.basicConfig(
      level=logging.INFO,
  )
  image_q = Queue(maxsize=200)
  processed_q = Queue(maxsize=200)
  # setup image input thread
  input_process = threading.Thread(target=image_worker, args=(image_q, args.video))
  detection_graph = load_graph(model_name=args.model)
  category_index = load_label_map(label_map_name=LABEL_FILE_NAME, num_class=NUM_CLASSES)

  # setup fps counter
  fps = fps_measure.FPS()
  fps.start_count()
  # setup object detection process
  detector_process = threading.Thread(
                      target=object_detection_worker,
                      args=(image_q, processed_q, detection_graph, category_index, fps))
  input_process.start()
  detector_process.start()

  if args.save:
    print("open video handle")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20, (512, 288))

  while True and (is_quit.get_num() == 0):
      ann_image = processed_q.get()
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(ann_image, 'FPS:{}'.format(int(fps.get_fps())), (50, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
      if args.save:
        print("write into video", ann_image.shape)
        out.write(ann_image)
      cv2.imshow('frame', ann_image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          is_quit.set_num(1)
          break

  # input_process.terminate()
  # detector_process.terminate()
  #
  # input_process.join()
  # detector_process.join()
  print("release vcideo handle")
  if args.save:
    out.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
