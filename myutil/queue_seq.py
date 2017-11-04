import multiprocessing as mp
import logging
import threading
import queue


class Queue_Seq(object):
    """
    helper class to make sure the frame put into queue is in right order
    """
    def __init__(self, maxsize=200, thread=False):
        self.thread = thread
        if self.thread:
            self.queue = queue.Queue(maxsize)
            self.current_frame = -1
            self._event = threading.Event()
            self.lock = threading.Lock()
        else:
            self.queue = mp.Queue(maxsize)
            self.current_frame = mp.Value('i',-1)
            self._event = mp.Event()

    def put(self, element):
        """
        element is a tupel: (frame_count, frame)
        """
        logging.info("processed image: put")
        if self.thread:
            while True:
                self.lock.acquire()
                if element[0] != self.current_frame + 1:
                    logging.info(
                                "frame count:{} -> current_frame:{}".format(
                                element[0], self.current_frame))
                    self.lock.release()
                    self._event.wait()
                else:
                    self.queue.put(element)
                    self.current_frame = element[0]
                    self.lock.release()
                    self._event.set()
                    break
        else:
            while True:
                if element[0] != self.current_frame.value +1:
                    logging.info("frame count:{} -> current_frame:{}".format(element[0], self.current_frame.value))
                    self._event.wait()
                else:
                    self.queue.put(element)
                    self.current_frame.value = element[0]
                    self._event.set()
                    break

    def get(self):
        return self.queue.get()
