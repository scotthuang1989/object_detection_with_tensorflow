import time
from multiprocessing import Value
import threading


class FPS(object):
    def __init__(self, thread=False):
        self.start_time = -1
        self.thread = thread
        if self.thread:
            self.frame_count = 0
            self.lock = threading.Lock()
        else:
            self.frame_count = Value('i', 0)

    def start_count(self):
        self.start_time = time.time()

    def add_frame(self):
        if self.thread:
            self.lock.acquire()
            self.frame_count = self.frame_count + 1
            self.lock.release()
        else:
            with self.frame_count.get_lock():
                self.frame_count.value += 1

    def get_fps(self):
        if self.thread:
            self.lock.acquire()
            fps = self.frame_count/(time.time()-self.start_time)
            self.lock.release()
            return fps
        else:
            return self.frame_count.value/(time.time()-self.start_time)
