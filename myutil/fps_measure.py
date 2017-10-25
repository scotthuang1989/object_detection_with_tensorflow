import time
from multiprocessing import Value


class FPS(object):
    def __init__(self):
        self.start_time = -1
        self.frame_count = Value('i', 0)

    def start_count(self):
        self.start_time = time.time()

    def add_frame(self):
        with self.frame_count.get_lock():
            self.frame_count.value += 1

    def get_fps(self):
        return self.frame_count.value/(time.time()-self.start_time)
