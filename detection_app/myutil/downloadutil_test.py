import unittest
import os
from util import downloadutil

class DownLoadUtilTest(unittest.TestCase):
    """Testcase for download util"""
    def test_downloadutil(self):
        data_url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz"
        downloadutil.maybe_download('/tmp',
                                    "ssd_mobilenet_v1_coco_11_06_2017.tar.gz",
                                    data_url)
        self.assertTrue(os.path.exists("/tmp/ssd_mobilenet_v1_coco_11_06_2017.tar.gz"))
