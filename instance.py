import numpy as np
import cv2
import os
import sys

from kalman import KalmanFilter
from config import MAX_NUM_MISSING_PERMISSION
from util import centroid_to_bbx, bbx_to_center


class Instance(object):
    def __init__(self):  # fps: frame per second
        self.num_misses = 0
        self.max_misses = MAX_NUM_MISSING_PERMISSION

        self.has_match = False
        self.history = []

        # flags: self.delete.......

        self.kalman = KalmanFilter()

    def add_to_track(self, tag, bbox):
        center = bbx_to_center(bbox)
        corrected_bbox = self.kalman.correct(center)
        #box = centroid_to_bbx(corrected_bbox)
        self.history.append(corrected_bbox)

    def get_predicted_bbox(self):
        # get a prediction
        center = self.kalman.get_predicted_bbox()
        bbox = centroid_to_bbx(center)
        return list(map(lambda x: int(x)+1, bbox))

    def get_latest_bbx(self):
        return self.history[-1]

    def correct_track(self, detection):
        center = bbx_to_center(detection)
        self.kalman.correct(center)
