import numpy as np
import cv2
import os
import sys

from kalman import KalmanFilter
from config import MAX_NUM_MISSING_PERMISSION
class Instance(object):
    def __init__(self):   # fps: frame per second
        self.num_misses = 0
        self.max_misses = MAX_NUM_MISSING_PERMISSION

        self.has_match = False

        # flags: self.delete.......

        self.kalman = KalmanFilter()
        # self.history

    def add_to_track(self, tag, bbox):
        corrected_bbox = self.kalman.correct(bbox)
        # self.history.append(corrected_bbox)

    def get_predicted_bbox(self):
        # get a prediction
        return self.kalman.get_predicted_bbx()