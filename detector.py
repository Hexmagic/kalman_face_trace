'''
    File name         : detectors.py
    File Description  : Detect objects in video frame
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
import cv2

# set to 1 for pipeline images
debug = 0


class Detector(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #if (debug == 1):
        #    cv2.imshow('gray', gray)

        # Perform Background Subtraction
        #fgmask = self.fgbg.apply(gray)
        fgmask = gray
        #if (debug == 0):
        #    cv2.imshow('bgsub', fgmask)

        # Detect edges
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        edges = cv2.Canny(thresh, 50, 190, 3)

        #if (debug == 1):
        #    cv2.imshow('Edges', edges)

        # Retain only edges within the threshold
        

        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        #if (debug == 0):
        #    cv2.imshow('thresh', thresh)

        detections = []  # vector of object centroids in a frame
        # we only care about centroids with size of bug in this example
        # recommended to be tunned based on expected object size for
        # improved performance
        blob_radius_thresh = 8
        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # Calculate and draw circle
                x, y, w, h = cv2.boundingRect(cnt)
                if (w >= blob_radius_thresh):
                    w=h=16                    
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),
                    #              2)
                    detections.append({'bbox': [x, y, x+w, y+h]})
            except ZeroDivisionError:
                pass

        # show contours of tracking objects
        #cv2.imshow('Track Bugs', frame)

        return detections


# faceboxes