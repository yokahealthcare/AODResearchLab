import cv2
import numpy as np

from .object_tracker import ObjectTracker

class AodDetector:
    def __init__(self, abandoned_threshold_frame=500):
        self.tracker = ObjectTracker(abandoned_threshold_frame)

        self.frame_difference = None
        self.edged = None
        self.dilate = None
        self.morph = None
        self.contours = None

        self.threshold_difference = 0.0

    @staticmethod
    def calculate_difference(frame1, frame2):
        return cv2.absdiff(frame1, frame2)

    @staticmethod
    def calculate_difference_average(difference, frame_width, frame_height):
        return np.sum(difference) / (frame_width * frame_height)

    @staticmethod
    def calculate_canny_edge(frame_difference):
        return cv2.Canny(frame_difference, 100, 210, L2gradient=True)

    @staticmethod
    def calculate_dilation(canny_edge):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(canny_edge, kernel, iterations=1)

    @staticmethod
    def calculate_morphology_ex(canny_edge):
        kernel = np.ones((20, 20), np.uint8)
        return cv2.morphologyEx(canny_edge, cv2.MORPH_CLOSE, kernel, iterations=2)

    @staticmethod
    def calculate_countours(threshold):
        return cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Main
    def detect(self, current_frame, longterm_frame):
        height, width = current_frame.shape[:2]

        self.frame_difference = self.calculate_difference(longterm_frame, current_frame)

        # Decrease the amount of threshold_difference to self.frame_difference
        # For example, if set to 0.2, then the frame_difference value will be decreased by 20% from its orginal value
        factor = 1 - self.threshold_difference
        self.frame_difference = np.clip(self.frame_difference * factor, 0, 255).astype(np.uint8)

        self.edged = self.calculate_canny_edge(self.frame_difference)
        self.dilate = self.calculate_dilation(self.edged)
        self.morph = self.calculate_morphology_ex(self.dilate)
        self.contours, hierarchy = self.calculate_countours(self.morph)

        boxes_xyxy = []
        for contour in self.contours:
            contour_area = cv2.contourArea(contour)
            if 10 < contour_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                boxes_xyxy.append((x, y, w, h))

        _, abandoned_objects = self.tracker.update(boxes_xyxy)
        return abandoned_objects

    def get_variables(self):
        return self.frame_difference, self.edged, self.dilate, self.morph

    def set_threshold_difference(self, new_threshold):
        self.threshold_difference = new_threshold
