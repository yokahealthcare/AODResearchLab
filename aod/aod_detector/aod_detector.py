import math

import cv2
import numpy as np


class ObjectTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

        self.abandoned_temp = {}
        self.abandoned_threshold_frame = 500       # Define how long the system decide if the object abandoned or not

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []
        abandoned_object = []
        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) / 2
            cy = (y + y + h) / 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                distance = math.hypot(cx - pt[0], cy - pt[1])

                if distance < 25:
                    # update the center point
                    self.center_points[id] = (cx, cy)

                    objects_bbs_ids.append([x, y, w, h, id, distance])
                    same_object_detected = True

                    #   Add same object to the abandoned_temp dictionary. if the object is
                    #   still in the temp dictionary for certain threshold count then
                    #   the object will be considered as abandoned object
                    if id in self.abandoned_temp:
                        if distance < 1:
                            if self.abandoned_temp[id] > self.abandoned_threshold_frame:
                                abandoned_object.append([id, x, y, w, h, distance])
                            else:
                                self.abandoned_temp[id] += 1  # Increase count for the object

                    break

            # If new object is detected then assign the ID to that object
            if same_object_detected is False:
                # print(False)
                self.center_points[self.id_count] = (cx, cy)
                self.abandoned_temp[self.id_count] = 1  # Add new object with initial count 1
                objects_bbs_ids.append([x, y, w, h, self.id_count, None])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        abandoned_temp_2 = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, _ = obj_bb_id
            center = self.center_points[object_id]

            new_center_points[object_id] = center

            if object_id in self.abandoned_temp:
                counts = self.abandoned_temp[object_id]
                abandoned_temp_2[object_id] = counts

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        self.abandoned_temp = abandoned_temp_2.copy()
        return objects_bbs_ids, abandoned_object


tracker = ObjectTracker()


class AodDetector:
    def __init__(self):
        self.first_frame = None
        self.first_frame_gray = None
        self.first_frame_blur = None

        self.frame_difference = None
        self.edged = None
        self.threshold = None
        self.contour = None

    @staticmethod
    def calculate_difference(frame1, frame2):
        return cv2.absdiff(frame1, frame2)

    @staticmethod
    def calculate_canny_edge(frame_difference):
        return cv2.Canny(frame_difference, 100, 200, L2gradient=True)

    @staticmethod
    def calculate_morphology_ex(canny_edge):
        kernel = np.ones((20, 20), np.uint8)
        return cv2.morphologyEx(canny_edge, cv2.MORPH_CLOSE, kernel, iterations=2)

    @staticmethod
    def calculate_countours(threshold):
        return cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Main
    def detect(self, frame):
        h, w = frame.shape[:2]

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(frame_gray, (3, 3), 0)

        self.frame_difference = self.calculate_difference(self.first_frame_blur, frame_blur)
        self.edged = self.calculate_canny_edge(self.frame_difference)
        self.threshold = self.calculate_morphology_ex(self.edged)
        self.contours, hierarchy = self.calculate_countours(self.threshold)

        boxes_xyxy = []
        for contour in self.contours:
            contour_area = cv2.contourArea(contour)
            if 100 < contour_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                x1 = x
                y1 = y
                x2 = w + x
                y2 = h + y

                boxes_xyxy.append((x1, y1, x2, y2))

        _, abandoned_objects = tracker.update(boxes_xyxy)

        return abandoned_objects

    # Getter & Setter
    def set_first_frame(self, frame):
        self.first_frame = frame
        self.first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.first_frame_blur = cv2.GaussianBlur(self.first_frame_gray, (3, 3), 0)

    def get_variables(self):
        return self.first_frame_blur, self.frame_difference, self.edged, self.threshold