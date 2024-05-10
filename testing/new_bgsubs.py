import math
from collections import deque

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
        self.abandoned_threshold_frame = 500  # Define how long the system decide if the object abandoned or not

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

def get_median_very_new(images):
    images = np.asarray(images)

    # Reshape to combine frames along a new axis (efficient for stacking)
    result = images.transpose([1, 2, 0]).astype(np.uint8)

    # Reshape the array to flatten each element pair
    flattened_arr = result.reshape(-1, len(images))
    # Calculate the median along the columns (axis=1)
    medians = np.median(flattened_arr, axis=1)
    # Reshape the medians back to the original
    median_array = medians.reshape(result.shape[0], result.shape[1])
    return median_array.astype(np.uint8)


def get_median_old(images, n):
    result = np.zeros(shape=(height, width, n))
    for idx, frame_ in enumerate(images):
        for y in range(frame_.shape[0]):
            for x in range(frame_.shape[1]):
                pixel = frame_[y][x]
                result[y][x][idx] = pixel

    # Reshape the array to flatten each element pair
    flattened_arr = result.reshape(-1, n)
    # Calculate the median along the columns (axis=1)
    medians = np.median(flattened_arr, axis=1)
    # Reshape the medians back to the original shape
    median_array = medians.reshape(result.shape[0], result.shape[1]).astype(np.uint8)
    return median_array


def get_median_new(images, n):
    # Convert deque to NumPy array for efficient operations
    frames_array = np.asarray(images)
    # Calculate the median along the last axis (frames) using reshape and transpose
    median_array = np.median(frames_array.reshape(-1, n), axis=1).astype(np.uint8)
    return median_array


def get_temporal_minimum(images):
    images = np.asarray(images)

    # Reshape to combine frames along a new axis (efficient for stacking)
    result = images.transpose([1, 2, 0]).astype(np.uint8)

    # Reshape the array to flatten each element pair
    flattened_arr = result.reshape(-1, len(images))
    # Calculate the median along the columns (axis=1)
    medians = np.min(flattened_arr, axis=1)
    # Reshape the medians back to the original
    median_array = medians.reshape(result.shape[0], result.shape[1])
    return median_array.astype(np.uint8)


def calculate_canny_edge(frame_difference):
    return cv2.Canny(frame_difference, 100, 200, L2gradient=True)

def calculate_morphology_ex(canny_edge):
    kernel = np.ones((20, 20), np.uint8)
    return cv2.morphologyEx(canny_edge, cv2.MORPH_CLOSE, kernel, iterations=2)

def calculate_countours(threshold):
    return cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if __name__ == '__main__':
    # Create the background subtractor object
    backSub = cv2.createBackgroundSubtractorMOG2()  # You can choose other algorithms here

    cap = cv2.VideoCapture("../asset/video/video1.mp4")
    # Get video width and height using property access
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = 200
    m = 20
    threshold = 100
    last_n_frames = deque(maxlen=n)
    last_n_frames_short = deque(maxlen=m)

    last_n_masks = deque(maxlen=m)

    n_frames = 0
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        n_frames += 1

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        if n_frames >= n:
            long_term = get_median_very_new(last_n_frames)
            cv2.imshow('long term', long_term)

            # short_term = get_median_very_new(last_n_frames_short)
            # cv2.imshow('short term', short_term)

            # difference = cv2.the(frame - long_term) < 0, 0, 255).astype(np.uint8)
            # difference = cv2.threshold((frame - long_term).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)[1]

            difference = cv2.absdiff(long_term, frame)
            ret, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
            cv2.imshow('difference', difference)

            canny = calculate_canny_edge(difference)
            morph = calculate_morphology_ex(canny)
            cv2.imshow("morph", morph)
            contours, hierarchy = calculate_countours(morph)

            boxes_xyxy = []
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if 100 < contour_area:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    x1 = x
                    y1 = y
                    x2 = w + x
                    y2 = h + y

                    boxes_xyxy.append((x1, y1, x2, y2))

            _, abandoned_objects = tracker.update(boxes_xyxy)
            for _, x1, y1, x2, y2, _ in abandoned_objects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=1)


            # Apply background subtraction
            # fgMask = backSub.apply(frame)
            # cv2.imshow('fgMask', fgMask)

            # print('long term', long_term.shape, long_term.min(), long_term.max(), long_term.dtype)
            # print('frame', frame.shape, frame.min(), frame.max(), frame.dtype)

            # # difference = np.abs(frame - long_term)
            # difference = cv2.absdiff(frame, long_term)
            # # print(difference, difference.min(), difference.max(), difference.mean(), difference.dtype)
            # mask = np.where(difference >= threshold, 255, 0).astype(np.uint8)
            # # print(mask)
            # # cv2.imshow('mask', mask)
            #
            # last_n_masks.append(mask)
            # if len(last_n_masks) == m:
            #     # Short term
            #     short_term = get_temporal_minimum(last_n_masks)
            #     cv2.imshow('short term', short_term)

        last_n_frames.append(frame)
        if len(last_n_frames) >= 80:
            last_n_frames_short.append(frame)

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
