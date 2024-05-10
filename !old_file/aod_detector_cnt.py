import cv2

from aod_with_dual_background.aod_detector.object_tracker import ObjectTracker

tracker = ObjectTracker()


class AodDetectorCnt:
    def __init__(self):
        self.cnt = cv2.bgsegm.createBackgroundSubtractorCNT(useHistory=True)

        self.mask = None
        self.countours = None

        self.n_frames = 0

    @staticmethod
    def calculate_countours(threshold):
        return cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def detect(self, frame, lr=-1):
        self.n_frames += 1
        if self.n_frames > 1000:
            lr = 0.0001

        self.mask = self.cnt.apply(frame, learningRate=lr)

        self.contours, hierarchy = self.calculate_countours(self.mask)
        boxes_xyxy = []
        for contour in self.contours:
            contour_area = cv2.contourArea(contour)
            if 10 < contour_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                boxes_xyxy.append((x, y, w, h))

        _, abandoned_objects = tracker.update(boxes_xyxy)
        return abandoned_objects

    def get_variables(self):
        return self.mask, self.cnt.getBackgroundImage()

    def set_threshold_difference(self, new_threshold):
        self.threshold_difference = new_threshold
