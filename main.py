import cv2
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.predict(source, classes=[24, 25, 26, 28, 31], device=0)[0]


class AODDetector:
    def __init__(self):
        self.short_term = None
        self.long_term = None
        self.previous_short_term = None
        self.previous_long_term = None

    @staticmethod
    def convert_to_grayscale(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def run(self, frame):
        frame = self.convert_to_grayscale(frame)
        frame = frame.astype(np.float32)

        bgr_diff = cv2.absdiff(frame, self.short_term)

        # Update short-term background
        cv2.accumulateWeighted(bgr_diff, self.short_term, 0.01)
        # Update long-term background (slower update)
        cv2.accumulateWeighted(bgr_diff, self.long_term, 0.001)

        # Temporal transition information
        gts = self.long_term > self.short_term
        for y, row in enumerate(gts):
            for x, gt in enumerate(row):
                if gt:
                    self.short_term[y][x] += 1
        lts = self.long_term < self.short_term
        for y, row in enumerate(lts):
            for x, lt in enumerate(row):
                if lt:
                    self.short_term[y][x] -= 1
        eqs = self.long_term == self.short_term
        for y, row in enumerate(eqs):
            for x, eq in enumerate(row):
                if eq:
                    self.short_term[y][x] = self.long_term[y][x]

        # # Temporal transition information
        # gts = self.previous_long_term > self.previous_short_term
        # for y, row in enumerate(gts):
        #     for x, gt in enumerate(row):
        #         if gt:
        #             self.short_term[y][x] += 1
        #
        # lts = self.previous_long_term < self.previous_short_term
        # for y, row in enumerate(lts):
        #     for x, lt in enumerate(row):
        #         if lt:
        #             self.short_term[y][x] -= 1
        #
        # eqs = self.previous_long_term == self.previous_short_term
        # for y, row in enumerate(eqs):
        #     for x, eq in enumerate(row):
        #         if eq:
        #             self.short_term[y][x] = self.previous_short_term[y][x]
        #
        # self.previous_short_term = self.short_term
        # self.previous_long_term = self.long_term

        DF = self.long_term - self.short_term

        # Threshold for foreground detection -> Masking
        DF = 255 - cv2.threshold(DF.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)[1]
        mask = (DF / 255).astype(np.uint8)
        return mask

    def set_short_term(self, frame):
        frame = self.convert_to_grayscale(frame)

        self.previous_short_term = frame.astype(np.float32)
        self.short_term = frame.astype(np.float32)

    def set_long_term(self, frame):
        frame = self.convert_to_grayscale(frame)

        self.previous_long_term = frame.astype(np.float32)
        self.long_term = frame.astype(np.float32)


if __name__ == '__main__':
    yolo = YoloDetector("asset/yolo/yolov8n.pt")
    aod = AODDetector()

    cap = cv2.VideoCapture(0)

    # Initialize short-term and long-term backgrounds (first frame as reference)
    ret, frame = cap.read()
    aod.set_short_term(frame)
    aod.set_long_term(frame)

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        mask = aod.run(frame)
        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('webcam', foreground)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
