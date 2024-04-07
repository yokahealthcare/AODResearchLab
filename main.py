import cv2
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.predict(source, classes=[24, 25, 26, 28, 31], device=0)[0]


if __name__ == '__main__':
    yolo = YoloDetector("asset/yolo/yolov8n.pt")
    cap = cv2.VideoCapture(0)

    # Initialize short-term and long-term backgrounds (first frame as reference)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bgr_short = frame.copy().astype(np.float32)
    bgr_long = frame.copy().astype(np.float32)

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        frame_bgr = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update short-term background
        bgr_diff = cv2.absdiff(frame.astype(np.float32), bgr_short)
        cv2.accumulateWeighted(bgr_diff, bgr_short, 0.02)
        # Update long-term background (slower update)
        cv2.accumulateWeighted(bgr_diff, bgr_long, 0.01)

        DF = bgr_long - bgr_short

        # Threshold for foreground detection -> Masking
        thresh = 255 - cv2.threshold(DF.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)[1]
        mask = (thresh / 255).astype(np.uint8)
        foreground = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

        result = yolo.run(foreground)

        cv2.imshow('webcam', result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
