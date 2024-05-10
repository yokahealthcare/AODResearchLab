import cv2
import torch

from aod_detector import AodDetectorMog
from yolo import YoloDetector

device = torch.device("cuda:0" if torch.cuda.is_available() else "")

if __name__ == '__main__':
    cap = cv2.VideoCapture("../aod_with_dual_background/sample/video/video10.avi")
    yolo = YoloDetector("yolo/model/yolov8n.pt")
    aod = AodDetectorMog()

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        return_value = []

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (7, 7), 0)

        abandoned_objects = aod.detect(frame)

        # Plotting all variable inside the AOD Calculation (Optional, just for visualisation)
        mask, background = aod.get_variables()
        cv2.imshow("Mask", mask)
        cv2.imshow("Background", background)

        # XYXY Coordinate of abandoned object
        for _, x, y, w, h, _ in abandoned_objects:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=1)

            cropped = frame[y1:y2, x1:x2]

            _class = "unknown"

            result_dict = {
                "region": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "class": _class,
                "image": cropped
            }
            return_value.append(result_dict)

        print(return_value)

        cv2.imshow("Original", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
