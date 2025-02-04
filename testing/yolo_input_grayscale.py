import cv2
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.predict(source, device=0)[0]


if __name__ == '__main__':
    yolo = YoloDetector("../asset/yolo/yolov8l.pt")

    cap = cv2.VideoCapture(0)
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = yolo.run(frame)
        cv2.imshow("webcam", result.plot())
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
