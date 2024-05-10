from ultralytics import YOLO


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)
        self.result = None

    def run(self, source):
        self.result = self.model.predict(source, classes=[24, 25, 26, 28, 31], device=0)[0]
        return self.result
