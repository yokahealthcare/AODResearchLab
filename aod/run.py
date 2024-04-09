import cv2

from yolo.yolo_detector import YoloDetector
from aod_detector.aod_detector import AodDetector

if __name__ == '__main__':
    yolo = YoloDetector("yolo/model/yolov8n.pt")
    aod = AodDetector()
    cap = cv2.VideoCapture("sample/video/stable_light_scenario.avi")

    """
        Setting up first frame
    """
    # Option 1: You can use the first frame within VideoCapture()
    # _, frame = cap.read()
    # aod.set_first_frame(frame)

    # Option 2: You can individual initialize with imread()
    frame = cv2.imread("sample/background/stable_light_scenario.png")
    aod.set_first_frame(frame)

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        return_value = []

        # Detect abandoned objects
        abandoned_objects = aod.detect(frame)

        # Plotting all variable inside the AOD Calculation (Optional, just for visualisation)
        first_frame_blur, frame_difference, edged, threshold = aod.get_variables()
        cv2.imshow("First Frame", first_frame_blur)
        cv2.imshow("Frame Difference", frame_difference)
        cv2.imshow("Canny Edge Detection", edged)
        cv2.imshow("Morphology Ex", threshold)

        # XYXY Coordinate of abandoned object
        for _, x1, y1, x2, y2, _ in abandoned_objects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=1)

            cropped = frame[y1:y2, x1:x2]

            # Yolo detector
            result = yolo.run(cropped)
            class_idx = result.boxes.cls.clone().tolist()
            if len(class_idx) == 0:
                _class = "Unknown"
            else:
                _class = yolo.model.names[class_idx[0]]     # We choose the first index only for class

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

        print(return_value)         # See the result

        cv2.imshow("Original", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
