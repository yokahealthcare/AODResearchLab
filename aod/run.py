import os
import time

import cv2

from aod_detector.aod_detector import AodDetector
from yolo.yolo_detector import YoloDetector


def get_hour_minute_second(elapsed_time):
    minute, second = divmod(elapsed_time, 60)
    hour, minute = divmod(minute, 60)

    hour, minute, second = str(hour).zfill(2), str(minute).zfill(2), str(second).zfill(2)
    return hour, minute, second


if __name__ == '__main__':
    FOLDER_PATH = "sample"
    FILENAME = "video9"

    cap = cv2.VideoCapture(f"{FOLDER_PATH}/video/{FILENAME}.avi")
    yolo = YoloDetector("yolo/model/yolov8n.pt")
    aod = AodDetector()

    # Load the backgrounds image from folder
    backgrounds = os.listdir(f"{FOLDER_PATH}/background/{FILENAME}")
    backgrounds = [i.split(".")[0] for i in backgrounds]  # Take out the extension
    backgrounds.sort()

    """
        Setting up first frame
    """
    # Option 1: You can use the first frame within VideoCapture()
    # _, frame = cap.read()
    # aod.set_first_frame(frame)

    # Option 2: You can individual initialize with imread()
    frame = cv2.imread(f"{FOLDER_PATH}/background/{FILENAME}/{backgrounds[0]}.png")
    aod.set_first_frame(frame)

    elapsed = 0
    start = time.time()
    while True:
        return_value = []
        has_frame, frame = cap.read()
        if not has_frame:
            break

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
                _class = yolo.model.names[class_idx[0]]  # We choose the first index only for class

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

        end = time.time()
        if end - start > 1:
            elapsed += 1
            hour, minute, second = get_hour_minute_second(elapsed)
            current_duration = f"{hour}:{minute}:{second}"

            if current_duration in backgrounds:
                print("Changing background")
                index = backgrounds.index(current_duration)
                frame = cv2.imread(f"{FOLDER_PATH}/background/{FILENAME}/{backgrounds[index]}.png")
                aod.set_first_frame(frame)

            print(f"Current Duration : {current_duration}")
            start = end

        cv2.imshow("Original", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
