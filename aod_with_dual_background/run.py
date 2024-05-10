from collections import deque

import cv2
import numpy as np
# import cupy as cp
import torch

from aod_detector import AodDetector
from yolo import YoloDetector

device = torch.device("cuda:0" if torch.cuda.is_available() else "")

def get_median_filter_tensor(images):
    # Convert images to PyTorch tensors (assuming NumPy arrays)
    images_tensor = torch.as_tensor(np.asarray(images)).to(device)  # Convert to float for calculations

    # Reshape to combine frames along a new dimension (efficient for stacking)
    images_tensor = images_tensor.permute(1, 2, 0)  # Equivalent to transpose with PyTorch

    # Flatten the tensor to calculate medians efficiently
    flattened_tensor = images_tensor.view(-1, len(images)).to(device)

    # Calculate the median along the feature dimension (axis=1)
    medians, _ = torch.median(flattened_tensor, dim=1)
    medians = medians.to(device)

    # Reshape the medians back to the original image shape
    median_tensor = medians.view(images_tensor.shape[0], images_tensor.shape[1])

    # Convert PyTorch tensor to NumPy array
    median_array = median_tensor.detach().cpu().numpy().astype(np.uint8)

    return median_array  # Return PyTorch tensor


# def get_median_filter_numpy(images):
#     images = cp.asarray(images)
#
#     # Reshape to combine frames along a new axis (efficient for stacking)
#     result = images.transpose([1, 2, 0])
#
#     # Reshape the array to flatten each element pair
#     flattened_arr = result.reshape(-1, len(images))
#     # Calculate the median along the columns (axis=1)
#     medians = cp.median(flattened_arr, axis=1)
#     # Reshape the medians back to the original
#     median_array = medians.reshape(result.shape[0], result.shape[1])
#     return cp.asnumpy(median_array).astype(np.uint8)


if __name__ == '__main__':
    cap = cv2.VideoCapture("sample/video/video11.avi")
    yolo = YoloDetector("yolo/model/yolov8n.pt")
    aod = AodDetector()

    n = 1000  # Number of frames for the long term background
    last_n_frames = deque(maxlen=n)

    n_frames = 0
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        n_frames += 1

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        if n_frames >= n:
            return_value = []

            longterm = get_median_filter_tensor(last_n_frames)
            cv2.imshow("long term", longterm)

            # Detect abandoned objects
            abandoned_objects = aod.detect(frame, longterm)

            # Plotting all variable inside the AOD Calculation (Optional, just for visualisation)
            frame_difference, edged, dilate, threshold = aod.get_variables()
            # frame_difference, edged, threshold = aod.get_variables()
            cv2.imshow("Frame Difference", frame_difference)
            cv2.imshow("Canny Edge", edged)
            cv2.imshow("Dilate", dilate)
            cv2.imshow("Morphology Ex", threshold)

            # XYXY Coordinate of abandoned object
            for _, x, y, w, h, _ in abandoned_objects:
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=1)

                cropped = frame[y1:y2, x1:x2]

                # Yolo detector
                # result = yolo.run(cropped)
                # class_idx = result.boxes.cls.clone().tolist()
                # if len(class_idx) == 0:
                #     _class = "Unknown"
                # else:
                #     _class = yolo.model.names[class_idx[0]]  # We choose the first index only for class

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

        last_n_frames.append(frame)

        cv2.imshow("Original", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
