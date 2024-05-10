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
    cap = cv2.VideoCapture("sample/video/video2.avi")
    yolo = YoloDetector("yolo/model/yolov8n.pt")
    aod = AodDetector()

    mog = cv2.createBackgroundSubtractorMOG2(history=4000, detectShadows=False)
    knn = cv2.createBackgroundSubtractorKNN(history=4000, detectShadows=False)
    # cnt = cv2.createBackgroundSubtractorCNT(nComponents=50)

    background = cv2.imread("../aod/sample/background/video2/00:00:00.png", 0)
    canny_t1 = 100
    canny_t2 = 210

    morph_kernel = 20
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (9, 9), 0)

        # frame_difference = cv2.absdiff(frame, background)
        frame_difference = knn.apply(frame)
        cv2.imshow("Frame Difference", frame_difference)
        #
        # canny = cv2.Canny(frame_difference, canny_t1, canny_t2, L2gradient=True)
        # cv2.imshow("Canny", canny)
        #
        # kernel = np.ones((5, 5), np.uint8)
        # dilate = cv2.dilate(canny, kernel, iterations=1)
        # cv2.imshow("Dilate", dilate)
        #
        # kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        # morph = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel, iterations=2)
        # cv2.imshow("Morph", morph)


        cv2.imshow("Original", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("w"):
            canny_t1 += 10
            print(f"Canny T1 increase 10: {canny_t1}")
        elif key == ord("s"):
            canny_t1 -= 10
            print(f"Canny T1 decrease 10: {canny_t1}")
        elif key == ord("e"):
            canny_t2 += 10
            print(f"Canny T2 increase 10: {canny_t2}")
        elif key == ord("d"):
            canny_t2 -= 10
            print(f"Canny T2 decrease 10: {canny_t2}")

        elif key == ord("r"):
            morph_kernel += 1
            print(f"Morph kernel increase 1: {morph_kernel}")
        elif key == ord("f"):
            morph_kernel -= 1
            print(f"Morph kernel decrease 1: {morph_kernel}")

