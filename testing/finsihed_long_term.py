from collections import deque

import cv2
import numpy as np


def get_median_very_new(images):
    images = np.asarray(images)

    # Reshape to combine frames along a new axis (efficient for stacking)
    result = images.transpose([1, 2, 0]).astype(np.uint8)

    # Reshape the array to flatten each element pair
    flattened_arr = result.reshape(-1, len(images))
    # Calculate the median along the columns (axis=1)
    medians = np.median(flattened_arr, axis=1)
    # Reshape the medians back to the original
    median_array = medians.reshape(result.shape[0], result.shape[1])
    return median_array.astype(np.uint8)

if __name__ == '__main__':
    cap = cv2.VideoCapture("../asset/video/aod#5.mp4")
    # Get video width and height using property access
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = 100
    m = 20
    threshold = 100
    last_n_frames = deque(maxlen=n)

    last_n_masks = deque(maxlen=m)

    n_frames = 0
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        n_frames += 1

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_n_frames.append(frame)

        if n_frames > n:
            long_term = get_median_very_new(last_n_frames)
            cv2.imshow('long term', long_term)

            # print('long term', long_term.shape, long_term.min(), long_term.max(), long_term.dtype)
            # print('frame', frame.shape, frame.min(), frame.max(), frame.dtype)

            # difference = np.abs(frame - long_term)
            # difference = cv2.absdiff(frame, long_term)
            # print(difference, difference.min(), difference.max(), difference.mean(), difference.dtype)
            # mask = np.where(difference > threshold, 255, 0).astype(np.uint8)
            # print(mask)
            # cv2.imshow('mask', mask)

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
