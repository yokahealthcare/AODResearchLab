import time

import cv2
import numpy as np

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    _, first_frame = cap.read()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # absdiff
        bgr_diff = cv2.absdiff(frame, first_frame)

        current_time = time.time()
        if current_time - start_time > 5:

            first_frame = frame
            start_time = current_time

        result_frame = np.concatenate([frame, first_frame, bgr_diff], axis=1)
        cv2.imshow('webcam', result_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
