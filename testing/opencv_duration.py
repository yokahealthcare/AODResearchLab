import time

import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture("../asset/video/lagu#1.mp4")
    elapsed = 0

    start = time.time()
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        cv2.imshow("webcam", frame)

        end = time.time()
        if end - start > 1:
            elapsed += 1
            minutes, seconds = divmod(elapsed, 60)
            hours, minutes = divmod(minutes, 60)

            hours, minutes, seconds = str(hours).zfill(2), str(minutes).zfill(2), str(seconds).zfill(2)

            print(f"Elapsed time: {hours}:{minutes}:{seconds}")

            start = end

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()