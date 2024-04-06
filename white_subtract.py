import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture("../asset/video/video1.avi")

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        cv2.imshow('webcam', frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
