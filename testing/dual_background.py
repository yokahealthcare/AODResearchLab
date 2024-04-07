import cv2
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.predict(source, classes=[24, 25, 26, 28, 31], conf=0.8, device=0)[0]


def dual_background_model(video_capture, alpha=0.5, learning_rate_short=0.01, learning_rate_long=0.0001):
    """
  Implements a dual background model for real-time abandoned object detection from video streams.

  Args:
      video_capture (cv2.VideoCapture): OpenCV video capture object.
      alpha (float, optional): Blending weight between short-term and long-term backgrounds. Defaults to 0.5 (equal weight).
      learning_rate_short (float, optional): Learning rate for updating the short-term background. Defaults to 0.01.
      learning_rate_long (float, optional): Learning rate for updating the long-term background. Defaults to 0.001.

  Yields:
      numpy.ndarray: The final frame with foreground objects potentially highlighted in front of a blended background.
  """
    yolo = YoloDetector("../asset/yolo/yolov8l.pt")

    # Initialize short-term and long-term backgrounds (first frame as reference)
    ret, first_frame = video_capture.read()
    if not ret:
        return

    bgr_short = first_frame.copy().astype(np.float32)  # Convert to float for updates
    bgr_long = first_frame.copy().astype(np.float32)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break


        # Update short-term background
        bgr_diff = cv2.absdiff(frame.astype(np.float32), bgr_short)
        cv2.accumulateWeighted(bgr_diff, bgr_short, learning_rate_short)
        # Update long-term background (slower update)
        cv2.accumulateWeighted(bgr_diff, bgr_long, learning_rate_long)

        # Threshold for foreground detection -> Masking
        thresh = cv2.threshold(bgr_short.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)[1]

        # Yolo Object Detection
        # result = yolo.run(thresh_bgr)
        # classes = result.names
        # boxes = result.boxes.xyxy.clone().tolist()
        # cls = result.boxes.cls.clone().tolist()

        # # Blend short-term and long-term backgrounds -> Blending Short & Long Background
        # blended_background = cv2.convertScaleAbs(cv2.addWeighted(bgr_short, alpha, bgr_long, 1 - alpha, 0))
        #
        # # Combine foreground with blended background (optional for visualization)
        # foreground = cv2.bitwise_and(frame, frame, mask=thresh)
        # final_frame = cv2.add(foreground, blended_background)

        # for box, idx_label in zip(boxes, cls):
        #     x1, y1, x2, y2 = box
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     idx_label = int(idx_label)
        #     cv2.rectangle(final_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        #     cv2.putText(final_frame, f"{classes[idx_label]}", (x1, y2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        # # Optional: Further processing for abandoned object detection (size, duration)
        # # ...

    yield thresh  # Yield for real-time video processing


# Example usage (replace 'path/to/video.mp4' with your video path)
cap = cv2.VideoCapture('../asset/video/video1.avi')

for frame in dual_background_model(cap):
    cv2.imshow("Dual Background Model", frame)
    if cv2.waitKey(20) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
