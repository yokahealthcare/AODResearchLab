import cv2
import numpy as np

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Create the background subtractor object
# Use the last 70 frames to build the background
backSub = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=100, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background object on each frame
    fgMask = 255 - backSub.apply(image=frame, learningRate=-1)

    # Mark the static objects on the frame
    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Show the current frame and the fg masks
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    # Quit if 'q' is pressed
    if cv2.waitKey(30) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
