import cv2

# Initialize the background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# Set the accumulation weight
accum_weight = 0.01

# Start capturing video
cap = cv2.VideoCapture("../asset/video/video#1.mp4")

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Apply the background subtractor
    fgMask = backSub.apply(frame, learningRate=accum_weight)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
