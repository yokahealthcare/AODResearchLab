from collections import deque

import cv2

# Define buffer size (number of recent frames to store)
buffer_size = 100  # Adjust based on desired slow-motion effect and processing power

# Open video capture source (replace 0 with your camera index or video file path)
cap = cv2.VideoCapture("../asset/video/video#1.mp4")

# Check if video capture opened successfully
if not cap.isOpened():
    print("Error opening video capture")
    exit()

# Window names
current_frame_window = "Current Frame"
slow_motion_window = "Slow Motion"

# Create windows for displaying frames
cv2.namedWindow(current_frame_window, cv2.WINDOW_NORMAL)
cv2.namedWindow(slow_motion_window, cv2.WINDOW_NORMAL)

# Initialize variables
frame_buffer = deque(maxlen=buffer_size)  # Use deque for efficient buffer management
prev_blend_frame = None

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Check if frame is read successfully
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Update frame buffer
    frame_buffer.append(frame.copy())

    # Blend current frame with a frame from the buffer (if available)
    if len(frame_buffer) >= buffer_size:
        # Select frame to blend with based on buffer size and desired slow-motion effect
        blend_frame = frame_buffer[0]  # Adjust index for different slow-motion speeds
        if prev_blend_frame is not None:
            slow_motion_frame = cv2.addWeighted(prev_blend_frame, 0.8, frame, 0.01, 0)
        else:
            slow_motion_frame = frame.copy()  # Use current frame initially
        prev_blend_frame = slow_motion_frame  # Update for next blending iteration
    else:
        # Not enough frames in buffer, display current frame for now
        slow_motion_frame = frame.copy()

    # Display current and slow-motion frames
    cv2.imshow(current_frame_window, frame)
    cv2.imshow(slow_motion_window, slow_motion_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
