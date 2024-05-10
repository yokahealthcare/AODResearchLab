import cv2
import numpy as np

if __name__ == '__main__':
    # Sample frames (replace with your actual frames)
    frame1 = np.array([[255, 255, 255], [180, 200, 220]], dtype=np.uint8)
    frame2 = np.array([[80, 100, 130], [160, 180, 200]], dtype=np.uint8)

    # Calculate absolute difference
    absdiff = cv2.absdiff(frame1, frame2)
    print(absdiff)
    cv2.imshow('before', absdiff)

    # Factor for decrease (1 - 0.2 = 0.8)
    factor = 1 - 0.7

    # Apply decrease using element-wise multiplication with clipping
    decreased_absdiff = np.clip(absdiff * factor, 0, 255)  # Clip to valid pixel range
    decreased_absdiff = decreased_absdiff.astype(np.uint8)
    # You can now use decreased_absdiff for further processing
    cv2.imshow('after', decreased_absdiff)
    print(decreased_absdiff)

    cv2.waitKey(0)
