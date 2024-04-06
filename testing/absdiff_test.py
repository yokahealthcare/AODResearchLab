import cv2

# Load two images (you can replace 'left.jpg' and 'right.jpg' with your own image filenames)
image1 = cv2.imread("../asset/image/aerith#1.jpg")  # Load as grayscale
image2 = cv2.imread("../asset/image/aerith#1.jpg")  # Load as grayscale

# Calculate the per-element absolute difference between the two images
diff = cv2.absdiff(image1, image2)
print(diff.shape)

# Display the resulting difference image
cv2.imshow("Difference Image", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
