import cv2
import numpy as np

# Load two images (you can replace 'left.jpg' and 'right.jpg' with your own image filenames)
image1 = cv2.imread("../asset/image/aerith#2.jpg", 0)  # Load as grayscale
image2 = cv2.imread("../asset/image/aerith#2.5.jpg", 0)  # Load as grayscale
blank = np.ones_like(image1) * 255
print("blank")
print(blank)
print(blank.shape)

# Calculate the per-element absolute difference between the two images
diff = cv2.absdiff(image1, image2)
cv2.imshow("Diff", diff)
print(diff)
print(diff.shape)
print(np.sum(diff) / (image1.shape[0] * image2.shape[1]))
print("----------------")
edge = cv2.Canny(diff, 5, 200)

kernel = np.ones((20, 20), np.uint8)
morph = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)

contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("edge")
print(edge)

print("morph")
print(morph)

print("contours")
# print(contours)
print(len(contours[0]))
for x in range(len(contours[0])-1):
    x1, y1 = contours[0][x][0]
    x2, y2 = contours[0][x+1][0]

    cv2.line(blank, (x1, y1), (x2, y2), (0, 0, 0), 1)

print(cv2.contourArea(contours[0]))

# Display the resulting difference image
cv2.imshow("Image1", image1)
cv2.imshow("Image2", image2)

cv2.imshow("Edge", edge)
cv2.imshow("Morph", morph)
cv2.imshow("Countour", blank)

cv2.waitKey(0)
cv2.destroyAllWindows()
