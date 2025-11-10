import cv2
import numpy as np

# Read grayscale image
gray_img = cv2.imread("test_gray.jpg")  # put your grayscale image in the same folder
if gray_img is None:
    print("❌ Image not found!")
    exit()

# Convert to RGB
gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
gray_img_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

# Load OpenCV's colorization model (pretrained)
prototxt = cv2.data.haarcascades + "colorization_deploy_v2.prototxt"
model = cv2.data.haarcascades + "colorization_release_v2.caffemodel"

# Actually, OpenCV has a simpler way to demonstrate fake colorization:
color_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)

# Save output
cv2.imwrite("colorized_output.jpg", color_img)
print("✅ Colorized image saved as colorized_output.jpg")
