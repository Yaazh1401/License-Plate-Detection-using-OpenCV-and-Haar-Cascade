# License-Plate-Detection-using-OpenCV-and-Haar-Cascade

## NAME - YAAZHINI S
## REG NO - 212224230308

## Project Overview
This project implements a License Plate Detection system using OpenCV’s Haar Cascade Classifier.

The model identifies and locates vehicle license plates in an input image, draws bounding boxes, and extracts (crops) the plate region for further analysis.

The Haar Cascade used is haarcascade_russian_plate_number.xml — a pre-trained classifier provided by OpenCV.

## Algorithm
Read the input image containing the vehicle using OpenCV.

Convert the image to grayscale to simplify processing.

Load the Haar Cascade classifier for license plate detection.

Apply the classifier using detectMultiScale() to locate plate regions.

Draw bounding boxes around the detected license plates.

Crop and save the detected plate area as a separate image for further use.

## Program

# Step 1: Import Libraries
```
import cv2
import os
import urllib.request
import matplotlib.pyplot as plt
```
# Replace 'car.jpg' with your test image filename
```
img = cv2.imread("C:\\Users\\admin\\Downloads\\car.jpg")
```
# Convert image from BGR to RGB for display
```
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Input Image")
plt.axis('off')
plt.show()


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
cascade_file = "haarcascade_russian_plate_number.xml"

if not os.path.exists(cascade_file):
    urllib.request.urlretrieve(cascade_url, cascade_file)
    print(" Haar Cascade downloaded successfully.")
else:
    print(" Haar Cascade file found.")

plate_cascade = cv2.CascadeClassifier(cascade_file)

if plate_cascade.empty():
    raise IOError(" Haar Cascade failed to load. Check file path or download again.")
else:
    print(" Haar Cascade loaded successfully.")
```

# Apply Gaussian blur and histogram equalization to improve detection
```
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
gray_eq = cv2.equalizeHist(gray_blur)

plt.imshow(gray_eq, cmap='gray')
plt.title("Preprocessed Image")
plt.axis('off')
plt.show()

plates = plate_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5)

print(f"Detected {len(plates)} plate(s).")
```

# Draw bounding boxes on a copy of the original image
```
output_img = img_rgb.copy()

for (x, y, w, h) in plates:
    cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    plate_region = img[y:y+h, x:x+w]
    cv2.imwrite(f"plate_{x}_{y}.png", plate_region)  # Save cropped plate image

plt.imshow(output_img)
plt.title("Detected License Plate(s)")
plt.axis('off')
plt.show()
```


## Output

<img width="863" height="809" alt="image" src="https://github.com/user-attachments/assets/97c41f19-416a-4355-85d4-3204e42e25db" />

<img width="951" height="403" alt="image" src="https://github.com/user-attachments/assets/770f4c97-7457-4a49-9458-606c2ccf627f" />

<img width="929" height="389" alt="image" src="https://github.com/user-attachments/assets/5b2bcf32-f9a5-46b6-b7f4-bab78b986f8f" />

## Result
The Haar Cascade classifier successfully detected the license plate region from the input image. After preprocessing (Gaussian Blur and Histogram Equalization), the detection became more stable and accurate.
