import cv2
import numpy as np

# Load pre-trained model and config files (download from OpenCV GitHub or use your own)
model_path = 'MobileNetSSD_deploy.caffemodel'
config_path = 'MobileNetSSD_deploy.prototxt'

# Class labels MobileNet SSD was trained on (COCO-like classes)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Load your image
image = cv2.imread('images/images (1).jpg')
(h, w) = image.shape[:2]

# Prepare input blob for detection
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                             0.007843, (300, 300), 127.5)

net.setInput(blob)
detections = net.forward()

car_box = None
confidence_threshold = 0.5

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > confidence_threshold:
        idx = int(detections[0, 0, i, 1])

        # Check if detected object is a car (class id 7 in MobileNetSSD)
        if CLASSES[idx] == "car":
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            car_box = (startX, startY, endX - startX, endY - startY)
            break  # Take first detected car only

if car_box is None:
    raise ValueError("No car detected in the image.")

print(f"Detected car bounding box: {car_box}")

# Now you can proceed to detect pothole by other means (manual or contour detection)
# For example, manual ROI:
clone = image.copy()
cv2.imshow("Image", clone)
print("Please select pothole bounding box manually...")
pothole_box = cv2.selectROI("Image", clone, False)
cv2.destroyAllWindows()

if pothole_box == (0, 0, 0, 0):
    raise ValueError("No pothole selected.")

# Calculate relative depth
_, _, car_w, car_h = car_box
_, _, pothole_w, pothole_h = pothole_box

car_dim = max(car_w, car_h)
pothole_dim = max(pothole_w, pothole_h)

depth_percentage = (pothole_dim / car_dim) * 100

print(f"Estimated pothole depth: {depth_percentage:.2f}% of car size")
