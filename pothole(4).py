from inference_sdk import InferenceHTTPClient
import cv2

# Initialize client with your API key
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="rf_pfHRnz1D7KThkw7hezE0oO568x32"  # Replace with your actual Roboflow API key
)

# Image path (can be local path or URL)
image_path = "images (1).jpg"

# Run inference on the image using the pothole detection model from Roboflow
result = CLIENT.infer(image_path, model_id="pothole-detection-hw8pl/2")

# Load image for drawing
image = cv2.imread(image_path)
height, width, _ = image.shape

# Parse predictions
potholes = []
others = []

for pred in result["predictions"]:
    class_name = pred["class"]
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    area = w * h

    if class_name.lower() == "pothole":
        potholes.append(area)
    else:
        others.append(area)

# Calculate max other object area for rating comparison
max_other_area = max(others) if others else (width * height)

# Rate potholes and draw bounding boxes + rating on image
for i, pred in enumerate(result["predictions"]):
    if pred["class"].lower() == "pothole":
        x1 = int(pred["x"] - pred["width"] / 2)
        y1 = int(pred["y"] - pred["height"] / 2)
        x2 = int(pred["x"] + pred["width"] / 2)
        y2 = int(pred["y"] + pred["height"] / 2)
        area = pred["width"] * pred["height"]
        rating = int((area / max_other_area) * 100)
        rating = min(rating, 99)

        # Draw box and rating
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Rating: {rating}/100', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        print(f"Pothole {i+1} rating: {rating}/100")

# Save output image with ratings
cv2.imwrite("output_with_ratings.jpg", image)
print("Output saved to output_with_ratings.jpg")
