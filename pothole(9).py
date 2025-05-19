import os
import pandas as pd
import numpy as np
import cv2

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Config ---
LABEL_DIR = "runs4/predict/labels"
IMAGE_DIR = "runs4/predict"
CLASS_NAMES = ["Potholes"]

# --- Step 1: Parse YOLO label files and extract features ---
data = []

for label_file in os.listdir(LABEL_DIR):
    if not label_file.endswith(".txt"):
        continue

    image_name = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(IMAGE_DIR, image_name)
    if not os.path.exists(image_path):
        continue

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(os.path.join(LABEL_DIR, label_file), "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) != 5:
                continue  # skip malformed lines
            cls, x_center, y_center, bbox_w, bbox_h = parts

            abs_x = x_center * w
            abs_y = y_center * h
            abs_w = bbox_w * w
            abs_h = bbox_h * h
            area = abs_w * abs_h
            aspect_ratio = abs_w / abs_h if abs_h != 0 else 0

            data.append({
                "image": image_name,
                "x_center": abs_x,
                "y_center": abs_y,
                "width": abs_w,
                "height": abs_h,
                "area": area,
                "aspect_ratio": aspect_ratio
            })

df = pd.DataFrame(data)

if df.empty:
    print("No detections found. Exiting.")
    exit()

# --- Step 2: Assign severity labels using percentiles ---
q1, q2 = np.percentile(df["area"], [33, 66])
df["severity_label"] = df["area"].apply(
    lambda a: "Low" if a <= q1 else "Medium" if a <= q2 else "High"
)

# --- Step 3: Train Random Forest classifier ---
X = df[["area", "aspect_ratio"]]
y = df["severity_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

df.loc[X_test.index, "predicted_severity"] = rf.predict(X_test)

# --- Step 4: Save to CSV ---
df.to_csv("pothole_features.csv", index=False)
print("✅ Saved 'pothole_features2.csv' with features and severity labels.")

# --- Step 5: Summary Table by Image ---
# First, map severity to numeric for mean/median calculations
# Step 4.1: Drop duplicate detections for safety
df = df.drop_duplicates()

# Step 5: Group by image and compute mean/median severity
severity_numeric = {"Low": 1, "Medium": 2, "High": 3}
df["severity_numeric"] = df["severity_label"].map(severity_numeric)

# Ensure each image is grouped only once
summary = df.groupby("image").agg({
    "severity_numeric": ["mean", "median"]
}).reset_index()

summary.columns = ["image", "mean_severity", "median_severity"]

# Convert numeric severity back to label
def numeric_to_label(val):
    if val <= 1.5:
        return "Low"
    elif val <= 2.5:
        return "Medium"
    else:
        return "High"

summary["mean_label"] = summary["mean_severity"].apply(numeric_to_label)
summary["median_label"] = summary["median_severity"].apply(numeric_to_label)

# Print summary
print("\n📊 Pothole Severity Summary by Image (No Duplicates):\n")
print(summary[["image", "mean_severity", "mean_label", "median_severity", "median_label"]].to_string(index=False))
