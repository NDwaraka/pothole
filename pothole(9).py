import os
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# --- Config ---
LABEL_DIR = r"C:\Users\Lenovo\Documents\vnr-proj\pothole\runs6\predict\labels"
IMAGE_DIR = r"C:\Users\Lenovo\Documents\vnr-proj\pothole\runs6\predict"
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
                continue  # Skip malformed lines
            cls, x_center, y_center, bbox_w, bbox_h = parts

            abs_x = x_center * w
            abs_y = y_center * h
            abs_w = bbox_w * w
            abs_h = bbox_h * h
            area = abs_w * abs_h
            aspect_ratio = abs_w / abs_h if abs_h != 0 else 0
            perimeter = 2 * (abs_w + abs_h)
            solidity = area / (w * h)
            rectangularity = area / (abs_w * abs_h) if abs_w * abs_h != 0 else 0

            data.append({
                "image": image_name,
                "x_center": abs_x,
                "y_center": abs_y,
                "width": abs_w,
                "height": abs_h,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "perimeter": perimeter,
                "solidity": solidity,
                "rectangularity": rectangularity
            })

df = pd.DataFrame(data)

if df.empty:
    print("No detections found. Exiting.")
    exit()

# --- Step 2: Normalize and Cluster ---
features = df[["area", "aspect_ratio"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

dbscan = DBSCAN(eps=1.0, min_samples=2)
df["severity_cluster"] = dbscan.fit_predict(X_scaled)

# --- Step 3: Assign severity using percentiles ---
q1, q2 = np.percentile(df["area"], [33, 66])
df["severity_label"] = df["area"].apply(
    lambda a: "Low" if a <= q1 else "Medium" if a <= q2 else "High"
)

# --- Step 4: Train/Test Split ---
X = df[["area", "aspect_ratio", "perimeter", "solidity", "rectangularity"]]
y = df["severity_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 5: Handle Imbalanced Data ---
print("\n Class distribution:\n", y_train.value_counts())

# --- Step 6: Grid Search for Best Parameters ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print("\n Best parameters:", grid_search.best_params_)

# --- Step 7: Train Final Model ---
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
df.loc[X_test.index, "predicted_severity"] = y_pred

# --- Step 8: Save to CSV ---
df.to_csv("pothole_features_improved.csv", index=False)
print(" Saved 'pothole_features_improved.csv' with enriched features and predictions.")

# --- Step 9: Summary by Image ---
df["severity_numeric"] = df["severity_label"].map({"Low": 1, "Medium": 2, "High": 3})
summary = df.groupby("image").agg({
    "severity_numeric": ["mean", "median"]
}).reset_index()
summary.columns = ["image", "mean_severity", "median_severity"]

def numeric_to_label(val):
    if val <= 1.5:
        return "Low"
    elif val <= 2.5:
        return "Medium"
    else:
        return "High"

summary["mean_label"] = summary["mean_severity"].apply(numeric_to_label)
summary["median_label"] = summary["median_severity"].apply(numeric_to_label)

print("\n Pothole Severity Summary by Image:\n")
print(summary[["image", "mean_severity", "mean_label", "median_severity", "median_label"]].to_string(index=False))

# --- Step 10: Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {accuracy:.2f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:\n", cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_rf.classes_, yticklabels=best_rf.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --- Step 11: Feature Importance ---
importances = best_rf.feature_importances_
features = X.columns
print("\n Feature Importances:")
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.4f}")

sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.show()
