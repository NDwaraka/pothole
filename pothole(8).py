import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load Video ---
video_path = 'videos/movpot(3).mp4'
cap = cv2.VideoCapture(video_path)

# --- Read two frames ---
ret, frame1 = cap.read()
for _ in range(5):  # skip frames to simulate motion
    ret, frame2 = cap.read()
cap.release()

# --- Grayscale ---
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# --- Feature detection ---
orb = cv2.ORB_create(2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# --- Feature matching ---
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# --- Approximate camera matrix for phone camera ---
h, w = gray1.shape
focal_length = 0.5 * w  # Assumption: focal length in pixels
K = np.array([[focal_length, 0, w / 2],
              [0, focal_length, h / 2],
              [0, 0, 1]])

# --- Essential matrix and pose recovery ---
E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# --- Triangulation ---
pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)

proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
proj2 = np.hstack((R, t))
pts4D = cv2.triangulatePoints(K @ proj1, K @ proj2, pts1.T, pts2.T)
pts3D = pts4D[:3] / pts4D[3]  # convert from homogeneous

# --- Choose two 3D points that are likely on the white line ---
# We'll pick the two closest points in 2D
min_dist = float('inf')
idx1, idx2 = 0, 1
for i in range(len(pts1)):
    for j in range(i + 1, len(pts1)):
        dist = np.linalg.norm(pts1[i] - pts1[j])
        if 15 < dist < 50 and dist < min_dist:  # reasonable pixel distance
            min_dist = dist
            idx1, idx2 = i, j

# --- Real-world reference ---
ref_3D_dist = np.linalg.norm(pts3D[:, idx1] - pts3D[:, idx2])
scale_factor = 5.0 / ref_3D_dist  # white line is 5 cm

# --- Apply scaling to all depths ---
# depths_raw = pts3D[2]
# depths_raw = depths_raw[np.isfinite(depths_raw)]
# depths_cm = depths_raw * scale_factor
# depths_cm = depths_cm[(depths_cm > 0) & (depths_cm < np.percentile(depths_cm, 99))]

# Apply scale factor to depths
depths_raw = pts3D[2]
depths_raw = depths_raw[np.isfinite(depths_raw)]

# Keep only positive depths
depths_positive = depths_raw[depths_raw > 0]

# Remove extreme outliers by clipping at 95th percentile
max_depth_cutoff = np.percentile(depths_positive, 95)
filtered_depths = depths_positive[depths_positive <= max_depth_cutoff]

# Apply scale
depths_cm = filtered_depths * scale_factor


# --- Display statistics ---
print("\n--- Estimated Pothole Depth Statistics ---")
print(f"Min Depth   : {np.min(depths_cm):.2f} cm")
print(f"Max Depth   : {np.max(depths_cm):.2f} cm")
print(f"Mean Depth  : {np.mean(depths_cm):.2f} cm")
print(f"Median Depth: {np.median(depths_cm):.2f} cm")

# --- Optional histogram plot ---
plt.hist(depths_cm, bins=50, log=True)
plt.title("Depth Distribution (in cm)")
plt.xlabel("Depth")
plt.ylabel("Point Count")
plt.grid()
plt.show()
