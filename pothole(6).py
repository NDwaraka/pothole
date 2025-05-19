import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the video
cap = cv2.VideoCapture('videos/movpot(1).mp4')  # Replace with your actual path

# Get two frames for Structure-from-Motion
ret, frame1 = cap.read()
for _ in range(5):  # Skip a few frames to simulate movement
    ret, frame2 = cap.read()

# Convert to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Detect feature points using ORB
orb = cv2.ORB_create(2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Match features using brute force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)   
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Camera intrinsic matrix (calibration)
K = np.array([[700, 0, frame1.shape[1]/2],
              [0, 700, frame1.shape[0]/2],
              [0,   0,               1]])

# Essential matrix and pose recovery
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

# Undistort and normalize coordinates
pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)

# Projection matrices for triangulation
proj1 = np.hstack((np.eye(3), np.zeros((3,1))))
proj2 = np.hstack((R, t))

# Triangulate 3D points
pts4D = cv2.triangulatePoints(K @ proj1, K @ proj2, pts1.T, pts2.T)
pts3D = pts4D[:3, :] / pts4D[3, :]  # Convert from homogeneous to 3D

# Filter invalid or extreme depth values
depths = pts3D[2]
depths = depths[np.isfinite(depths)]
depths = depths[(depths > 0) & (depths < np.percentile(depths, 99))]

# === Histogram ===
plt.hist(depths, bins=50, log=True)
plt.title("Depth Distribution")
plt.xlabel("Depth (relative units)")
plt.ylabel("Point Count")
plt.grid()
plt.show()

# === 3D Point Cloud Visualization ===
sample = np.random.choice(pts3D.shape[1], size=min(1000, pts3D.shape[1]), replace=False)
pts3D_sampled = pts3D[:, sample]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3D_sampled[0], pts3D_sampled[1], pts3D_sampled[2], c=pts3D_sampled[2], cmap='plasma', s=2)
ax.set_title("3D Point Cloud")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Depth (Z)")
plt.show()

# === Real-World Severity Estimation ===
focal_length_px = 700
baseline_m = 0.3  # Estimated camera shift in meters
scaling_factor = baseline_m / focal_length_px  # meters per depth unit

real_depths = depths * scaling_factor
severe_threshold = 0.1  # meters (e.g., dips deeper than 10 cm)

severity_score = np.sum(real_depths > severe_threshold) / len(real_depths) * 100
print(f"Pothole Severity Score (real-world logic): {severity_score:.2f}% points deeper than {severe_threshold*100:.1f} cm")
