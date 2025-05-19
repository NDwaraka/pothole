import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
cap = cv2.VideoCapture('videos/movpot(3).mp4')  # Use forward slashes or raw string

# Set fixed frame positions
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret1, frame1 = cap.read()

cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
ret2, frame2 = cap.read()

if not ret1 or not ret2:
    print("Failed to read frames from the video.")
    exit()

# Convert to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Detect ORB features
orb = cv2.ORB_create(2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Filter matches
if len(matches) < 20:
    print("Too few good matches. Skipping analysis.")
    exit()

# Extract matched points
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Camera intrinsic matrix (example, tune to your real camera if needed)
K = np.array([[700, 0, frame1.shape[1] / 2],
              [0, 700, frame1.shape[0] / 2],
              [0,   0,                1]])

# Compute essential matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

# Triangulate
pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)

proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
proj2 = np.hstack((R, t))
pts4D = cv2.triangulatePoints(K @ proj1, K @ proj2, pts1.T, pts2.T)

# Convert from homogeneous to 3D
pts3D = pts4D[:3, :] / pts4D[3, :]
depths = pts3D[2]  # Z coordinate = depth

# Clean depths
depths = depths[np.isfinite(depths)]
depths = depths[(depths > 0) & (depths < np.percentile(depths, 99))]

if len(depths) == 0:
    print("No valid depth points detected.")
    exit()

# Plot histogram
plt.hist(depths, bins=50, log=True)
plt.title("Depth Distribution")
plt.xlabel("Depth (relative units)")
plt.ylabel("Point Count")
plt.grid()
plt.show()

# Estimate pothole severity
depth_threshold = 10.0  # Example: 10 cm threshold for severity
severe_points = np.sum(depths > depth_threshold)
severity_score = 100 * severe_points / len(depths)

print(f"Pothole Severity Score (real-world logic): {severity_score:.2f}% points deeper than {depth_threshold} cm")
