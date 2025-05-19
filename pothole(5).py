import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
cap = cv2.VideoCapture('videos\movpot(3).mp4')  # Replace with your video file

# Get two frames for SfM
ret, frame1 = cap.read()
for _ in range(5):  # Skip a few frames to simulate movement
    ret, frame2 = cap.read()

# Convert to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Detect feature points in first frame using ORB
orb = cv2.ORB_create(2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched points
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Camera intrinsic matrix (example, adjust to your camera)
K = np.array([[700, 0, frame1.shape[1]/2],
              [0, 700, frame1.shape[0]/2],
              [0,   0,               1]])

# Find essential matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

# Triangulate points
pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)

proj1 = np.hstack((np.eye(3), np.zeros((3,1))))
proj2 = np.hstack((R, t))
pts4D = cv2.triangulatePoints(K @ proj1, K @ proj2, pts1.T, pts2.T)

# Convert from homogeneous to 3D
pts3D = pts4D[:3, :] / pts4D[3, :]

# Depth values
depths = pts3D[2]
depths = depths[np.isfinite(depths)]
depths = depths[(depths > 0) & (depths < np.percentile(depths, 99))]


# Plot histogram of depth
plt.hist(depths, bins=50, log=True)

# plt.hist(depths, bins=50)
plt.title("Depth Distribution")
plt.xlabel("Depth (relative units)")
plt.ylabel("Point Count")
plt.grid()
plt.show()

# Severity rating (mock logic: deeper → more severe)

depth_rating = np.clip(100 - np.percentile(depths, 10), 0, 100)
print(f"Pothole Severity Rating (relative): {depth_rating:.2f}/100")
