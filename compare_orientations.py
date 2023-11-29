import cv2
import numpy as np

def compare_image_orientation(path_a, path_b):
    # Load images
    img1 = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create BFMatcher and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Check if there are enough matches to consider
    if len(matches) > 10:
        # Analyze the matches
        # For simplicity, we can check the average of the difference in angle of the matches
        angle_diff = np.mean([kp1[m.queryIdx].angle - kp2[m.trainIdx].angle for m in matches])
        if abs(angle_diff) < 10:  # Threshold for deciding if orientation is similar
            return 'same-orientation'
        else:
            return 'different-orientation'
    else:
        return 'Not enough matches to determine'

# Example usage
# result = compare_image_orientation('path_to_image_A.jpg', 'path_to_image_B.jpg')
# print(result)
