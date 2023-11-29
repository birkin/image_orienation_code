"""
Credit: <https://chat.openai.com/share/eb7fd810-c9a3-481e-8f68-a54ff1db6719>
"""

import cv2
import numpy as np
import argparse


def compare_image_orientation(path_a, path_b):
    # Load images
    img1 = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector
    orb = cv2.ORB_create()  # type: ignore

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare the orientation of two images.')
    parser.add_argument('--path_a', type=str, help='Path to the first image')
    parser.add_argument('--path_b', type=str, help='Path to the second image')

    args = parser.parse_args()

    result = compare_image_orientation(args.path_a, args.path_b)
    print(result)
