import numpy as np


def get_keypoint_coords_from_matches(kp1, kp2, matches):
    """
    Helper to extract coordinates from matches.
    """

    # Point Extraction
    # Convert matched keypoints into simple numpy arrays of (x, y) coordinates.
    # pts1: Points in the Left Image (from queryIdx)
    # pts2: Points in the Right Image (from trainIdx)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2
