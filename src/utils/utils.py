import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


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


def visualize_matcher(img1, kp1, img2, kp2, matches):

    # Visualization
    matches_to_show = min(100, len(matches))

    # drawMatches creates a new image containing both img1 and img2 side-by-side
    # and draws lines connecting the matched keypoints.
    img_matches = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:matches_to_show],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,  # Only draw matched points, not all detected ones
    )

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.title("Feature matches (subset)")
    plt.imshow(img_matches)
    plt.axis("off")
    plt.show()


def compute_reprojection_error(P1, P2, points_3d, pts1, pts2):
    """
    Computes the average re-projection error (RMSE) in pixels.
    """
    # Convert 3D points to homogeneous coordinates (4xN)
    # Shape becomes (4, N) -> [[x, y, z, 1], ...]
    points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T

    # Project into Image 1
    # P1 is 3x4, points_3d_hom is 4xN -> projected_1 is 3xN
    projected_1_hom = np.dot(P1, points_3d_hom)

    # Normalize by the last row (Z/W) to get pixel coordinates (u, v)
    projected_1 = projected_1_hom[:2] / projected_1_hom[2]

    # Calculate Euclidean distance between observed and projected points
    # pts1 is (N, 2), projected_1.T is (N, 2)
    error_1 = np.linalg.norm(pts1 - projected_1.T, axis=1)

    # Project into Image 2
    projected_2_hom = np.dot(P2, points_3d_hom)
    projected_2 = projected_2_hom[:2] / projected_2_hom[2]
    error_2 = np.linalg.norm(pts2 - projected_2.T, axis=1)

    # Compute Mean Error
    mean_error_1 = np.mean(error_1)
    mean_error_2 = np.mean(error_2)
    total_mean_error = (mean_error_1 + mean_error_2) / 2.0

    return mean_error_1, mean_error_2, total_mean_error
