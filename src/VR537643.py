import os

import numpy as np
import cv2 as cv

from src.calibration.calibration import calibration, load_calibration
from src.epipolar_geometry.epipolar_lines import draw_epipolar_lines
from src.epipolar_geometry.essential_matrix import (
    compute_essential_matrix,
    enforce_essential_constraints,
    estimate_essential_matrix,
    find_essential_matrix,
)
from src.epipolar_geometry.fundamental_matrix import (
    estimate_fundamental_matrix,
    find_fundamental_matrix,
)
from src.feature_detection_matching.feature_detection import (
    brute_force_based_matcher,
    feature_detection,
    flann_based_matcher,
)
from src.triangulation.projection_matrices import (
    estimate_projection_matrices,
    find_camera_matrices,
)
from src.triangulation.triangulation_cloud_points import (
    plot_3d_point_cloud,
    triangulate_3d_points,
)
from src.utils.utils import get_keypoint_coords_from_matches


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


if __name__ == "__main__":
    # Check if we need to run calibration, or just load it
    camera_path = "phone/vertical"
    calib_path = f"assets/calibration/{camera_path}/calibration.npz"
    debug = True

    K, dist = None, None

    if os.path.exists(calib_path) and False:
        print("Loading existing calibration...")
        K, dist = load_calibration(calib_path)
    else:
        print("Running calibration...")
        K, dist = calibration(
            calibration_assets_path=f"assets/calibration/{camera_path}",
            square_size=0.024,
            debug=debug,
        )

    if debug:
        print(f"Camera Matrix K: {K}")
        print(f"Distortion Coefficients: {dist}")

    ### Feature Extraction and Matching ###

    if debug:
        print("\n\n### Feature Extraction and Matching ###\n")

    # Run feature detection WITH undistortion
    img1, kp1, des1, img2, kp2, des2 = feature_detection(
        img1_path=f"assets/{camera_path}/1.jpg",
        img2_path=f"assets/{camera_path}/2.jpg",
        K=K,
        dist=dist,
        debug=debug,
    )

    matches = []
    use_FLANN_matcher = True

    # Matching Routing
    # Depending on the flag, we route the features to different matching strategies.
    if use_FLANN_matcher:
        # FLANN (Fast Library for Approximate Nearest Neighbors) is faster for large datasets.
        # Note: Since ORB uses binary descriptors, FLANN requires specific tuning (LSH index).
        matches = flann_based_matcher(img1, kp1, des1, img2, kp2, des2, debug)
    else:
        # Brute-Force checks every keypoint in img1 against every keypoint in img2.
        # It is slower but provides the mathematically 'perfect' nearest neighbor.
        matches = brute_force_based_matcher(img1, kp1, des1, img2, kp2, des2, debug)

    ### Epipolar Geometry Estimation ###

    if debug:
        print("\n\n### Epipolar Geometry Estimation ###\n")

    # Extract matched points
    pts1, pts2 = get_keypoint_coords_from_matches(kp1, kp2, matches)

    if debug:
        print(f"Initial Matches: {len(pts1)}")

    F, inliers1, inliers2 = find_fundamental_matrix(
        img1, kp1, pts1, img2, kp2, pts2, matches, debug=debug
    )

    F, pts1_in, pts2_in, mask = estimate_fundamental_matrix(
        pts1, pts2, ransac_thresh=1.0, confidence=0.999
    )

    if debug:
        print(f"F inliers: {len(pts1_in)} / {len(pts1)}")

    E, inliers1, inliers2 = find_essential_matrix(pts1, pts2, K, debug=debug)

    # Essential matrix from Fundamental matrix
    # At this point E is NOT guaranteed to be a valid Essential matrix.
    E = K.T @ F @ K

    E = enforce_essential_constraints(E)

    # Essential matrix directly
    E, pts1_in, pts2_in, mask = estimate_essential_matrix(pts1, pts2, K, debug=True)

    if debug:
        print(f"E inliers: {len(pts1_in)} / {len(pts1)}")

    # Recover pose
    # This extracts the rotation and translation.
    # It uses the points (pts1, pts2) to check which of the 4 solutions is valid.
    # pts1 and pts2 should be the INLIERS from your previous steps.
    num_points, R, t, mask = cv.recoverPose(E, pts1_in, pts2_in, K)

    # Filter points again using the pose mask
    mask_pose = mask.ravel().astype(bool)
    pts1_valid = pts1[mask_pose]
    pts2_valid = pts2[mask_pose]

    if debug:
        print(f"Recovered Pose with {num_points} valid points.")
        print("\nRotation matrix R:\n", R)
        print("\nTranslation vector t:\n", t)
        print(f"RecoverPose kept {num_points} points (Cheirality check)")

    draw_epipolar_lines(img1, pts1, img2, pts2, F)

    ### Triangulation and 3D Reconstruction ###

    if debug:
        print("\n\n### Triangulation and 3D Reconstruction ###\n")

    P1, P2, valid_pts1, valid_pts2 = find_camera_matrices(
        E, K, inliers1, inliers2, debug=debug
    )

    P1, P2 = estimate_projection_matrices(K, R, t)

    if debug:
        print('\nProjection Matrix 1 "Camera 1":\n', P1)
        print('\nProjection Matrix 2 "Camera 2":\n', P2)

    points_3d = triangulate_3d_points(P1, P2, pts1_in, pts2_in)

    if debug:
        print(f"Generated {len(points_3d)} 3D points.")

    # Calculate quantitative error
    mean_error_1, mean_error_2, total_mean_error = compute_reprojection_error(
        P1, P2, points_3d, pts1_in, pts2_in
    )

    if debug:
        print(f"Re-projection Error Camera 1: {mean_error_1:.4f} px")
        print(f"Re-projection Error Camera 2: {mean_error_2:.4f} px")
        print(f"Total Mean Re-projection Error: {total_mean_error:.4f} px")

    plot_3d_point_cloud(points_3d)
