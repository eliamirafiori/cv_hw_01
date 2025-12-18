import numpy as np
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D visualization
from matplotlib import pyplot as plt
import glob
import os

"""
================================================================================
FUNDAMENTAL MATRIX ESTIMATION AND EPIPOLAR GEOMETRY (PIXEL COORDINATES)
================================================================================

This section concerns the estimation of the Fundamental Matrix F, which encodes
the epipolar geometry between two views when camera intrinsics are UNKNOWN or
NOT explicitly used.

Unlike the Essential Matrix, the Fundamental Matrix operates directly in pixel
coordinates and applies to uncalibrated camera systems.

------------------------------------------------------------------------------
1. EPIPOLAR CONSTRAINT IN PIXEL COORDINATES
------------------------------------------------------------------------------

Given a 3D point X projected into two images as pixel coordinates x1 and x2,
the epipolar constraint is:

    x2ᵀ F x1 = 0

where:
    - x1, x2 are homogeneous pixel coordinates (x, y, 1)
    - F is the Fundamental Matrix

This constraint means that the projection x2 must lie on the epipolar line
associated with x1, and vice versa.

------------------------------------------------------------------------------
2. GEOMETRIC MEANING OF THE FUNDAMENTAL MATRIX
------------------------------------------------------------------------------

The Fundamental Matrix maps:
    - a point in image 1 → an epipolar line in image 2
    - a point in image 2 → an epipolar line in image 1

Specifically:
    l2 = F x1
    l1 = Fᵀ x2

All corresponding points must satisfy the epipolar constraint exactly in the
noise-free case.

------------------------------------------------------------------------------
3. RELATIONSHIP BETWEEN F AND E
------------------------------------------------------------------------------

When the camera intrinsic matrices K1 and K2 are known, the Fundamental Matrix
and Essential Matrix are related by:

    E = K2ᵀ F K1
    F = K2⁻ᵀ E K1⁻¹

Key differences:

+----------------------+----------------------+----------------------+
| Fundamental Matrix   | Essential Matrix     |                      |
+----------------------+----------------------+----------------------+
| Uncalibrated         | Calibrated           |                      |
| Pixel coordinates    | Normalized coords    |                      |
| Includes intrinsics  | Intrinsics removed   |                      |
| rank(F) = 2          | rank(E) = 2          |                      |
| Arbitrary singular   | Two equal singular   |                      |
+----------------------+----------------------+----------------------+

------------------------------------------------------------------------------
4. ESTIMATION VIA THE EIGHT-POINT ALGORITHM
------------------------------------------------------------------------------

For each correspondence (x1, x2), the epipolar constraint expands to a linear
equation in the entries of F:

    x2ᵀ F x1 = 0

Stacking N ≥ 8 correspondences yields a homogeneous linear system:

    A f = 0

where:
    - A ∈ ℝ^(N×9)
    - f is the vectorized form of F

The solution is obtained via Singular Value Decomposition (SVD), selecting the
right singular vector associated with the smallest singular value.

------------------------------------------------------------------------------
5. RANK-2 CONSTRAINT
------------------------------------------------------------------------------

A valid Fundamental Matrix must satisfy:

    rank(F) = 2

This constraint arises from projective geometry and the existence of epipoles.
In practice, the linear solution is corrected by enforcing the rank-2 constraint
through SVD by zeroing the smallest singular value.

------------------------------------------------------------------------------
6. ROBUST ESTIMATION USING RANSAC
------------------------------------------------------------------------------

Real-world feature matching contains outliers due to:
    - incorrect matches
    - repeated textures
    - occlusions
    - noise

To robustly estimate F, Random Sample Consensus (RANSAC) is used:

Procedure:
    1. Randomly sample minimal point sets
    2. Estimate candidate F
    3. Count inliers based on geometric error
    4. Repeat and select the model with most inliers

In OpenCV, the reprojection threshold defines the maximum pixel distance a point
may lie from its epipolar line to be considered an inlier.

------------------------------------------------------------------------------
7. FUNDAMENTAL MATRIX IN PRACTICE
------------------------------------------------------------------------------

- F is defined up to an arbitrary scale factor:
      F ≡ αF , α ≠ 0
- F and -F represent the same epipolar geometry
- F alone does NOT allow metric reconstruction
- F is sufficient for:
      * Outlier rejection
      * Epipolar line visualization
      * Stereo correspondence constraints

Metric reconstruction requires camera calibration and conversion to E.

------------------------------------------------------------------------------
8. CONNECTION TO THE IMPLEMENTATION
------------------------------------------------------------------------------

In this file:
    - Feature correspondences are extracted from keypoint matches
    - cv.findFundamentalMat is used with RANSAC for robustness
    - Inliers consistent with epipolar geometry are selected
    - Visualization highlights geometrically valid matches

This process is typically the FIRST step before:
    - Camera calibration
    - Essential Matrix estimation
    - Pose recovery
    - 3D reconstruction

------------------------------------------------------------------------------
REFERENCES
------------------------------------------------------------------------------

- Hartley, R., Zisserman, A. "Multiple View Geometry in Computer Vision"
- Faugeras, O. "Three-Dimensional Computer Vision"
- OpenCV Documentation (Fundamental Matrix Estimation)

================================================================================
"""


def estimate_fundamental_matrix(
    pts1,
    pts2,
    ransac_thresh=1.0,
    confidence=0.999,
):
    """
    Estimates the Fundamental Matrix F using RANSAC and visualizes inliers.
    """
    # Fundamental Matrix Estimation
    # This calculates the relationship between pixel coordinates in two views.
    # cv.FM_RANSAC: Uses Random Sample Consensus to ignore outliers (bad matches).
    # ransacReprojThreshold: The maximum distance (in pixels) a point can be from the epipolar line to be considered an inlier.
    F, mask = cv.findFundamentalMat(
        pts1,
        pts2,
        cv.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=confidence,
    )

    if F is None:
        raise RuntimeError("Fundamental matrix estimation failed")

    # Filtering Inliers
    # "mask" is a list of 0s (outliers) and 1s (inliers).
    # We flatten it to a 1D array of booleans.
    mask = mask.ravel().astype(bool)

    # Select ONLY the points that agree with the Fundamental Matrix geometry.
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]


    return F, pts1_in, pts2_in, mask


def find_fundamental_matrix(
    img1,
    kp1,
    pts1,
    img2,
    kp2,
    pts2,
    matches,
    ransacReprojThreshold: float = 0.999,
    confidence: float = 1.0,
    debug=False,
):
    """
    Estimates the Fundamental Matrix F using RANSAC and visualizes inliers.
    """

    # Debugging: Ensure we have (N, 2) matrices
    if debug:
        print("pts1 shape:", pts1.shape)
        print("pts2 shape:", pts2.shape)

    # Fundamental Matrix Estimation
    # This calculates the relationship between pixel coordinates in two views.
    # cv.FM_RANSAC: Uses Random Sample Consensus to ignore outliers (bad matches).
    # ransacReprojThreshold: The maximum distance (in pixels) a point can be from the epipolar line to be considered an inlier.
    F, mask = cv.findFundamentalMat(
        pts1,
        pts2,
        method=cv.FM_RANSAC,
        ransacReprojThreshold=ransacReprojThreshold,
        confidence=confidence,
    )

    if F is None:
        raise RuntimeError(
            "Fundamental matrix estimation failed. Try different images or parameters."
        )

    # Filtering Inliers
    # 'mask' is a list of 0s (outliers) and 1s (inliers).
    # We flatten it to a 1D array of booleans.
    inlier_mask = mask.ravel().astype(bool)

    # Select ONLY the points that agree with the Fundamental Matrix geometry.
    inliers1 = pts1[inlier_mask]
    inliers2 = pts2[inlier_mask]

    if debug:
        print("\nEstimated fundamental matrix F:\n", F)
        print(f"Inliers after RANSAC: {inliers1.shape[0]} / {pts1.shape[0]}")

    # Visualization
    # Filter the original DMatch objects to visualize only the "good" geometry matches.
    inlier_matches = [gm for gm, keep in zip(matches, inlier_mask) if keep]

    # Show a subset to avoid drawing a messy "hairball" of lines.
    matches_to_show = min(80, len(inlier_matches))
    img_inlier_matches = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        inlier_matches[:matches_to_show],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(15, 7))
    plt.title("Inlier matches after RANSAC")
    plt.imshow(img_inlier_matches)
    plt.axis("off")
    plt.show()

    return F, inliers1, inliers2
