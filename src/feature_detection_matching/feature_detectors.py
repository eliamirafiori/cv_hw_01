import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
================================================================================
FEATURE DETECTION, DESCRIPTION, AND MATCHING
================================================================================

This section covers the detection of salient image features, the computation of
local descriptors, and the matching of these descriptors across two images.

Feature detection and matching provide the raw point correspondences required
for estimating geometric relationships such as the Fundamental Matrix,
Essential Matrix, camera pose, and 3D reconstruction.

------------------------------------------------------------------------------
1. ROLE OF FEATURE MATCHING IN MULTI-VIEW GEOMETRY
------------------------------------------------------------------------------

Multi-view geometry relies on identifying corresponding image points that are
projections of the same 3D scene points.

Given a set of matched 2D points:
    (x1ᵢ, x2ᵢ)

we can estimate:
    - Fundamental Matrix (uncalibrated)
    - Essential Matrix (calibrated)
    - Camera pose (R, t)
    - 3D structure via triangulation

The accuracy of the entire pipeline depends heavily on the quality of these
matches.

------------------------------------------------------------------------------
2. FEATURE DETECTION
------------------------------------------------------------------------------

Feature detection identifies image locations that are:
    - Repeatable across viewpoints
    - Distinctive in appearance
    - Robust to noise, scale, and rotation

Typical features include corners, blobs, and textured regions where image
gradients change significantly.

In this implementation, ORB (Oriented FAST and Rotated BRIEF) is used:

    - FAST: Corner detector for keypoint localization
    - BRIEF: Binary descriptor for local appearance
    - Orientation invariance is added via intensity centroid

ORB is chosen because it is:
    - Computationally efficient
    - Rotation invariant
    - Scale robust
    - Suitable for real-time applications

------------------------------------------------------------------------------
3. FEATURE DESCRIPTORS
------------------------------------------------------------------------------

For each detected keypoint, a descriptor is computed to encode the local image
appearance.

ORB descriptors:
    - Are binary vectors
    - Compare image patch intensities
    - Enable fast matching via Hamming distance

Descriptors transform local image patches into a numerical representation that
can be compared across images.

------------------------------------------------------------------------------
4. PREPROCESSING: IMAGE UNDISTORTION
------------------------------------------------------------------------------

If camera intrinsics K and distortion coefficients are known, images can be
undistorted prior to feature detection.

This ensures that:
    - Features lie on ideal pinhole camera projections
    - Epipolar geometry is consistent
    - Matching accuracy improves near image boundaries

Undistortion is optional but recommended for calibrated pipelines.

------------------------------------------------------------------------------
5. FEATURE MATCHING
------------------------------------------------------------------------------

Feature matching associates descriptors from image 1 with descriptors from
image 2 based on similarity.

Two matching strategies are used:

------------------------------------------------------------------------------
5.1 BRUTE-FORCE MATCHING (HAMMING DISTANCE)
------------------------------------------------------------------------------

Brute-force matching computes the distance between every descriptor pair and
selects the closest match.

For binary descriptors, the Hamming distance is used:
    - Counts differing bits between descriptors
    - Efficient via XOR operations

Cross-checking enforces mutual consistency:
    - Feature A in image 1 must match feature B in image 2
    - Feature B in image 2 must also match feature A in image 1

This significantly reduces false matches.

------------------------------------------------------------------------------
5.2 FLANN-BASED MATCHING (LSH)
------------------------------------------------------------------------------

FLANN (Fast Library for Approximate Nearest Neighbors) accelerates matching for
large descriptor sets.

For binary descriptors, Locality Sensitive Hashing (LSH) is used instead of
KD-Trees.

Matching procedure:
    - K-Nearest Neighbors (k=2) search
    - Lowe's Ratio Test:
          d₁ < α · d₂
      where α ≈ 0.7

This removes ambiguous matches caused by repetitive textures.

------------------------------------------------------------------------------
6. GEOMETRIC VERIFICATION
------------------------------------------------------------------------------

Descriptor similarity alone is insufficient; geometrically inconsistent matches
often survive appearance-based filtering.

Geometric verification enforces physical plausibility:
    - Homography estimation via RANSAC
    - Outlier rejection based on reprojection error

Only matches consistent with a dominant geometric model are retained.

In later stages, epipolar geometry (F or E) provides a stronger geometric
constraint.

------------------------------------------------------------------------------
7. ROBUSTNESS AND OUTLIER REJECTION
------------------------------------------------------------------------------

Common sources of outliers include:
    - Repeated textures
    - Occlusions
    - Motion blur
    - Illumination changes

To address this:
    - Ratio tests reject ambiguous matches
    - Cross-checking enforces mutual consistency
    - RANSAC removes geometrically inconsistent matches

These steps are critical to ensure stable downstream estimation.

------------------------------------------------------------------------------
8. VISUALIZATION AND DEBUGGING
------------------------------------------------------------------------------

Match visualization is used to:
    - Inspect matching quality
    - Detect systematic errors
    - Tune detector and matcher parameters

Only a subset of matches is displayed to avoid visual clutter.

------------------------------------------------------------------------------
9. CONNECTION TO THE FULL PIPELINE
------------------------------------------------------------------------------

Feature matching is the ENTRY POINT of the reconstruction pipeline:

    Images → Keypoints → Matches
           → Fundamental Matrix
           → Essential Matrix
           → Pose (R, t)
           → Projection Matrices
           → Triangulation
           → 3D Point Cloud

Errors at this stage propagate to all subsequent steps.

------------------------------------------------------------------------------
REFERENCES
------------------------------------------------------------------------------

- Lowe, D. "Distinctive Image Features from Scale-Invariant Keypoints"
- Rublee et al. "ORB: An efficient alternative to SIFT or SURF"
- Hartley, R., Zisserman, A. "Multiple View Geometry in Computer Vision"
- OpenCV Documentation (Feature Detection and Matching)

================================================================================
"""


def feature_detection(
    img1_path: str,
    img2_path: str,
    detector=None,
    K=None,
    dist=None,
    debug: bool = False,
):
    """
    Detects features in two images using ORB and routes them to a matcher.
    """

    # Health check
    assert os.path.exists(img1_path), f"Left image not found: {img1_path}"
    assert os.path.exists(img2_path), f"Right image not found: {img2_path}"

    # Loads in GRAYSCALE because feature detection relies on intensity changes (gradients).
    # Color information is usually unnecessary for this and adds computational cost (3 channels vs 1).
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)  # Query image (left)
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)  # Train image (right)

    if K is not None and dist is not None:
        if debug:
            print("Undistorting images before detection...")

        # Refine camera matrix to avoid losing pixels at the edges
        # h, w = img1.shape[:2]
        # new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

        # Undistort
        img1 = cv.undistort(img1, K, dist, None, K)
        img2 = cv.undistort(img2, K, dist, None, K)

    # Initialize Detector
    # 'nfeatures=5000' is the maximum number of keypoints to retain.
    # The default is often 500, but 5000 is better for high-res images or detailed scenes
    # to ensure enough matches are found later.
    if detector is None:
        detector = cv.ORB_create(nfeatures=10000)

    # Detection & Description
    # detectAndCompute performs two steps:
    #   1. Detect: Finds 'Keypoints' (points of interest like corners/edges)
    #   2. Compute: Calculates 'Descriptors' (binary vectors that describe the area around the keypoint)
    # The 'mask=None' argument means we look for features in the entire image.
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # akaze = cv.AKAZE_create(threshold=0.0005)
    #
    # kp1, des1 = akaze.detectAndCompute(img1, None)
    # kp2, des2 = akaze.detectAndCompute(img2, None)

    if debug:
        print(f"Detected {len(kp1)} keypoints in left image.")
        print(f"Detected {len(kp2)} keypoints in right image.")

    # Safety Check
    # If an image has no texture (e.g., a blank wall), descriptors might be None.
    # Proceeding without this check would cause the matchers to crash.
    if des1 is None or des2 is None:
        raise RuntimeError(
            "Descriptors are None. Try different images or a different feature extractor."
        )

    return img1, kp1, des1, img2, kp2, des2

