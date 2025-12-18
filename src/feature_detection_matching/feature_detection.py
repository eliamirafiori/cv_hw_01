import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

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

        # (Optional) refine camera matrix to avoid losing pixels at the edges
        # h, w = img1.shape[:2]
        # new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

        # Undistort
        img1 = cv.undistort(img1, K, dist, None, K)
        img2 = cv.undistort(img2, K, dist, None, K)

    # Initialize ORB Detector
    # 'nfeatures=5000' is the maximum number of keypoints to retain.
    # The default is often 500, but 5000 is better for high-res images or detailed scenes
    # to ensure enough matches are found later.
    orb = cv.ORB_create(nfeatures=10000)

    # Detection & Description
    # detectAndCompute performs two steps:
    #   1. Detect: Finds 'Keypoints' (points of interest like corners/edges)
    #   2. Compute: Calculates 'Descriptors' (binary vectors that describe the area around the keypoint)
    # The 'mask=None' argument means we look for features in the entire image.
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

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


def brute_force_based_matcher(
    img1,
    kp1,
    des1,
    img2,
    kp2,
    des2,
    max_matches: int = 5000,
    debug: bool = False,
):
    """
    Matches ORB descriptors using Brute Force Hamming distance with Cross-Check.
    """

    # Initialize Matcher
    # cv.NORM_HAMMING: Essential for binary descriptors (ORB, BRIEF).
    # It calculates distance by counting the number of differing bits (XOR operation).
    # crossCheck=True: This enforces a mutual match.
    # Feature A in img1 must match B in img2, AND Feature B in img2 must match A in img1.
    # This filters out many false positives automatically.
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Perform Matching
    # bf.match() returns a list of DMatch objects.
    # A DMatch object has four main attributes:
    # Match(
    #    queryIdx,  # index of the descriptor in the first set (des1)
    #    trainIdx,  # index of the descriptor in the second set (des2)
    #    imgIdx,    # index of the training image (used with multiple images)
    #    distance   # distance between the two descriptors
    # )
    # Unlike knnMatch, this returns the single best match for each keypoint (because of crossCheck).
    matches = bf.match(des1, des2)

    # Sorting
    # DMatch objects have a 'distance' attribute.
    # Lower distance = more similar descriptors = better match.
    matches = sorted(matches, key=lambda x: x.distance)

    # Selection
    # We slice the list to keep only the top N matches.
    # This removes weak matches that might just be noise or repetitive textures.
    good_matches = matches[: min(max_matches, len(matches))]

    if debug:
        print(
            f"Using {len(good_matches)} best matches for fundamental matrix estimation."
        )

    # Visualization
    matches_to_show = min(100, len(good_matches))

    # drawMatches creates a new image containing both img1 and img2 side-by-side
    # and draws lines connecting the matched keypoints.
    img_matches = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches[:matches_to_show],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,  # Only draw matched points, not all detected ones
    )

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.title("ORB feature matches (subset)")
    plt.imshow(img_matches)
    plt.axis("off")
    plt.show()

    return good_matches


def flann_based_matcher(
    img1,
    kp1,
    des1,
    img2,
    kp2,
    des2,
    min_matches: int = 10,
    debug: bool = False,
):
    """
    Matches ORB descriptors using FLANN (LSH Indexing) with geometric verification (Homography).
    """

    # Configure FLANN for Binary Descriptors
    # Standard FLANN uses KD-Trees, which work for floating-point descriptors (SIFT/SURF).
    # For binary descriptors (ORB), KD-Trees fail. We MUST use LSH (Locality Sensitive Hashing).
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # Number of hash tables (more = more accurate, slower)
        key_size=12,  # Size of the hash key in bits (typically 10-20)
        multi_probe_level=1,  # Number of nearby buckets to search (higher = better recall)
    )

    # checks=50: The number of times the trees/tables are traversed.
    # Higher values give higher precision but slower performance.
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    # KNN Matching
    # We find the top 2 matches (k=2) for every descriptor.
    # We need the 2nd best match to perform Lowe's Ratio Test.
    matches = flann.knnMatch(des1, des2, k=2)

    # Prepare a mask to determine which matches to draw.
    # Initially, all are [0, 0] (draw nothing).
    matchesMask = [[0, 0] for i in range(len(matches))]

    # Lowe's Ratio Test
    # Filter: "Is the best match significantly better than the second best?"
    # If m.distance < 0.7 * n.distance, the match is distinct and likely correct.
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            # If it passes the ratio test, mark it as a candidate for drawing
            matchesMask[i] = [1, 0]

    # Geometric Verification (Homography)
    # Even after the ratio test, outliers may exist.
    # We check if the points physically align via a perspective transformation.
    if len(good_matches) > min_matches:

        # Extract coordinates of the good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # findHomography uses RANSAC to find the transformation matrix M.
        # It rejects outliers that don't fit the dominant geometric model.
        # 5.0 is the reprojection threshold in pixels.
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        # 'mask' here is a list of 1s (inliers) and 0s (outliers) corresponding to 'good_matches'.
        matchesMask_inliers = mask.ravel().tolist()

        # Update matchesMask to only draw RANSAC inliers
        # We iterate through good_matches and update the global matchesMask
        j = 0
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                # If this was a good match, check if RANSAC accepted it
                if matchesMask_inliers[j] == 1:
                    matchesMask[i] = [1, 0]
                else:
                    matchesMask[i] = [0, 0]  # Rejected by RANSAC
                j += 1

        # Project Object Boundary
        # Draw a box around the object in the target image
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )

        if M is not None:
            dst = cv.perspectiveTransform(pts, M)
            # Draw the bounding box on img2
            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    else:
        print(f"Not enough matches are found - {len(good_matches)}/{min_matches}")
        matchesMask = None

    # Visualization
    draw_params = dict(
        matchColor=(0, 255, 0),  # Draw valid matches in Green
        singlePointColor=None,
        matchesMask=matchesMask,  # Draw only inliers
        flags=cv.DrawMatchesFlags_DEFAULT,
    )

    # Use drawMatchesKnn because 'matches' is a list of lists (k=2)
    img_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    plt.figure(figsize=(15, 7))
    plt.title("FLANN feature matches (Homography Checked)")
    plt.imshow(img_matches)
    plt.axis("off")
    plt.show()

    return good_matches
