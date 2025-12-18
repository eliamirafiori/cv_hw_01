import numpy as np
import cv2 as cv

"""
================================================================================
ESSENTIAL MATRIX ESTIMATION AND POSE DISAMBIGUATION (NUMPY IMPLEMENTATION)
================================================================================

This file implements the computation of the Essential Matrix E and the recovery
of the relative camera pose (R, t) between two calibrated views using ONLY NumPy,
without relying on OpenCV's high-level functions.

------------------------------------------------------------------------------
1. EPIPOLAR GEOMETRY BACKGROUND
------------------------------------------------------------------------------

Given two calibrated pinhole cameras observing the same 3D point X, the image
projections x1 and x2 satisfy the epipolar constraint:

    x̂2ᵀ E x̂1 = 0

where:
    - x̂1, x̂2 are normalized image points (camera coordinates)
    - E is the Essential Matrix

Normalized points are obtained via:

    x̂ = K⁻¹ x

where K is the camera intrinsic matrix.

------------------------------------------------------------------------------
2. DEFINITION OF THE ESSENTIAL MATRIX
------------------------------------------------------------------------------

The Essential Matrix encodes the relative pose between two cameras:

    E = [t]× R

where:
    - R ∈ SO(3) is the relative rotation
    - t ∈ ℝ³ is the relative translation (defined up to scale)
    - [t]× is the skew-symmetric matrix of t

Properties of a valid Essential Matrix:
    - rank(E) = 2
    - two equal non-zero singular values
    - E is defined up to a non-zero scale factor

------------------------------------------------------------------------------
3. ESTIMATION VIA THE EIGHT-POINT ALGORITHM
------------------------------------------------------------------------------

For each correspondence (x̂1, x̂2), the epipolar constraint expands into a linear
equation in the entries of E:

    x̂2ᵀ E x̂1 = 0

Stacking N ≥ 8 correspondences yields a homogeneous system:

    A e = 0

where:
    - A ∈ ℝ^(N×9)
    - e is the vectorized form of E

The solution is obtained by Singular Value Decomposition (SVD) as the right
singular vector corresponding to the smallest singular value.

------------------------------------------------------------------------------
4. ESSENTIAL MATRIX CONSTRAINT ENFORCEMENT
------------------------------------------------------------------------------

Due to noise, the linear solution does not exactly satisfy the theoretical
constraints of an Essential Matrix. Therefore, E is projected onto the essential
manifold by enforcing:

    Σ = diag(1, 1, 0)

via SVD:

    E = U Σ Vᵀ

This ensures rank(E) = 2 and equal non-zero singular values.

------------------------------------------------------------------------------
5. DECOMPOSITION OF E INTO (R, t)
------------------------------------------------------------------------------

Decomposing E yields FOUR possible (R, t) solutions:

    (R1, +t), (R1, -t), (R2, +t), (R2, -t)

This ambiguity arises from:
    - sign ambiguity of t
    - two valid rotations derived from SVD

All four solutions satisfy the epipolar constraint but are not all physically
valid.

------------------------------------------------------------------------------
6. CHEIRALITY CONSTRAINT (PHYSICAL DISAMBIGUATION)
------------------------------------------------------------------------------

The correct camera pose is selected using the cheirality constraint:

    A 3D point must lie in front of BOTH cameras

Procedure:
    1. Assume camera 1: P1 = [I | 0]
    2. For each candidate (R, t), define camera 2: P2 = [R | t]
    3. Triangulate 3D points from point correspondences
    4. Count points with positive depth in both camera frames
    5. Select the (R, t) yielding the maximum number of valid points

Only ONE of the four solutions satisfies this constraint.

------------------------------------------------------------------------------
7. IMPORTANT NOTES
------------------------------------------------------------------------------

- The Essential Matrix is defined up to scale:
      E ≡ αE , α ≠ 0
- E and -E represent the same epipolar geometry
- Translation is recovered up to an unknown scale
- This pipeline is the foundation of:
      * Visual Odometry
      * Structure-from-Motion (SfM)
      * SLAM

------------------------------------------------------------------------------
REFERENCES
------------------------------------------------------------------------------

- Hartley, R., Zisserman, A. "Multiple View Geometry in Computer Vision"
- Longuet-Higgins, H. "A computer algorithm for reconstructing a scene"
- OpenCV documentation (for comparison only)

================================================================================
"""


def estimate_essential_matrix(
    pts1,
    pts2,
    K,
    prob=0.999,
    threshold=1.0,
    debug=False,
):
    """
    Estimates the Essential Matrix E using RANSAC and the Camera Matrix K.
    """

    # Essential Matrix Estimation
    # method=cv.RANSAC: Standard RANSAC
    # prob=0.999: Confidence level (higher is better for E)
    # threshold=1.0: Max distance from epipolar line (in pixels)
    E, mask = cv.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv.RANSAC,
        prob=prob,
        threshold=threshold,
    )

    if E is None:
        raise RuntimeError("Essential matrix estimation failed")

    # Flatten the mask
    # "mask" is a list of 0s (outliers) and 1s (inliers).
    # We flatten it to a 1D array of booleans.
    mask = mask.ravel().astype(bool)

    # Select ONLY the points that agree with the Fundamental Matrix geometry.
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    return E, pts1_in, pts2_in, mask


def enforce_essential_constraints(E):
    """
    Enforce Essential Matrix constraints (mandatory)

    A valid Essential Matrix must have:
        - rank = 2
        - singular values = (σ, σ, 0)
    """
    U, S, Vt = np.linalg.svd(E)
    sigma = (S[0] + S[1]) / 2.0
    E_corrected = U @ np.diag([sigma, sigma, 0]) @ Vt
    return E_corrected


def find_essential_matrix(
    pts1, pts2, K, prob: float = 0.999, threshold: float = 1.0, debug: bool = False
):
    """
    Estimates the Essential Matrix E using RANSAC and the Camera Matrix K.
    """

    # Essential Matrix Estimation
    # method=cv.RANSAC: Standard RANSAC
    # prob=0.999: Confidence level (higher is better for E)
    # threshold=1.0: Max distance from epipolar line (in pixels)
    E, mask = cv.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv.RANSAC,
        prob=prob,
        threshold=threshold,
    )

    if E is None:
        raise RuntimeError("Essential Matrix estimation failed.")

    # Flatten the mask
    mask = mask.ravel().astype(bool)

    # Filter the points immediately
    pts1_inliers = pts1[mask]
    pts2_inliers = pts2[mask]

    if debug:
        print(f"\nEstimated Essential Matrix E:\n{E}")
        print(f"Essential Matrix kept {len(pts1_inliers)} / {len(pts1)} points")

    # We return the mask because findEssentialMat might filter even more points
    # than your previous Fundamental Matrix step.
    return E, pts1_inliers, pts2_inliers


def compute_essential_matrix(pts1, pts2, K):
    """
    Full Eight-Point Algorithm for Essential Matrix estimation.

    Steps:
    1. Normalize points using K
    2. Estimate E via eight-point algorithm
    3. Enforce essential constraints
    4. Decompose E into 4 (R, t) candidates
    5. Select the correct (R, t) using cheirality
    6. Rebuild E = [t]× R
    """

    # Normalize points
    pts1_norm = normalize_points(pts1, K)
    pts2_norm = normalize_points(pts2, K)

    # Estimate Essential Matrix
    E_est = estimate_E(pts1_norm, pts2_norm)
    E_est = enforce_essential_constraints(E_est)

    # Select correct pose
    R, t = select_correct_pose(E_est, pts1_norm, pts2_norm)

    # Rebuild the correct Essential Matrix
    E_correct = skew(t) @ R

    return E_correct, R, t


def normalize_points(pts, K):
    """
    Normalize image points using the intrinsic matrix.

    x̂ = K⁻¹ x
    where x is in homogeneous pixel coordinates.
    """
    K_inv = np.linalg.inv(K)

    # Convert to homogeneous coordinates: (x, y) → (x, y, 1)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])

    # Apply inverse intrinsics
    pts_norm = (K_inv @ pts_h.T).T

    return pts_norm[:, :2]


def estimate_E(pts1, pts2):
    """
    Solve A e = 0 using SVD.
    """
    A = build_matrix_A(pts1, pts2)

    # A = U Σ Vᵀ → solution is last column of V
    _, _, Vt = np.linalg.svd(A)

    E = Vt[-1].reshape(3, 3)
    return E


def build_matrix_A(pts1, pts2):
    """
    Build the linear system A e = 0 from point correspondences.

    Each row encodes:
    x̂₂ᵀ E x̂₁ = 0
    """
    A = np.zeros((pts1.shape[0], 9))

    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
    return A


def enforce_essential_constraints(E):
    """
    Enforce rank-2 constraint and equal singular values.

    Math:
    E = U diag(1, 1, 0) Vᵀ
    """
    U, _, Vt = np.linalg.svd(E)

    S = np.diag([1, 1, 0])
    E_corrected = U @ S @ Vt

    return E_corrected


def select_correct_pose(E, pts1, pts2):
    candidates = decompose_E(E)

    best_count = -1
    best_pose = None

    for R, t in candidates:
        count = count_cheirality_inliers(R, t, pts1, pts2)
        if count > best_count:
            best_count = count
            best_pose = (R, t)

    return best_pose


def decompose_E(E):
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotation matrices
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def count_cheirality_inliers(R, t, pts1, pts2):
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])

    count = 0

    for x1, x2 in zip(pts1, pts2):
        x1_h = np.array([x1[0], x1[1], 1.0])
        x2_h = np.array([x2[0], x2[1], 1.0])

        X = triangulate_point(P1, P2, x1_h, x2_h)

        # Depth in camera 1
        Z1 = X[2]

        # Depth in camera 2
        X2 = R @ X[:3] + t
        Z2 = X2[2]

        if Z1 > 0 and Z2 > 0:
            count += 1

    return count


def triangulate_point(P1, P2, x1, x2):
    A = np.zeros((4, 4))

    A[0] = x1[0] * P1[2] - P1[0]
    A[1] = x1[1] * P1[2] - P1[1]
    A[2] = x2[0] * P2[2] - P2[0]
    A[3] = x2[1] * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X / X[3]


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
