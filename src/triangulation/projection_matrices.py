import numpy as np
import cv2 as cv

"""
================================================================================
CAMERA PROJECTION MATRICES AND POSE RECOVERY
================================================================================

This section describes the construction of camera projection matrices P1 and P2
from the Essential Matrix E and the intrinsic calibration matrix K.

Projection matrices define how 3D world points are mapped onto 2D image points
and are a fundamental component of 3D reconstruction, triangulation, and
Structure-from-Motion pipelines.

------------------------------------------------------------------------------
1. PINHOLE CAMERA MODEL
------------------------------------------------------------------------------

In homogeneous coordinates, the projection of a 3D point X ∈ ℝ³ onto an image
point x ∈ ℝ² is given by:

    x ~ P X

where:
    - X = (X, Y, Z, 1)ᵀ is a homogeneous 3D point
    - x = (u, v, 1)ᵀ is a homogeneous image point
    - P ∈ ℝ^(3×4) is the camera projection matrix
    - "~" denotes equality up to scale

------------------------------------------------------------------------------
2. STRUCTURE OF THE PROJECTION MATRIX
------------------------------------------------------------------------------

The projection matrix can be decomposed as:

    P = K [R | t]

where:
    - K ∈ ℝ^(3×3) is the intrinsic calibration matrix
    - R ∈ SO(3) is the rotation from world to camera coordinates
    - t ∈ ℝ³ is the translation from world to camera coordinates

K encodes intrinsic parameters such as focal length and principal point,
while [R | t] encodes the camera pose (extrinsic parameters).

------------------------------------------------------------------------------
3. REFERENCE CAMERA (P1)
------------------------------------------------------------------------------

Without loss of generality, the first camera is chosen as the reference frame:

    R1 = I
    t1 = 0

Thus, its projection matrix is:

    P1 = K [I | 0]

This choice fixes the world coordinate system at the optical center of the first
camera and removes gauge freedom in the reconstruction.

------------------------------------------------------------------------------
4. SECOND CAMERA (P2)
------------------------------------------------------------------------------

The second camera is related to the first by a relative pose (R, t) recovered
from the Essential Matrix:

    E = [t]× R

Once the correct (R, t) pair is selected (via cheirality), the second projection
matrix is constructed as:

    P2 = K [R | t]

Note:
    - The translation vector t is recovered only up to scale
    - This scale ambiguity does not affect triangulation consistency

------------------------------------------------------------------------------
5. POSE RECOVERY FROM THE ESSENTIAL MATRIX
------------------------------------------------------------------------------

Decomposing the Essential Matrix yields four possible (R, t) solutions.
Only one corresponds to a physically valid configuration.

OpenCV's recoverPose function:
    - Decomposes E via SVD
    - Tests all four pose hypotheses
    - Triangulates 3D points
    - Applies the cheirality constraint (positive depth)
    - Returns the (R, t) that places points in front of both cameras

Thus, recoverPose performs pose disambiguation implicitly.

------------------------------------------------------------------------------
6. ROLE OF POINT CORRESPONDENCES
------------------------------------------------------------------------------

The point correspondences (pts1, pts2) are used to:
    - Validate the correct pose among the four candidates
    - Reject points that violate the cheirality constraint
    - Return a mask of points consistent with the recovered pose

Only these inlier points should be used for subsequent triangulation and
3D reconstruction.

------------------------------------------------------------------------------
7. GEOMETRIC INTERPRETATION
------------------------------------------------------------------------------

Given P1 and P2:
    - Each image point defines a ray in 3D space
    - Corresponding rays from two views intersect at the 3D point
    - This intersection is computed via triangulation

Projection matrices thus link:
    2D image geometry ↔ 3D scene structure

------------------------------------------------------------------------------
8. PRACTICAL NOTES
------------------------------------------------------------------------------

- Projection matrices are defined up to a global scale
- Camera centers can be recovered as:
      C = -Rᵀ t
- The baseline between cameras is proportional to ||t||
- Accurate P matrices are critical for:
      * Triangulation
      * Bundle Adjustment
      * Dense reconstruction

------------------------------------------------------------------------------
REFERENCES
------------------------------------------------------------------------------

- Hartley, R., Zisserman, A. "Multiple View Geometry in Computer Vision"
- Faugeras, O. "Three-Dimensional Computer Vision"
- OpenCV Documentation (recoverPose)

================================================================================
"""


def estimate_projection_matrices(K, R, t):
    """
    Computes P1 and P2 from the Essential Matrix using recoverPose.
    """
    # Construct P1 (Origin)
    # P1 = K @ [I | 0]
    # Identity matrix (3x3) concatenated with a zero column (3x1)
    I = np.eye(3)
    zeros = np.zeros((3, 1))
    P1 = np.dot(K, np.hstack((I, zeros)))

    # Construct P2 (New View)
    # P2 = K @ [R | t]
    P2 = np.dot(K, np.hstack((R, t)))

    # Return P1, P2 AND the points that actually define this pose
    return P1, P2


def find_camera_matrices(E, K, pts1, pts2, debug: bool = False):
    """
    Computes P1 and P2 from the Essential Matrix using recoverPose.
    """

    # Recover R and t
    # This extracts the rotation and translation.
    # It uses the points (pts1, pts2) to check which of the 4 solutions is valid.
    # pts1 and pts2 should be the INLIERS from your previous steps.
    points, R, t, mask = cv.recoverPose(E, pts1, pts2, K)

    # Filter points again using the pose mask
    mask_pose = mask.ravel().astype(bool)
    pts1_valid = pts1[mask_pose]
    pts2_valid = pts2[mask_pose]

    if debug:
        print(f"Recovered Pose with {points} valid points.")
        print("\nRotation matrix R:\n", R)
        print("\nTranslation vector t:\n", t)
        print(f"RecoverPose kept {points} points (Cheirality check)")

    # Construct P1 (Origin)
    # P1 = K @ [I | 0]
    # Identity matrix (3x3) concatenated with a zero column (3x1)
    I = np.eye(3)
    zeros = np.zeros((3, 1))
    P1 = np.dot(K, np.hstack((I, zeros)))

    # Construct P2 (New View)
    # P2 = K @ [R | t]
    P2 = np.dot(K, np.hstack((R, t)))

    if debug:
        print('\nProjection Matrix 1 "Camera 1":\n', P1)
        print('\nProjection Matrix 2 "Camera 2":\n', P2)

    # Return P1, P2 AND the points that actually define this pose
    return P1, P2, pts1_valid, pts2_valid
