import numpy as np
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D visualization
from matplotlib import pyplot as plt
import glob
import os

"""
================================================================================
TRIANGULATION AND 3D POINT CLOUD RECONSTRUCTION
================================================================================

This section describes the reconstruction of 3D scene points from two calibrated
views using triangulation, followed by visualization of the resulting 3D point
cloud.

Triangulation is the process of estimating the 3D position of a point by
intersecting the back-projected rays originating from two camera centers and
passing through corresponding image points.

------------------------------------------------------------------------------
1. GEOMETRIC PRINCIPLE OF TRIANGULATION
------------------------------------------------------------------------------

Given two cameras with projection matrices:

    P1 = K [I | 0]
    P2 = K [R | t]

and corresponding image points x1 and x2, the 3D point X satisfies:

    x1 ~ P1 X
    x2 ~ P2 X

Each image observation defines a ray in 3D space. The true 3D point lies at the
intersection of these rays. Due to noise, the rays generally do not intersect
exactly, and triangulation computes the point that best satisfies both
projection constraints.

------------------------------------------------------------------------------
2. LINEAR TRIANGULATION FORMULATION
------------------------------------------------------------------------------

The projection equations can be rewritten using the cross-product constraint:

    x × (P X) = 0

For each view, this yields two independent linear equations. Stacking equations
from both views leads to a homogeneous linear system:

    A X = 0

where:
    - A ∈ ℝ^(4×4)
    - X is the homogeneous 3D point

The solution is obtained via Singular Value Decomposition (SVD) as the right
singular vector corresponding to the smallest singular value.

------------------------------------------------------------------------------
3. HOMOGENEOUS COORDINATES
------------------------------------------------------------------------------

Triangulation returns points in homogeneous coordinates:

    X_h = (x, y, z, w)ᵀ

Conversion to Euclidean coordinates is performed by normalization:

    X = (x / w, y / w, z / w)

This step removes the projective scale ambiguity inherent in homogeneous
representations.

------------------------------------------------------------------------------
4. ROLE OF PROJECTION MATRICES
------------------------------------------------------------------------------

Accurate triangulation requires:
    - Correct intrinsic calibration (K)
    - Correct relative pose (R, t)
    - Properly disambiguated projection matrices (cheirality enforced)

Errors in any of these components directly affect the quality of the 3D
reconstruction.

------------------------------------------------------------------------------
5. SCALE AMBIGUITY AND SCENE UNITS
------------------------------------------------------------------------------

The reconstructed 3D points are defined up to an unknown global scale due to the
scale ambiguity of the translation vector t recovered from the Essential Matrix.

As a result:
    - Absolute distances are not metric
    - Relative structure and shape are preserved
    - Scale can only be recovered with additional information

------------------------------------------------------------------------------
6. POINT CLOUD PROPERTIES
------------------------------------------------------------------------------

The output of triangulation is a sparse 3D point cloud where:
    - Each point corresponds to a matched feature
    - Point density depends on feature detection and matching quality
    - Noise increases with small baselines and poor triangulation angles

Points may be further filtered based on:
    - Reprojection error
    - Depth consistency
    - Visibility constraints

------------------------------------------------------------------------------
7. VISUALIZATION OF THE 3D POINT CLOUD
------------------------------------------------------------------------------

The reconstructed 3D points are visualized using Matplotlib's 3D plotting tools.

Important considerations:
    - Matplotlib does not enforce equal axis scaling by default
    - Manual axis scaling is used to preserve geometric proportions
    - The viewing angle is adjusted for intuitive interpretation

Visualization is qualitative and intended for inspection and debugging rather
than precise measurement.

------------------------------------------------------------------------------
8. PRACTICAL NOTES
------------------------------------------------------------------------------

- Triangulation accuracy improves with larger baselines
- Points close to epipoles are poorly triangulated
- Outliers in correspondences result in incorrect 3D points
- Bundle Adjustment is typically applied after triangulation to refine:
      * Camera poses
      * 3D point positions

------------------------------------------------------------------------------
REFERENCES
------------------------------------------------------------------------------

- Hartley, R., Zisserman, A. "Multiple View Geometry in Computer Vision"
- Szeliski, R. "Computer Vision: Algorithms and Applications"
- OpenCV Documentation (triangulatePoints)

================================================================================
"""


def triangulate_3d_points(P1, P2, pts1, pts2):
    """
    Triangulates 2D points from two views into 3D space.
    """

    # Transpose points
    # cv.triangulatePoints expects data in shape (2, N),
    # but our points are currently (N, 2).
    pts1_t = pts1.T
    pts2_t = pts2.T

    # Triangulate
    # This returns a 4xN matrix of homogeneous coordinates (x, y, z, w)
    points_4d = cv.triangulatePoints(P1, P2, pts1_t, pts2_t)

    # Convert Homogeneous (4D) to Euclidean (3D)
    # We must divide x, y, and z by w.
    points_3d = points_4d[:3, :] / points_4d[3, :]

    # Transpose back to (N, 4) and take the first 3 columns (x, y, z)
    points_3d = points_3d[:3].T

    return points_3d


def plot_3d_point_cloud(points_3d):
    """
    Visualizes the 3D point cloud using Matplotlib.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Extract X, Y, Z columns
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    ax.scatter(X, Y, Z, c="b", marker=".", s=2)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("3D Reconstruction")

    # Important: Matplotlib 3D scaling is often weird.
    # This helps keep the aspect ratio somewhat correct.
    # (Though for perfect equal aspect, you usually need a custom function)
    max_range = (
        np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    )
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Look from the bottom (elevation -90 degrees)
    ax.view_init(elev=-90, azim=-90)

    plt.show()
