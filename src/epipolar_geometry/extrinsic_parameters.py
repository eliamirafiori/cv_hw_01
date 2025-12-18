import numpy as np
import cv2 as cv

"""
Decompose E using SVD:

Perform SVD on your Essential Matrix E: E = UΣV^T.
The singular values in Σ should ideally be [σ, σ, 0] (e.g., [20, 20, 0]).
If not perfectly zero due to noise,
you can enforce this by setting the smallest singular value to zero: Σ_ideal = diag(1, 1, 0) (after normalization).


Construct the Z Matrix:

Define a special skew-symmetric matrix Z = [[0, -1, 0], [1, 0, 0], [0, 0, 0]].


Generate Four Candidate Solutions:

From U, V, and Z, you get four potential (R, t) pairs:
    R1: U * Z * V^T, t1: U * Z^T * Σ_diag * V^T (or similar, with sign flip)
    R2: U * Z^T * V^T, t2: U * Z * Σ_diag * V^T (or similar, with sign flip)
    R3: U * Z * V^T, t3: U * Z * Σ_diag * V^T (with opposite signs)
    R4: U * Z^T * V^T, t4: U * Z^T * Σ_diag * V^T (with opposite signs)
    (Note: The exact formula varies slightly by source,
    but the key is using U, V, Z, and the singular values to get four possibilities).

    
Select the Correct Pair (Chirality Test):

For each (R, t) pair, triangulate your 2D points to find their 3D positions (X).
The correct pair is the one where the resulting 3D points X have positive depth (i.e., they are in front of both cameras).
The translation vector t from E is a direction; its magnitude isn't recovered, so you often normalize it to a unit vector. 

NOTES:
Essential Matrix (E): E = [t]_x R,
where [t]_x is the skew-symmetric matrix of the translation vector t, and R is the rotation matrix.
Degrees of Freedom: E has 5 DOFs (3 for rotation, 2 for translation direction).
Normalization: The scale of t is arbitrary; it's often normalized to a unit vector. 
"""

def find_rotation_translation(E: np.ndarray, K: np.ndarray, pts1, pts2):
    """
    Step 1
    E, K, pts1, pts2

    Step 2
    decompose_E(E)

    Step 3
    for each (R, t):
        triangulate points
        count points with positive depth

    Step 4
    select (R, t) with max positive depth

    Step 5
    return R, t
    """
    candidates = decompose_E(E)

    best = None
    max_positive = 0

    for R, t in candidates:
        count = chirality_count(R, t, K, pts1, pts2)
        if count > max_positive:
            best = (R, t)
            max_positive = count

    return best


def decompose_E(E: np.ndarray) -> list:
    """
    Singular Value Decomposition (SVD) is a factorization method in linear algebra that decomposes a matrix into three other matrices, providing a way to represent data in terms of its singular values.

    SVD helps you split that table into three parts:

    U: An m×m orthogonal matrix whose columns are the left singular vectors of E.
    Σ: A diagonal m×n matrix containing the singular values of E in descending order.
    Vᵀ: The transpose of an n×n orthogonal matrix, where the columns are the right singular vectors of E.

    References: https://www.geeksforgeeks.org/machine-learning/singular-value-decomposition-svd/

    :param E: Description
    :type E: np.ndarray
    :return: Description
    :rtype: list
    """

    # Perform SVD on E
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotations
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]  # translation direction (up to scale)

    # Returns the four candidates
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def chirality_count(R, t, K, pts1, pts2):
    """
    Counts how many triangulated points are in front of both cameras
    """

    # Projection matrices
    # P1 = K [ I | 0 ]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # P = K [ R | t ]
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    # Triangulate points
    # X_h =
    # [[X1, X2, ..., XN],
    # [Y1, Y2, ..., YN],
    # [Z1, Z2, ..., ZN],
    # [W1, W2, ..., WN]]
    #
    # Each column is one 3D point in homogeneous form
    X_h = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = X_h[:3] / X_h[3] # Homogeneous normalization

    # Depth in camera 1
    z1 = X[2]  # Think about "This point is in front of camera 1"

    # Depth in camera 2
    X2 = R @ X + t.reshape(3, 1) # Matrix rotation + translation
    z2 = X2[2]  # Think about "This point is in front of camera 2"

    # Performs element-wise logical AND and sum the 1s
    # Think about "This point is in front of both cameras"
    return np.sum((z1 > 0) & (z2 > 0))
