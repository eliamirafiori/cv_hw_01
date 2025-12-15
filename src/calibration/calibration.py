import numpy as np
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D visualization
from matplotlib import pyplot as plt
import glob
import os

"""
================================================================================
CAMERA CALIBRATION USING CHESSBOARD PATTERNS
================================================================================

This section describes the process of calibrating a camera to estimate its
intrinsic parameters and lens distortion coefficients.

Camera calibration is a prerequisite for metric 3D reconstruction, as it
ensures that image points can be accurately related to real-world coordinates.

------------------------------------------------------------------------------
1. GOAL OF CAMERA CALIBRATION
------------------------------------------------------------------------------

The goal is to determine:

    - Intrinsic matrix K:
        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

      where fx, fy are focal lengths (pixels),
      and (cx, cy) is the principal point (optical center).

    - Lens distortion coefficients:
        dist = [k1, k2, p1, p2, k3]

      which account for:
        * Radial distortion (k1, k2, k3)
        * Tangential distortion (p1, p2)

Accurate K and distortion coefficients allow:
    - Correcting distorted images (undistortion)
    - Converting image coordinates to normalized camera coordinates
    - Computing accurate Fundamental and Essential Matrices

------------------------------------------------------------------------------
2. CHESSBOARD CALIBRATION METHOD
------------------------------------------------------------------------------

A planar chessboard pattern is used because:
    - The positions of inner corners are known precisely in real-world units
    - The pattern provides a dense grid of well-defined, repeatable points

Calibration steps:

1. Define 3D object points for the chessboard corners in the Z=0 plane:
       X = (i*square_size, j*square_size, 0)
   for i=0..columns-2, j=0..rows-2

2. Detect corresponding 2D image points in each calibration image:
       x = (u, v)
   using `cv.findChessboardCorners` and subpixel refinement with
   `cv.cornerSubPix`.

3. Store object points and image points for all images.

------------------------------------------------------------------------------
3. CALIBRATION SOLUTION
------------------------------------------------------------------------------

OpenCV's `cv.calibrateCamera` estimates:

    - Intrinsic parameters K
    - Distortion coefficients
    - Extrinsic parameters (R, t) for each calibration image

Mathematically, calibration solves:

    min Σ_i || x_i - project(P_i, X_i) ||²

where:
    - project(P_i, X_i) projects the 3D object points into the image
    - P_i = K [R_i | t_i]
    - The sum is over all points and images

The solution minimizes the **reprojection error**, which is the Euclidean
distance (in pixels) between detected and projected points.

------------------------------------------------------------------------------
4. TERMINATION CRITERIA AND SUBPIXEL REFINEMENT
------------------------------------------------------------------------------

- Corner refinement uses iterative optimization:
    - Stops after maximum iterations or when convergence is below epsilon
    - Improves calibration precision

- High-precision corners reduce RMS reprojection error and improve downstream
  3D reconstruction.

------------------------------------------------------------------------------
5. REPROJECTION ERROR
------------------------------------------------------------------------------

Two error metrics are computed:

1. RMS error from `cv.calibrateCamera`:
    - Represents the root-mean-square reprojection error across all points

2. Mean per-image reprojection error:
    - Computes the average pixel error per image
    - Good calibration is typically <0.08 pixels

------------------------------------------------------------------------------
6. EXTRINSIC PARAMETERS
------------------------------------------------------------------------------

- For each calibration image, OpenCV returns:
    - Rotation vector (Rodrigues) → 3×3 rotation matrix
    - Translation vector → camera position in world coordinates

These extrinsics describe the pose of the camera relative to the chessboard
pattern in each calibration image.

------------------------------------------------------------------------------
7. IMAGE UNDISTORTION
------------------------------------------------------------------------------

Once K and distortion coefficients are known:

    - Images can be undistorted using `cv.undistort`
    - This removes radial and tangential distortion
    - Improves feature matching, epipolar geometry, and 3D reconstruction

------------------------------------------------------------------------------
8. DATA STORAGE
------------------------------------------------------------------------------

Calibration parameters are saved to a `.npz` file for later use:

    - K, distortion coefficients, RMS error, rotation vectors, translation vectors

This allows the reconstruction pipeline to:
    - Load calibration parameters without re-calibrating
    - Ensure reproducibility and consistency

------------------------------------------------------------------------------
9. PRACTICAL NOTES
------------------------------------------------------------------------------

- Ensure the chessboard pattern is fully visible in multiple images
- Use images from different angles and distances for robust calibration
- Subpixel corner detection significantly reduces reprojection error
- Calibration accuracy directly affects:
      * Essential/Fundamental matrix estimation
      * Triangulation
      * 3D point cloud accuracy

------------------------------------------------------------------------------
REFERENCES
------------------------------------------------------------------------------

- Zhang, Z. "A Flexible New Technique for Camera Calibration"
- Hartley, R., Zisserman, A. "Multiple View Geometry in Computer Vision"
- OpenCV Documentation (Camera Calibration)
  
================================================================================
"""


def load_calibration(path: str = "assets/calibration/phone/horizontal/calibration.npz"):
    if not os.path.exists(path):
        print("Calibration file not found.")
        return None, None

    with np.load(path) as data:
        K = data["K"]
        dist = data["dist_coeffs"]
        return K, dist


def calibration(
    calibration_assets_path: str = "assets/calibration/phone/horizontal",
    columns: int = 8,
    rows: int = 8,
    square_size: float = 1,
    debug: bool = False,
):
    """
    Calibration function
    """

    # Termination criteria for corner refinement (sub-pixel accuracy)
    # Stops when either:
    #  - max iterations are reached
    #  - or the desired accuracy is achieved
    criteria = (
        cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,
        30,  # max number of iterations
        0.001,  # minimum required accuracy (epsilon)
    )

    # Chessboard configuration
    inner_corners = (columns - 1, rows - 1)  # number of INNER corners (columns, rows)

    # Prepare 3D object points in real-world coordinates
    # The chessboard lies on the Z = 0 plane
    objp = np.zeros((inner_corners[0] * inner_corners[1], 3), np.float32)

    # Generate grid and scale it by the square size (meter)
    # ':'	All rows, ':2'	First two columns only (index 0 and 1)
    objp[:, :2] = (
        np.mgrid[0 : inner_corners[0], 0 : inner_corners[1]].T.reshape(-1, 2)
        * square_size
    )

    # Containers for calibration points
    objpoints = []  # 3D points in real-world space (meter)
    imgpoints = []  # 2D points in image plane (pixels)

    # Load all calibration images from disk
    # Each image should show the same chessboard pattern
    images = glob.glob(f"{calibration_assets_path}/*.jpg")

    # Loop over each calibration image
    for img_path in images:
        if debug:
            print(f"Path:\n\t{img_path}")

        # Read image from disk (OpenCV loads images in BGR format)
        img_bgr = cv.imread(img_path)

        # Convert image to grayscale
        # Chessboard detection works on single-channel images
        img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

        # Detect chessboard inner corners
        #
        # corners_found:
        #   - True if all expected corners are detected
        # corners:
        #   - Detected corner locations (pixel coordinates)
        #
        # chessboard_size = (columns, rows)
        # Must match the object points definition exactly
        corners_found, corners = cv.findChessboardCorners(img_gray, inner_corners, None)

        if debug:
            print(f"Corners found:\n\t{corners_found}")

        # If the chessboard was successfully detected
        if corners_found:

            # Store the known 3D object points (real-world coordinates)
            # Same for every image, since the chessboard geometry is fixed
            objpoints.append(objp)

            # Refine corner positions to sub-pixel accuracy
            #
            # This improves calibration precision significantly
            #
            # (11, 11)  -> search window size
            # (-1, -1)  -> use default dead zone
            # criteria  -> termination criteria defined earlier
            corners_refined = cv.cornerSubPix(
                img_gray, corners, (11, 11), (-1, -1), criteria
            )

            # Store the refined 2D image points (pixel coordinates)
            imgpoints.append(corners_refined)

            if debug:
                # Visual feedback: draw detected corners on the image
                cv.drawChessboardCorners(
                    img_bgr, inner_corners, corners_refined, corners_found
                )

                # Display the image briefly
                cv.imshow(
                    "Calibration Image",
                    cv.resize(
                        img_bgr,
                        (img_bgr.shape[1] // 4, img_bgr.shape[0] // 4),
                    ),
                )
                cv.waitKey(500)  # display for 500 ms

    if debug:
        cv.destroyAllWindows()

    # Use any image size from your dataset
    image_shape = cv.imread(images[0]).shape[:2][::-1]  # width, height

    # Camera calibration
    #
    # Inputs:
    #  - objpoints : list of 3D real-world points (meter)
    #  - imgpoints : list of corresponding 2D image points (pixels)
    #  - image size: (width, height)
    #
    # Outputs:
    #  - rms_error  : RMS re-projection error
    #  - K          : camera intrinsic matrix (3x3)
    #  - dist_coeffs: distortion coefficients (5x1)
    #  - rot_vecs   : rotation vectors (3x1) (one per image)
    #  - trans_vecs : translation vectors (3x1) (one per image)
    #
    # OpenCV uses Rodrigues vectors to represent rotation
    # - 3 numbers → axis-angle representation
    # - Converts to a 3×3 rotation matrix using:
    #   - R, _ = cv.Rodrigues(rot_vecs[i])
    rms_error, K, dist_coeffs, rot_vecs, trans_vecs = cv.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    if debug:
        # Compute the mean re-projection error (in pixels) over the calibration images
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(
                objpoints[i], rot_vecs[i], trans_vecs[i], K, dist_coeffs
            )
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        print(f"Camera Matrix K:\n{K}")
        print(f"Re-projection Error:\n\t{rms_error}")
        # The error is good when it's under 0.08
        print(f"Mean Re-projection Error (in pixels):\n\t{mean_error / len(objpoints)}")

        # Iterate over all images to show their individual poses
        for i, (r_vec, t_vec) in enumerate(zip(rot_vecs, trans_vecs)):
            R, _ = cv.Rodrigues(r_vec)
            print(f"\nImage {i} Pose")
            print(f"Rotation Matrix R:\n{R}")
            print(f"Translation t:\n{t_vec}")

    # Save calibration parameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    param_path = os.path.join(calibration_assets_path, "calibration.npz")

    # Save several arrays into a single file in uncompressed .npz format
    np.savez(
        param_path,
        rms_error=rms_error,
        K=K,
        dist_coeffs=dist_coeffs,
        rot_vecs=rot_vecs,
        trans_vecs=trans_vecs,
    )

    return K, dist_coeffs
