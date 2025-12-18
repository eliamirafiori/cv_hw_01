import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def draw_epipolar_lines(img1, pts1, img2, pts2, F):
    """
    Visualizes epipolar lines.
    For every point in one image, the Fundamental Matrix determines a line in the other image
    on which the corresponding point MUST lie.
    """

    # Right-to-Left Epilines
    # We take points from the RIGHT image (pts2) and calculate where their
    # corresponding epipolar lines should be in the LEFT image.
    # The '2' argument tells OpenCV the points are from the second image.
    # F maps p2 -> line1 (l = F.T * p2)
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(
        -1, 3
    )  # Reshape to N x 3 (a, b, c for line ax + by + c = 0)

    # Draw these lines on img1 (Left Image)
    # img5 will show lines on Left Image, img6 shows the points on Right Image
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Left-to-Right Epilines
    # We take points from the LEFT image (pts1) and find lines in the RIGHT image.
    # The '1' argument tells OpenCV the points are from the first image.
    # F maps p1 -> line2 (l = F * p1)
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    # Draw these lines on img2 (Right Image)
    # img3 will show lines on Right Image, img4 shows the points on Left Image
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    # Visualization
    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(img5)
    plt.title("Epilines on Left Image"), plt.axis("off")
    plt.subplot(122), plt.imshow(img3)
    plt.title("Epilines on Right Image"), plt.axis("off")
    plt.show()


def drawlines(img1, img2, lines, pts1, pts2):
    """
    Helper function to draw lines on one image and matching points on the other.

    Args:
        img1: Image on which we draw the epilines (lines).
        img2: Image on which we draw the points (pts2) that generated those lines.
        lines: The epipolar lines coefficients (a, b, c) where ax + by + c = 0
        pts1: Points in img1 (that should fall ON the lines).
        pts2: Points in img2 (that generated the lines via F).
    """

    # Get image dimensions (rows, cols) to calculate start/end points of lines
    r, c = img1.shape

    # Convert to BGR so we can draw colored lines/circles
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    # Iterate through lines and points simultaneously
    # 'lines' corresponds to pts2 (source) and matches pts1 (destination)
    for r, pt1, pt2 in zip(lines, pts1, pts2):

        # Random color for each pair to distinguish them
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # --- Line Calculation ---
        # Epipolar line equation: ax + by + c = 0
        # We need two points (x0, y0) and (x1, y1) to draw the line segment.
        # r[0]=a, r[1]=b, r[2]=c

        # Point 1 (Left edge, x=0):
        # a(0) + by + c = 0  =>  by = -c  =>  y = -c/b
        x0, y0 = map(int, [0, -r[2] / r[1]])

        # Point 2 (Right edge, x=c (width)):
        # a(c) + by + c = 0  =>  by = -(c + ac)  =>  y = -(c + ac)/b
        # Note: variable 'c' here is image width (cols), r[2] is line coefficient 'c'
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        # Draw the epipolar line on the target image
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)

        # Draw the corresponding point on the target image (should be ON the line)
        # Note: pt1 needs to be integer tuple for cv.circle
        img1 = cv.circle(img1, tuple(np.int32(pt1)), 5, color, -1)

        # Draw the source point on the source image (just for reference)
        img2 = cv.circle(img2, tuple(np.int32(pt2)), 5, color, -1)

    return img1, img2
