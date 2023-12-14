import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# You may add support functions here, if desired.

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    # Initialize A
    A = np.zeros((8, 9))

    # Construct A based on the given points
    # Four 2×9 A_i matrices are stacked on top of one another
    for i in range(4):
        x, y = I1pts[:, i]
        u, v = I2pts[:, i]

        A[i*2] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[i*2 + 1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]

    # Compute the null space of A
    nullSpace = null_space(A)

    # The 1D null space of A is the solution space for H
    H = nullSpace[:, -1].reshape(3, 3)
    H = H / H[2,2]

    return H, A

def saddle_point(I):
    """
    Locate saddle point in an image patch.
    """
    # Get the number of rows and columns in the image patch.
    numRows, numCols = I.shape

    # Create a grid of coordinates corresponding to the image patch.
    y, x = np.mgrid[:numRows, :numCols]

    # Get flattened array for each coordinate direction.
    x = x.flatten()
    y = y.flatten()
    z = I.flatten()

    # Least squares fit on hyperbolic paraboloid.
    A = np.vstack([x**2, x*y, y**2, x, y, np.ones(len(x))])
    alpha, beta, gamma, delta, epsilon, zeta = lstsq(A.T, z, rcond = None)[0]

    # Compute saddle point location.
    reg_term = 1e-6
    pt = - inv(np.array([[2*alpha, beta], [beta, 2*gamma]]) + reg_term * np.eye(2)) @ np.array([[delta], [epsilon]])

    return pt

def refine_corners(I, corners, half_len = 20):
    corners_refined = []

    for x, y in corners.T:
        # Get the patch around the corner.
        patch = I[y-half_len:y+half_len, x-half_len:x+half_len]

        # Get the saddle point.
        pt = saddle_point(patch)

        # Add the corner to the list.
        corners_refined.append([x + pt[0] - half_len, y + pt[1] - half_len])

    return np.array(corners_refined).T


def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    # Estimate the world points bounding box
    square_len = abs(Wpts[0,1] - Wpts[0,0])
    W_xmin = min(Wpts[0, :]) - square_len*1.5
    W_xmax = max(Wpts[0, :]) + square_len*1.5
    W_ymin = min(Wpts[1, :]) - square_len*1.2
    W_ymax = max(Wpts[1, :]) + square_len*1.2 
    Wpts_box = np.array([[W_xmin, W_xmax, W_xmax, W_xmin], [W_ymin, W_ymin, W_ymax, W_ymax]])

    # Estimate the homography between the world to the image
    H, A = dlt_homography(Wpts_box, bpoly)

    # Transform the world points to image points
    Wpts[-1] = 1
    pts = H @ Wpts
    pts = pts / pts[-1]
    pts = np.round(pts[:-1]).astype(int)

    # Refine the corners using the saddle point method.
    Ipts = refine_corners(I, pts, 25)
    Ipts = Ipts.reshape(2, -1)

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts