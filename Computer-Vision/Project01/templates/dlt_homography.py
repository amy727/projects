import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def normalize(pts):
    """
    Normalize the points
    """
    # Get centroid
    centroid = np.mean(pts, axis=1)

    # Get centered points
    pts_centered = pts - centroid[:, np.newaxis]
    
    # Calculate the scale factor
    distances = norm(pts_centered, axis=0)
    scale = np.sqrt(2) / np.mean(distances)

    # Compute similarity transform T
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    
    # Normalize the points
    pts = np.vstack((pts, np.ones(pts.shape[1])))
    pts = np.dot(T, pts)
    pts = pts[:2] / pts[2]

    return T, pts

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
    # We will use the normalized DLT algorithm
    # Normalize the points
    T1, I1pts_normalized = normalize(I1pts)
    T2, I2pts_normalized = normalize(I2pts)
    
    # Initialize A
    A = np.zeros((8, 9))

    # Construct A based on the given points
    # Four 2×9 A_i matrices are stacked on top of one another
    for i in range(4):
        x, y = I1pts_normalized[:, i]
        u, v = I2pts_normalized[:, i]

        A[i*2] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[i*2 + 1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]

    # Compute the null space of A
    nullSpace = null_space(A)

    # The 1D null space of A is the solution space for H
    H = nullSpace[:, -1].reshape(3, 3)
    H = inv(T2) @ H @ T1
    H = H / H[2,2]

    return H, A
