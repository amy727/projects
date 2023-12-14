import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """
    # Extract the focal length and principal point from K 
    # (Focal length is given to be identical in x and y)
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]

    # Subtract the principal point from the image plane point
    u = pt[0] - cx
    v = pt[1] - cy

    # Compute the elements of the Jacobian matrix
    J = np.array([
        [-f/z, 0, u/z, u*v/f, -(f**2 + u**2)/f, v],
        [0, -f/z, v/z, (f**2 + v**2)/f, -u*v/f, -u]
    ], dtype=np.float64)

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J