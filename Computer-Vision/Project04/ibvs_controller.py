import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_controller(K, pts_des, pts_obs, zs, gain):
    """
    A simple proportional controller for IBVS.

    Implementation of a simple proportional controller for image-based
    visual servoing. The error is the difference between the desired and
    observed image plane points. Note that the number of points, n, may
    be greater than three. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K       - 3x3 np.array, camera intrinsic calibration matrix.
    pts_des - 2xn np.array, desired (target) image plane points.
    pts_obs - 2xn np.array, observed (current) image plane points.
    zs      - nx0 np.array, points depth values (may be estimated).
    gain    - Controller gain (lambda).

    Returns:
    --------
    v  - 6x1 np.array, desired tx, ty, tz, wx, wy, wz camera velocities.
    singularity - True if we've reached a singularity, else False.
    """
    v = np.zeros((6, 1))
    singularity = False

    # Total number of points
    num_pts = pts_des.shape[1]
    
    # Initialize the stacked Jacobian matrix and error vector
    J = np.zeros((2*num_pts, 6))
    error = np.zeros((2*num_pts, 1))

    # Fill in the Jacobian matrix and error vector
    for i in range(num_pts):
        J[2*i:2*i+2, :] = ibvs_jacobian(K, pts_obs[:, i], zs[i])
        error[2*i:2*i+2, :] = (pts_des[:, i] - pts_obs[:, i]).reshape(-1,1)

    # Determine the Moore-Penrose pseudoinverse of the Jacobian
    try:
        J_inv = inv(J.T @ J) @ J.T
    except np.linalg.LinAlgError:
        print("Singular matrix, cannot compute inverse!")
        singularity = True
        return v, singularity

    # Compute the desired camera velocity
    v = gain * J_inv @ error

    correct = isinstance(v, np.ndarray) and \
        v.dtype == np.float64 and v.shape == (6, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return v, singularity