import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_depth_finder(K, pts_obs, pts_prev, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    J = np.zeros((2*n, 6))
    zs_est = np.zeros(n)

    A = np.zeros((2*n, n))
    b = np.zeros(2*n)

    vel_trans = v_cam[:3]
    vel_rot = v_cam[3:]

    for i in range(n):
        # Compute the Jacobian for each point
        J = ibvs_jacobian(K, pts_obs[:, i], 1)

        # Populate matrix A
        A[2*i:2*i+2, i] = (J[:, :3] @ vel_trans).reshape(-1)

        # Compute and populate vector b
        b[2*i:2*i+2] = (pts_obs[:, i] - pts_prev[:, i]) - (J[:, 3:] @ vel_rot).reshape(-1)

    # Solve for X using least squares
    X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Inverse of X to get estimated depths
    zs_est = 1/X
    zs_est = zs_est.reshape(-1)

    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (n,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est