import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
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
    alpha, beta, gamma, delta, epsilon, zeta = lstsq(A.T, z, rcond=None)[0]

    # Compute saddle point location.
    pt = - inv(np.array([[2*alpha, beta], [beta, 2*gamma]])) @ np.array([[delta], [epsilon]])

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt