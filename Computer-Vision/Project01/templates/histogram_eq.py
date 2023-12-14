import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    # Compute histogram
    histogram, bins = np.histogram(I.flatten(), bins=256, range=[0, 256])

    # Compute CDF
    cdf = histogram.cumsum()

    # Normalize the CDF to bins from 0 to 255
    cdf = 255 * cdf / cdf[-1]

    # Use linear interpolation of cdf to create contrast-enhanced image
    J = np.interp(I.flatten(), bins[:-1], cdf).reshape(I.shape)

    return J
