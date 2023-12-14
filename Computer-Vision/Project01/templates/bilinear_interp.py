import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    # Point (x,y) is surround by points (x1,y1), (x1,y2), (x2,y1), and (x2,y2)
    x = pt[0,0]
    y = pt[1,0]
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = int(np.ceil(x))
    y2 = int(np.ceil(y))

    # Calculate distances between points
    dx1 = x - x1
    dy1 = y - y1
    dx2 = x2 - x
    dy2 = y2 - y

    # Interpolate the brightness value using bilinear interpolation
    b = dx2*dy2*I[y1,x1] + dx1*dy2*I[y1,x2] + dx2*dy1*I[y2,x1] + dx1*dy1*I[y2,x2]
    b = round(b)

    return b
