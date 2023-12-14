# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite
import matplotlib.pyplot as plt

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('C:/Users/achen/OneDrive/Documents/04/ROB501/rob501_assignment_1/billboard/billboard_hacked.png')
    Ist = imread('C:/Users/achen/OneDrive/Documents/04/ROB501/rob501_assignment_1/billboard/uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    Ist = histogram_eq(Ist)

    # Compute the perspective homography we need...
    H, A = dlt_homography(Iyd_pts, Ist_pts)

    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!
    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!
    x_start, x_end = bbox[0,0], bbox[0,1]
    y_start, y_end = bbox[1,0], bbox[1,2]
    pts = Path(Iyd_pts.T)
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            if pts.contains_point((i,j)):
                # Get the Ist point
                pt = H @ np.array([i, j, 1])
                pt = pt / pt[2]
                pt = pt[:2].reshape(2,1)

                # Set all 3 channels to the same interpolated brightness
                brightness = bilinear_interp(Ist, pt)
                Ihack[j, i, :] = brightness

    #------------------

    plt.imshow(Ihack)
    plt.show()
    imwrite(Ihack, 'billboard_hacked.png')

    return Ihack

if __name__ == "__main__":
    billboard_hack()