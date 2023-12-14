import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Initialize window size
    win_size = 9
    half_win = win_size // 2

    # Initialize SAD and offset arrays
    sad = np.zeros((2*maxd + 1, 1))
    offset = np.zeros((2*maxd + 1, 1))
    
    # Initialize disparity map
    Id = np.zeros(np.shape(Il)) 
    Il_pad = np.pad(Il, half_win, mode='edge')
    Ir_pad = np.pad(Ir, half_win, mode='edge')
    rows_pad, cols_pad = np.shape(Il_pad) # Dimensions of padded image

    # Loop over each image row, computing the local similarity measure, then aggregate
    for y in range(bbox[1, 0], bbox[1, 1] + 1):
        for x in range(bbox[0, 0], bbox[0, 1] + 1):
            i = 0

            # Extract window from left padded image
            win_left = Il_pad[y:y+win_size, x:x+win_size]
            
            # Loop over each disparity value for the disparity map
            for d in range(-maxd, maxd + 1):
                # Determine the window start and end points for the right image
                x_start = x + d
                x_end = x_start + win_size
                # If the window is within the padded right image, extract the window
                # and compute the SAD
                if (x_start >= 0) and (x_end < cols_pad):
                    win_right = Ir_pad[y:y+win_size, x_start:x_end]
                    sad[i, 0] = np.sum(np.abs(win_left - win_right))
                # Otherwise, assign a large SAD value
                else:
                    sad[i, 0] = float('inf')

                # Assign the disparity value to the offset array and increment the index
                offset[i, 0] = abs(d)
                i += 1

            # Find the index with the minimum SAD value
            best_match = np.argmin(sad)
            # Assign the corresponding disparity value to the disparity map
            Id[y,x] = offset[best_match, 0]

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id