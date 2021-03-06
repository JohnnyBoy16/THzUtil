"""
Utility module that contains a variety of unrelated functions that are useful
in data processing
"""
import pdb
import os
import copy

import pandas as pd
import numpy as np


def gaussian_1d(x, a, mu, sigma):
    """
    1D Gaussian function
    :param x: Location
    :param a: Amplitude at center
    :param mu: Center location
    :param sigma: standard deviation
    :return: Value of the function at the specified x location(s)
    """

    f = a * np.exp(-(x-mu)**2 / (2*sigma**2))

    return f


def gaussian_2d(x, y, a, x0, y0, std_x, std_y):
    """
    2D Gaussian function that can be used to curve fit or whenever a Gaussian
    function is needed for something. If trying to use to make a 2D function,
    params x and y should be 2D matrices from np.meshgrid
    :param x: The x array over which the data is to be calculated.
    :param y: The y array over which the data is to be calculated.
    :param a: The amplitude of the Gaussian
    :param x0: X center location
    :param y0: Y center location
    :param std_x: Standard Deviation of x
    :param std_y: Standard Deviation of y
    :return: The function value at the given (x, y) location
    """

    f = a * (np.exp(-((x-x0)**2/(2*std_x**2) + (y-y0)**2/(2*std_y**2))))

    return f


def expo_guassian_2d(x, y, a, x0, y0, std_x, std_y, lam_x, lam_y):
    """
    """

    zx = np.exp(-(np.abs(x)/lam_x) * (1-np.exp(-np.abs(x)/std_x)))
    zy = np.exp(-(np.abs(y)/lam_y) * (1-np.exp(-np.abs(y)/std_y)))

    return zx + zy


def sinc_2d(x, y, x0=0, y0=0, a=1, sigma=1, normalize=False):
    """
    2D sinc function
    :param x: A list of x coordinates
    :param y: A list of y coordinates
    :param x0: The x center of the central peak (Default = 0)
    :param y0: The y center of the central peak (Default = 0)
    :param a: The amplitude of the central peak (Default = 1)
    :param normalize: Whether or not to use the normalized sinc function
        (Default = False). See Wikipedia article for more information
    """

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    # setting the numpy error state to ignore divide by zero and invalid
    # floating operations within the with statement. In the np.where() function
    # call all possible outcomes are evaluated, so even though we have it set
    # to avoid division by zero errors it will still throw a warning. The
    # error are only ignored while in the with statement

    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            f = a * np.sin(np.pi**2*r) / (np.pi**2*r)

    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            f = a * np.where(r == 0, 1, np.sin(sigma*r) / (sigma*r))

    return f


def parabolic_equation(data, a, b, c, d, e):
    """
    Creates a paraboloid of the form ...
    f = ((x-a)/b)**2 + ((y-d)/e)**2 + c

    :param data: An array or single point over which to calculate the function.
        If an array expected to have the y or i values in the first row and the
        x or j values in the second row
    :param a: The x center of the parabola
    :param b: The y center of the parabola
    :param c: The height of the parabola
    :param d: Controls the width of the parabola in the x direction
    :param e: Controls the width of the parabola in the y direction
    :return: The value of the function at the specified point or points.
    """

    y, x = data

    paraboloid = ((x-a)/b)**2 + ((y-d)/e)**2 + c

    return paraboloid


def read_reference_data(filename, basedir=None, zero=True):
    """
    Reads in a reference file and returns the time values along with amplitudes

    :param filename: The name of the file to be loaded, or the full path if
        basedir is None
    :param basedir: The base directory that contains the file. If None,
        filename is expected to be the full path
    :param zero: If True (default), the reference signal time array is returned
        with zero as the first value. If False, no modification is made.

    :return: optical_delay: Array of time values
    :return: ref_amp: Array of amplitude values
    """

    # allow the user to enter full path in filename parameter
    if basedir is not None:
        filename = os.path.join(basedir, filename)

    # Read in the reference waveform and separate out the optical delay (time)
    # and the reference amplitude
    reference_data = pd.read_csv(filename, delimiter='\t')
    optical_delay = reference_data['Optical Delay/ps'].values
    ref_amp = reference_data['Raw_Data/a.u.'].values

    if zero:  # adjust optical delay so it starts at zero
        optical_delay -= optical_delay[0]

    return optical_delay, ref_amp


def make_c_scan(waveform, peak_bin, gate, signal_type, follow_gate_on=True):
    """
    Stand alone function to make a C-Scan given a waveform. This function is
    identical to the make_c_scan method in THzData.
    :param waveform: The waveform that is output by a THzProc code
    :param peak_bin: The peak_bin from a THzProc code
    :param gate: The gate from a THzProc Code
    :param signal_type: Determines how the C-Scan in made.
        signal_type choices
                0: Use Peak to Peak voltage with the front gates regardless of
                    whether follow gate is on or not.
                1: Use Peak to Peak voltage with the follow gates if on. If
                    follow gate is not on then use peak to peak voltage across
                    entire waveform
    :param follow_gate_on:
    """

    y_step = waveform.shape[0]
    x_step = waveform.shape[1]
    c_scan = np.zeros((y_step, x_step))

    if follow_gate_on:
        idx = 1
    else:
        idx = 0

    # use Vpp within the front gates regardless of whether follow gate is on or
    # not the gates are ABSOLUTE INDEX POSITION and DO NOT account for
    # differences in height of the front surface
    if signal_type == 0:
        max_amp = np.amax(waveform[:, :, gate[0][0]:gate[0][1]], axis=2)
        min_amp = np.amin(waveform[:, :, gate[0][0]:gate[0][1]], axis=2)
        c_scan = max_amp - min_amp

    # use Vpp within the follow gates if one, else use Vpp across the entire
    # A-Scan.
    elif signal_type == 1:
        if follow_gate_on:
            idx = 1
        else:
            idx = 0
        for i in range(y_step):
            for j in range(x_step):
                max_amp = waveform[i, j, peak_bin[0, idx, i, j]]
                min_amp = waveform[i, j, peak_bin[1, idx, i, j]]
                c_scan[i, j] = max_amp - min_amp

    return c_scan


def clear_small_defects(binary_image, min_area):
    """
    Clears defects that are smaller than the given area (in pixels) from the
    binary_image
    :param binary_image: A binary image
    :param min_area: The smallest number of pixels a defect can contain and
        still be considered a defect. A region that has exactly this number of
        pixels will not be removed.
    """
    from skimage.measure import regionprops, label

    return_binary = copy.deepcopy(binary_image)

    labeled_image = label(binary_image)

    for defect in regionprops(labeled_image):
        if defect.area < min_area:
            for loc in defect.coords:
                # use not expression, so it should work regardless of whether
                # defects are 1's and background is 0's or defects are 0's and
                # background is 1's
                return_binary[loc[0], loc[1]] = \
                    not return_binary[loc[0], loc[1]]

    return return_binary


def combine_close_defects(region_list, bbox_list):
    """
    Driving function to find close defects
    """
    n_defects = len(region_list)
    global_found = list()
    for i in range(n_defects):
        if i in [defect for local_flaw in global_found for defect in local_flaw]:
            continue

        home_flaw = region_list[i]
        bbox = bbox_list[i]
        # global_found.append(i)

        # coords includes the defects own coordinates and the coordinates of
        # other defects it there are any nearby
        _, local_found = _search_for_nearby_defect(home_flaw, region_list,
                                                   bbox, bbox_list)

        for found in local_found:
            for g in global_found:
                if found in g:
                    local_found += [flaw for flaw in g if flaw not in local_found]
                    global_found.remove(g)
                    break

        global_found.append(local_found)

    defect_coords = list()
    for i, local_list in enumerate(global_found):
        for j, defect_num in enumerate(local_list):
            if j == 0:
                coords = region_list[defect_num].coords
            else:
                coords = np.r_[coords, region_list[defect_num].coords]

        defect_coords.append(coords)
    # pdb.set_trace()
    return defect_coords


# private function
def _search_for_nearby_defect(home_defect, defect_list, bbox, bbox_list,
                              local_found=None):
    """
    Recursively searches a defect for nearby defects. A defect is considered
    nearby if at least 1 of its own coordinates is inside of another defect's
    bounding box.
    :param home_defect: The defect this is looking for other defects
    :param defect_list: A list of all defects in the sample area
    :param bbox: home_defect's bounding box
    :param bbox_list: A list of the bounding boxes for all defects in the
        sample area
    :param already_found: A list of defects that have already been considered
        by the algorithm
    :return own_coords: An array of coordinates of home_defects own coords and
        any other nearby defects coordinates. If no nearby defects are found,
        just returns its own coordinates.
    """

    own_coords = home_defect.coords

    if local_found is None:
        local_found = list()
        for i, defect in enumerate(defect_list):
            if defect == home_defect:
                local_found.append(i)

    for i, defect in enumerate(defect_list):
        if defect == home_defect or i in local_found:
            continue

        # whether or not the defect is within the bounding rows of home_defect
        in_bb_rows = (np.any(defect.coords[:, 0] > bbox[0]) and
                      np.any(defect.coords[:, 0] < bbox[2]))

        # whether or not the defect is within the bounding columns of
        # home_defect
        in_bb_cols = (np.any(defect.coords[:, 1] > bbox[1]) and
                      np.any(defect.coords[:, 1] < bbox[3]))

        # if defect is within both the bounding row and bounding columns of
        # home defect that it is within the bounding box
        if in_bb_rows and in_bb_cols:
            local_found.append(i)
            defect_coords, local_found \
                = _search_for_nearby_defect(defect, defect_list, bbox_list[i],
                                            bbox_list, local_found)

            own_coords = np.r_[own_coords, defect_coords]

    return own_coords, local_found
