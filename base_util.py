"""
Utility module that contains a variety of unrelated functions that are useful
in data processing
"""
import os

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


def gaussian_2d(data, a, x0, y0, std_x, std_y):
    """
    2D Gaussian function that can be used to curve fit or whenever a Gaussian
    function is needed for something.
    :param data: A 2xn matrix of location where the data is to be calculated at
        Y locations are in the first row and X locations are the 2nd
    :param a: The amplitude of the Gaussian
    :param x0: X center location
    :param y0: Y center location
    :param std_x: Standard Deviation of x
    :param std_y: Standard Deviation of y
    :return: The function value at the given (x, y) location
    """

    y, x = data

    f = a * np.exp(-((x-x0)**2/(2*std_x**2) + (y-y0)**2/(2*std_y**2)))

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
