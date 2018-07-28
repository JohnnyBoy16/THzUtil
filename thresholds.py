"""
A collection of thresholding functions
"""
import pdb

import numpy as np


def my_threshold_triangle(image, n_bins=256, offset=0, tail_side=None):
    """
    My implementation of the triangle threshold to compare to results from
    skimage.filters.threshold_traingle. This threshold is useful to separate a
    tail of pixels from a large distribution.
    See skimage code @
    https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/thresholding.py

    "Automatic Measurement of Sister Chromatid Exchange Frequency", Zack,
    Rogers, & Latt, 1977. Mainly figure 2. This function can include an offset
    as shown in figure 2 in reference, by passing a value for offset.

    :param image: The image to threshold
    :param n_bins: The number of bins to use when making the histogram.
            Default: 256 (8-bit image)
    :param offset: The value that the threshold should be offset from the
            position of maximum distance. Use a value > 0 to move the threshold
            value away from the histogram peak, and a value < 0 to move the
            threshold value closer to the histogram peak. A positive value for
            offset will move the threshold away from the histogram peak and a
            negative value will move the threshold towards the peak regardless
            of whether the tail is on the low or high side.
    :param tail_side: (default: None, otherwise pass 'high', or 'low'.
            Manually set what side the tail will be on. If left as None, the
            algorithm will declare the tail is on the side that the histogram
            peak is not one. For example if there are 100 bins and the histogram
            peak is at bin 15, then the threshold will be > 15. If the peak was
            at bin 80, then the threshold will be less than 80. Passing through
            either 'high' or 'low' will force the threshold to be higher or
            lower than the peak regardless of the peaks position. For example if
            tail_side = 'high' and the threshold peak is located at bin 75 out
            of 100, the threshold will be larger than 75.
    """

    # also see discussion on the ImageJ implementation here
    # http://forum.imagej.net/t/understanding-imagej-implementation-of-the-triangle-algorithm-for-threshold/752

    hist, bin_edges = np.histogram(image, n_bins)

    max_idx = hist.argmax()

    # if max index is closer to n_bins than 0, assume that tail is on the left
    # and not the right. i.e. more bright values than dark ones. If this is the
    # case the histogram will be flipped so the tail is on the right.
    flip = max_idx > n_bins - max_idx
    if flip and tail_side is None or tail_side == 'low':
        hist = np.flip(hist, axis=0)
        max_idx = hist.argmax()  # recalculate argmax with flipped histogram

    # normalize the histogram, so maximum value is 1
    hist_normalized = hist / hist.max()

    # how much the triangle height decreases at each step
    step = 1 / (n_bins - max_idx - 1)

    # create a unit triangle with argmax() and width
    width = n_bins - max_idx
    x = np.arange(width)  # list of x values from max_idx to last bin
    y = 1 - x * step  # list of y values that define triangle hypotenuse

    # calculate the vertical distance between the hypotenuse of the triangle
    # and the top of the bin edges
    height = y - hist_normalized[max_idx:]

    # the threshold is the index where the maximum hypotenuse distance occurs
    # however the hypotenuse is only a function of the height, so the location
    # of max height will also be the location of max hypotenuse length
    thresh_idx = height.argmax() + max_idx

    # the step size of each index between the histogram argmax and n_bins has
    # already been calculated as step. The offset value in indexes is determined
    # be dividing the offset (value between 0-1) by the step size per index.
    if offset != 0:
        # need to convert to int after rounding so this value can be used as an
        # index later
        thresh_idx += int(round(offset / step, 0))

    # if histogram has been flipped, threshold needs to be adjusted to be
    # relative to right side
    if flip:
        thresh_idx = n_bins - thresh_idx - 1

    # numpy histogram function returns bin edges, so let actual threshold value
    # be the average of the bin values (middle of bin)
    # this matches what is returned from skimge.fiters.threshold_triangle
    threshold = (bin_edges[thresh_idx] + bin_edges[thresh_idx+1]) / 2

    return threshold


def min_error_threshold(hist, n_bins=256):
    """
    Algorithm to compute the minimum error threshold that is defined by Kittler
    & Illingworth. This algorithm is designed to separate an image with two
    Gaussian distributed densities. This method is declared as one of the best
    algorithms by two different survey papers on thresholding techniques.

    Original Paper
    "Minimum Error Thresholding", Kittler & Illingworth, 1985

    Survey Papers
    "An Analysis of Histogram-Based Thresholding Algorithms", Glasby, 1993
    "Survey over image thresholding techniques and quantitative performance
        evaluation", Sezgin & Sankur, 2004

    :param image: The grayscale image to be thresholded
    :param n_bins: The number of bins to use in the histogram
        (Default=256, 8-bit)
    :param hist: Set to True if
    """
    # for now assume that user passes threshold
    # hist, bin_edges = np.histogram(image, n_bins)

    # J is the criterion function that we will seek to minimize, this is
    # equation 15 in Kittler & Illingworth's paper.
    J = np.zeros(n_bins)
    for i in range(n_bins):
        i1 = np.arange(i+1)
        i2 = np.arange(i+1, n_bins)
        # calculate the percentage of pixels on each side of the histogram
        # threshold, start at
        P1 = np.sum(hist[:i+1])  # according to paper this is inclusive
        P2 = np.sum(hist[i+1:])

        # calculate the mean bin for each density
        mu1 = np.sum(hist[:i+1]*i1) / P1
        mu2 = np.sum(hist[i+1:]*i2) / P2

        # calculate variance for each density
        var1 = np.sum(hist[:i+1] * (i1-mu1)**2) / P1
        var2 = np.sum(hist[i+1:] * (i2-mu2)**2) / P2

        sigma1 = np.sqrt(var1)
        sigma2 = np.sqrt(var2)

        J[i] = 1 + 2 * (P1*np.log10(sigma1) + P2*np.log10(sigma2) -
                        2 * (P1*np.log10(P1) + P2*np.log10(P2)))

    J[np.where(J == -np.inf)] = np.nan

    T = np.nanmin(J)
    T = np.where(T == J)[0][0]

    # thresh = (bin_edges[T] + bin_edges[T+1]) / 2

    return T
