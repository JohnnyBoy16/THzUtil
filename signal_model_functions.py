"""
Contains functions that are used when modeling a signal, such as reflection
and transmission coefficients.
"""
import pdb

import numpy as np


def reflection_coefficient(n1, n2, theta1=0.0, theta2=0.0):
    """
    Determine the reflection coefficient of a media transition with parallel
    polarized light. If n2 is given as np.inf, returns -1 for the reflection
    coefficient.
    :param n1: the refractive index of the media in which coming from
    :param n2: the refractive index of the media in which going to
    :param theta1: the angle of the incident ray, in radians
    :param theta2: the angle of the transmitted ray, in radians
    :return: The reflection coefficient
    """

    if np.abs(n1) == np.inf:
        return 0

    if np.abs(n2) == np.inf:
        return -1

    num = n1*np.cos(theta2) - n2*np.cos(theta1)
    denom = n1*np.cos(theta2) + n2*np.cos(theta1)

    r = num / denom

    return r


def transmission_coefficient(n1, n2, theta1=0, theta2=0):
    """
    Determine the transmission coefficient of a media transmission, independent
    of polarization
    :param n1: the refractive index of the media in which coming from
    :param n2: the refractive index of the media in which going to
    :param theta1: the angle of the incident ray, in radians
    :param theta2: the angle of the transmitted ray, in radians
    :return: The transmission coefficient
    """

    if np.abs(n1) == np.inf or np.abs(n2) == np.inf:
        return 0

    num = 2 * n1 * np.cos(theta1)
    denom = n1*np.cos(theta2) + n2*np.cos(theta1)

    t = num / denom

    return t


def get_theta_out(n0, n1, theta0):
    """
    Uses Snell's law to calculate the outgoing angle of light
    :param n0: The index of refraction of the incident media
    :param n1: The index of refraction of the outgoing media
    :param theta0: The angle of the incident ray in radians
    :return: theta1: The angle of the outgoing ray in radians
    """

    if np.abs(n0) == np.inf:
        return 0

    return np.arcsin(n0/n1 * np.sin(theta0))


def phase_screen_r(h, k, theta=0):
    """
    Adds the phase screen term from Jim Rose's paper for the reflection
    coefficient
    :param h: The standard deviation or RMS height of the surface in m
    :param k: The angular wavenumber of the incoming beam in meters
    :param theta: The angle of the incoming ray in radians
    :return: The exponential term to be multiplied by the reflection coefficient
    """
    # See Ref
    # "Surface roughness and th ultrasonic detection of subsurface scatters"
    # Nagy and Rose (1992)
    term = np.exp(-2 * h**2 * k**2 * np.cos(theta)**2)
    return term


def phase_screen_t(h, k1, k2, theta1=0, theta2=0):
    """
    Adds the phase screen term from Jim Rose's paper for the transmission
    coefficient
    :param h: The standard deviation or RMS height of the surface in meters
    :param k1: The angular wavenumber of the incoming beam in meters
    :param k2: The angular wavenumber of the outgoing beam in meters
    :param theta1: The angle of the incoming ray in radians
    :param theta2: The angle of the outgoing ray in radians
    :return: The exponential term that is to be multiplied by the transmission
    coefficient
    """
    # See Ref
    # "Surface roughness and th ultrasonic detection of subsurface scatters"
    # Nagy and Rose (1992)
    term = np.exp(-0.5*h**2 * (k2*np.cos(theta2) - k1*np.cos(theta1)) ** 2)
    return term


def global_reflection_model(n, theta, freq, d, n_layers, c=0.2998):
    """
    Calculates the global reflection coefficient given in Orfanidis
    "Electromagnetic Waves & Antennas". The global reflection coefficient is
    used to solve multilayer problems.
    :param n: The index of refraction of each of the layers, including the two
        half space media on either side, expected to be len(n_layers + 2)
    :param d: The thickness of each of the slabs, should be of length n_layer
    :param theta: The angle of the beam in each material including the two media
        on either side, length is n_layers+2
    :param freq: An array of frequencies over which to calculate the coefficient
    :param n_layers: The number of layers in the structure
    :param c: The speed of light (default = 0.2998 mm/ps)
    :return: The global reflection coefficient over the supplied frequency range
    """
    try:
        r = np.zeros((n_layers+1, len(freq)), dtype=complex)
        gamma = np.zeros((n_layers+1, len(freq)), dtype=complex)
    except TypeError:
        r = np.zeros(n_layers+1, dtype=complex)
        gamma = np.zeros(n_layers+1, dtype=complex)

    for i in range(n_layers + 1):
        # determine the local reflection
        r[i] = reflection_coefficient(n[i], n[i + 1], theta[i], theta[i + 1])

    # define the last global reflection coefficient as the local reflection
    # coefficient
    gamma[-1, :] = r[-1]

    # calculate global reflection coefficients recursively
    for i in range(n_layers - 1, -1, -1):
        # delta is Orfanidis eq. 8.1.2, with cosine
        delta = 2 * np.pi * freq / c * d[i] * n[i + 1] * np.cos(theta[i + 1])
        z = np.exp(-2j * delta)
        gamma[i, :] = (r[i]+gamma[i+1, :] * z) / (1+r[i] * gamma[i+1, :] * z)

    return gamma


def brute_force_search(freq_waveform, e0, freq, nr_array, ni_array, n_media, d,
                       theta0, start=0, stop=None, return_sum=False):
    """
    Function to perform a brute force search for the best index of refraction
    for the sample.
    :param freq_waveform: The frequency domain waveform from the area on the
        sample of interest.
    :param e0: The reference signal in the frequency domain
    :param freq: The frequency array over which to solve
    :param nr_array: The array of real index of refraction values to search over
    :param ni_array: The array of imaginary index of refraction values to search
        over. These should be negative
    :param n_media: an array that contains the index of refraction of the media
        on either side of the sample, with [0] containing the first media and
        [1] containing the media behind the sample
    :param d: The thickness of the sample in mm. If given as 0, will use FSE
        instead of BSE to calculate index of refraction
    :param theta0: The incoming angle of the THzBeam in radians.
    :param return_sum: Whether or not to return the sum over all frequencies at
        a given location. This would basically be solving for the single best
        index of refraction value over all frequencies. (Default: False)
    :return: The cost array. This is 5 dimensional. [y, x, nr, ni, freq]
    """

    # if the sample contains a lot of data points and the brute force search
    # grid is large, the cost array can actually take up more memory than
    # there is available RAM. In that case it may be necessary to only solve
    # for a single index of refraction value over all frequencies

    ncols = freq_waveform.shape[0]
    nrows = freq_waveform.shape[1]

    if stop is not None:
        freq = freq[start:stop]
        e0 = e0[start:stop]
        freq_waveform = freq_waveform[:, :, start:stop]

    if return_sum:
        size = (ncols, nrows, len(nr_array), len(ni_array))
    else:
        size = (ncols, nrows, len(nr_array), len(ni_array), len(freq))

    cost = np.zeros(size)

    for y in range(ncols):
        print('Row %d of %d' % (y+1, ncols))
        for x in range(nrows):
            # assume that user made freq_waveform from resized time domain
            # waveform?
            e2 = freq_waveform[y, x, :]
            for i, nr in enumerate(nr_array):
                for j, ni in enumerate(ni_array):
                    n = np.array([nr, ni])

                    raw_cost = half_space_mag_phase_equation(n, n_media, e0, e2,
                                                             freq, d, theta0)

                    if return_sum:
                        cost[y, x, i, j] = np.sum(raw_cost)
                    else:
                        cost[y, x, i, j, :] = raw_cost

    return cost


def parameter_gradient_descent(n0, n_media, e0, e2, theta0, d, freq, start=0,
                               stop=None, precision=1e-6, max_iter=1e4,
                               gamma=0.01):
    """
    Function to perform a gradient descent search on the cost function for
    material parameter estimation. The gradient descent algorithm is very
    similar to the one specified by Dorney et al. in reference [1].
    :param n0: The initial guess for the complex index of refraction. The
        imaginary part must be negative to cause extinction
    :param n_media: The indices of refraction of the media on either side of
        the sample material. [0] is the index of refraction of the front
        material and [1] is the index of refraction of the back material.
    :param e0: The reference waveform in the frequency domain
    :param e2: The sample waveform in the frequency domain
    :param theta0: The initial angle of the THz beam in radians
    :param d: The thickness of the sample in mm
    :param freq: An array of frequencies over which to calculate the complex
        index of refraction. The array that is returned is the same length as
        freq.
    :param start: The index of the frequency in the frequency array to start
        calculations at. If start is not zero, the points in the return array
        before this index will be (0 -j0)
    :param stop: The index of the frequency in the frequency array to end the
        calculations at.
    :param precision: The change is step size that will terminate the gradient
        descent. Default: 1e-6.
    :param max_iter: The maximum number of iterations at each frequency before
        the gradient descent is terminated. Default: 1e4
    :param gamma: What the error is multiplied by before it is added to the
        current best guess of the solution in the descent. Default: 0.01; This
        is the value that is suggested in [1].
    :return: n_array: The solution for the index of refraction at each
        frequency. n_array is the same length as freq.
    """

    # References
    # [1] "Material parameter estimation with terahertz time domain
    #     spectroscopy", Dorney et al, 2001.
    # [2] "Method for extraction of parameters in THz Time-Domain spectroscopy"
    #     Duvillaret et al, 1996

    if stop is None:
        stop = len(freq)

    n_array = np.zeros(stop, dtype=complex)

    for i in range(start, stop):
        n_sol = np.array([n_media[0], n0, n_media[1]])  # initial guess
        n_iter = 0
        n_step = 100  # reset steps to a large value so it won't stop right away
        k_step = 100
        while (n_step > precision or k_step > precision) and n_iter < max_iter:
            prev_n = n_sol[1].real
            prev_k = n_sol[1].imag

            theta1 = get_theta_out(n_sol[0], n_sol[1], theta0)
            model = half_space_model(e0[:stop], freq[:stop], n_sol, d,
                                     theta0, theta1)

            # transfer functions for the model and data
            T_model = model / e0[:stop]
            T_data = e2[:stop] / e0[:stop]

            # use the unwrapped phase so there are no discontinuities
            data_phase = np.unwrap(np.angle(T_data))
            model_phase = np.unwrap(np.angle(T_model))

            # start the DC phase at zero this is discussed in [1] & [2]
            data_phase -= data_phase[0]
            model_phase -= model_phase[0]

            # use absolute value because phase can be negative
            # this is the error function for phase and magnitude
            rho = (np.abs(data_phase[i]) - np.abs(model_phase)[i])
            phi = np.log(np.abs(T_data))[i] - np.log(np.abs(T_model))[i]

            # adjust the guess
            new_n = prev_n + gamma * rho
            new_k = prev_k + gamma * phi

            # determine how much it changes; when this value is less than
            # precision, loop will end
            n_step = np.abs(new_n - prev_n)
            k_step = np.abs(new_k - prev_k)

            n_sol[1] = complex(new_n, new_k)  # update n_sol

            n_iter += 1

        if n_iter == max_iter:
            print('Max iterations reached at frequency %0.3f!' % freq[i])

        n_array[i] = n_sol[1]  # store solution at that frequency

    return n_array


def global_gradient_descent(n0, n_media, e0, e2, theta0, d, freq, start=0,
                            stop=None, precision=1e-6, max_iter=1e4,
                            gamma=0.01):
    """
    Function to perform a gradient descent search on the cost function that is
    generated by the use of the global reflection model that is specified in
    Orfanidis.
    :param n0: The initial guess for the complex index of refraction. The
        imaginary part must be negative to cause extinction
    :param n_media: The indices of refraction of the media on either side of
        the sample material. [0] is the index of refraction of the front
        material and [1] is the index of refraction of the back material.
    :param e0: The reference waveform in the frequency domain
    :param e2: The sample waveform in the frequency domain
    :param theta0: The initial angle of the THz beam in radians
    :param d: The thickness of the sample in mm
    :param freq: An array of frequencies over which to calculate the complex
        index of refraction. The array that is returned is the same length as
        freq.
    :param start: The index of the frequency in the frequency array to start
        calculations at. If start is not zero, the points in the return array
        before this index will be (0 -j0)
    :param stop: The index of the frequency in the frequency array to end the
        calculations at.
    :param precision: The change is step size that will terminate the gradient
        descent. Default: 1e-6.
    :param max_iter: The maximum number of iterations at each frequency before
        the gradient descent is terminated. Default: 1e4
    :param gamma: What the error is multiplied by before it is added to the
        current best guess of the solution in the descent. Default: 0.01; This
        is the value that is suggested in [1].
    :return: n_array: The solution for the index of refraction at each
        frequency. n_array is the same length as freq.
    """
    # References
    # [1] "Material parameter estimation with terahertz time domain
    #     spectroscopy", Dorney et al, 2001.
    # [2] "Method for extraction of parameters in THz Time-Domain spectroscopy"
    #     Duvillaret et al, 1996

    if stop is None:
        stop = len(freq)

    n_array = np.zeros(stop, dtype=complex)

    for i in range(start, stop):
        n_sol = np.array([n_media[0], n0, n_media[1]])  # initial guess
        n_iter = 0
        n_step = 100  # reset steps to a large value so it won't stop right away
        k_step = 100
        while (n_step > precision or k_step > precision) and n_iter < max_iter:
            prev_n = n_sol[1].real
            prev_k = n_sol[1].imag

            theta1 = get_theta_out(n_sol[0], n_sol[1], theta0)
            theta = np.array([theta0, theta1, theta0])
            model = global_reflection_model(n_sol, theta, freq, d, 1)

            # transfer functions for the model and data
            T_model = model / e0[:stop]
            T_data = e2[:stop] / e0[:stop]

            # use the unwrapped phase so there are no discontinuities
            data_phase = np.unwrap(np.angle(T_data))
            model_phase = np.unwrap(np.angle(T_model))

            # start the DC phase at zero this is discussed in [1] & [2]
            data_phase -= data_phase[0]
            model_phase -= model_phase[0]

            # use absolute value because phase can be negative
            # this is the error function for phase and magnitude
            rho = (np.abs(data_phase[i]) - np.abs(model_phase)[0, i])
            phi = np.log(np.abs(T_data))[i] - np.log(np.abs(T_model))[0, i]

            # adjust the guess
            new_n = prev_n + gamma * rho
            new_k = prev_k + gamma * phi

            # determine how much it changes; when this value is less than
            # precision, loop will end
            n_step = np.abs(new_n - prev_n)
            k_step = np.abs(new_k - prev_k)

            n_sol[1] = complex(new_n, new_k)  # update n_sol

            n_iter += 1

        if n_iter == max_iter:
            print('Max iterations reached at frequency %0.3f!' % freq[i])

        n_array[i] = n_sol[1]  # store solution at that frequency

    return n_array


def scipy_optimize_parameters(data, n0, n_media, e0, d, stop_index):
    """
    Function that is a wrapper for scipy's optimization algorithm to determine
    the material parameters of the sample
    :param data: An instance of the THzProc Data class
    :param n0: the initial guess for the material's index of refraction.
        Expected to be an array that is [nr, ni].
    :param n_media: the index of refraction for the media on either side of the
        sample. [0] is the index of refraction for the media in front of the
        sample. [1] is the index of refraction for the media behind the sample
    :param e0: The reference waveform
    :param d: The thickness of the sample in mm
    :param stop_index: The index that corresponds to the highest frequency that
        we are looking to solve for
    :return: The value of n that best matches our model to the data across the
        sample
    """
    from scipy import optimize

    # optimize function does not like imaginary values, so put real and
    # imaginary part in an array
    n0 = np.array([n0.real, n0.imag])

    theta0 = data.theta0

    # for now let this function handle what happens if data has been resized
    if data.has_been_resized:
        y_step = data.y_step_small
        x_step = data.x_step_small
        freq_waveform = data.freq_waveform_small
    else:
        y_step = data.y_step
        x_step = data.x_step
        freq_waveform = data.freq_waveform

    n_array = np.zeros((y_step, x_step, stop_index), dtype=complex)

    for i in range(data.y_step):
        print('Row %d of %d' % (i+1, data.y_step))
        for j in range(data.x_step):
            e2 = freq_waveform[i, j, :]
            for k in range(stop_index):
                solution = \
                    optimize.fmin(half_space_mag_phase_equation, n0,
                                  args=(n_media, e0[:stop_index], e2[:stop_index],
                                        data.freq[:stop_index], d, theta0, k),
                                  disp=False)

                n_array[i, j, k] = complex(solution[0], solution[1])

    return n_array


def half_space_mag_phase_equation(n1, n_media, e0, e2, freq, d, theta0, start=0,
                                  stop=None, k=None, c=0.2998):
    """
    Function wrapper for the half space model that compares the magnitude and
    phase of the model that is derived from the reference signal (e0) to the
    experimental data (e2)
    :param n1: The index of refraction to build the model with. Expected to be
        a length 2 array [nr, ni].
    :param n_media: The index of refraction of the media in front of and behind
        the sample. Should be a length two array with [0] as the index of
        refraction of the 1st media and [1] as the index of refraction of the
        media behind the sample.
    :param e0: The reference signal in the frequency domain
    :param e2: The data signal in the frequency domain
    :param freq: The frequency array that define the frequency values in the
        calculations
    :param d: The thickness of the sample in mm
    :param theta0: The incoming angle of the THz Beam
    :param k: The index of the error term array to return. This allows this
        function to be called by another optimization function that is expecting
        a scalar return value, such as the ones in scipy's optimize package.
    :param c: The speed of light in mm/ps. Default: 0.2998
    """
    # scipy doesn't like to deal with complex values, so let n_in be a 2 element
    # array and then make it complex
    n_out = np.zeros(3, dtype=complex)
    n_out[0] = n_media[0]
    n_out[1] = n1[0] + 1j * n1[1]
    n_out[2] = n_media[1]

    # determine the angle in the material for the given index of refraction
    # and theta0
    theta1 = get_theta_out(n_out[0], n_out[1], theta0)

    # build the model
    model = half_space_model(e0, freq, n_out, d, theta0, theta1, c)

    T_model = model / e0
    T_data = e2 / e0

    # unwrap the phase so it is a continuous function
    # this makes for easier solving numerically
    model_phase = np.unwrap(np.angle(T_model))
    e2_phase = np.unwrap(np.angle(T_data))

    # set the DC phase to zero
    model_phase -= model_phase[0]
    e2_phase -= e2_phase[0]

    # add in the error function that is found in Duvillaret's 1996 paper [1]
    rho = np.log(np.abs(T_data)) - np.log(np.abs(T_model))
    phi = e2_phase - model_phase

    delta = rho**2 + phi**2

    if k is None:
        return delta
    else:
        return delta[k]


def half_space_model(e0, freq, n, d, theta0, theta1, c=0.2998):
    """
    Uses the half space model that does not include the Fabry-Perot effect.
    Creates the model signal by chaining together the Fresnel reflection
    coefficients, eg. E1 = E0*T01*R12*T10 + E0*T01*T12*R23*T21*T10*exp(...).
    Where exp(...) is the propagation factor.
    :param e0: The reference signal in the frequency domain
    :param freq: Array that contains the frequency values over which to build
        the model
    :param n: A length 3 complex array containing the index of refraction of
        the media before, the sample and the media after.
    :param theta0: Angle of the THz beam in radians
    :param theta1: Angle of the THz beam in the material
    :param c: The speed of light in mm/ps (Default: 0.2998)
    """
    # transmission from free space into sample
    t01 = transmission_coefficient(n[0], n[1], theta0, theta1)
    # transmission from sample into free space
    t10 = transmission_coefficient(n[1], n[0], theta1, theta0)

    # reflection at front surface
    r01 = reflection_coefficient(n[0], n[1], theta0, theta1)
    # reflection at back surface
    r10 = reflection_coefficient(n[1], n[2], theta1, theta0)

    # t_delay also includes imaginary n value, so signal should decrease
    # factor of two accounts for back and forth travel
    t_delay = 2 * n[1]*d / (c*np.cos(theta1))

    shift = np.exp(-1j * 2*np.pi * freq * t_delay)

    # if distance is given as 0, we just want to look at FSE reflection
    if d == 0:
        model = e0 * r01
    else:
        model = e0 * t01 * r10 * t10 * shift

    return model


def least_sq_wrapper(n_in, e0, e2, freq, d, theta0, c=0.2998):
    """
    Equation that is to be used with scipy's least_sq function to estimate a
    sample's material parameters numerically
    :param n_in: An array with two values, first value is the real part of the
        complex n, the second value is the imaginary part. Let the imaginary
        part be negative for extinction.
    :param e0: The reference signal in the frequency domain
    :param e2: The sample signal in the frequency domain
    :param freq: The frequency array over which to be solved
    :param d: The thickness of the sample in mm
    :param theta0: The initial angle of the THz beam in radians
    :param c: The speed of light in mm/ps (default: 0.2998)
    :return:
    """

    n = complex(n_in[0], n_in[1])

    # determine the ray angle of the THz beam in the material
    theta1 = get_theta_out(1.0, n, theta0)

    # build the model
    model = half_space_model(e0, freq, n, d, theta0, theta1, c)

    # create transfer function to try and remove system artifacts
    T_model = model / e0
    T_data = e2 / e0

    # unwrap the phase so it is a continuous function
    # this makes for easier solving numerically
    model_unwrapped_phase = np.unwrap(np.angle(T_model))
    e2_unwrapped_phase = np.unwrap(np.angle(T_data))

    model_mag = np.abs(T_model)
    e2_mag = np.abs(T_data)

    mag_array = np.log(e2_mag) - np.log(model_mag)
    phase_array = e2_unwrapped_phase - model_unwrapped_phase

    return_array = np.r_[mag_array, phase_array]

    return return_array


def brute_force2(E0, freq_waveform, n_media, nr_array, ni_array, freq, d,
                 theta0):
    """
    Function to create a cost function that compares the difference of the real
    and imaginary parts of the frequency domain waveforms that are generated
    by the model and the actual data
    :param E0: The reference waveform in the frequency domain
    :param freq_waveform: The THz waveform data in the frequency domain
    :param n: length 3 complex array that has the media of the front material
        in the fist spot and the backing material in the last spot
    :param nr_array: Array that contains values of the real part of the index
        of refraction to use to create the cost function
    :param ni_array: Array that contains the imaginary values over which to
        create the cost function
    :param freq: The frequency array from the scan
    :param d: The thickness of the layer in mm
    :param theta0: The initial angle of the THz system in radians
    """

    cost = np.zeros((len(nr_array), len(ni_array), len(freq)))

    n = np.array([n_media[0], 0, n_media[1]])

    for i, nr in enumerate(nr_array):
        for j, ni in enumerate(ni_array):
            n[1] = complex(nr, ni)
            theta1 = get_theta_out(n[0], n[1], theta0)
            model = half_space_model(E0, freq, n, d, theta0, theta1)

            cost[i, j, :] = np.abs(freq_waveform[:] - model)

    return cost
