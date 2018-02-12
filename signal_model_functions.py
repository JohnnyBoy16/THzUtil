"""
Contains functions that are used when modeling a signal, such as reflection
and transmission coefficients.
"""

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

    if n2 == np.inf or n2 == -np.inf:
        return -1

    num = n1*np.cos(theta2) - n2*np.cos(theta1)
    denom = n1*np.cos(theta2) + n2*np.cos(theta1)

    return num / denom


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

    if n2 == np.inf or n2 == -np.inf:
        return 0

    num = 2 * n1 * np.cos(theta1)
    denom = n1*np.cos(theta2) + n2*np.cos(theta1)

    return num / denom


def get_theta_out(n0, n1, theta0):
    """
    Uses Snell's law to calculate the outgoing angle of light
    :param n0: The index of refraction of the incident media
    :param n1: The index of refraction of the outgoing media
    :param theta0: The angle of the incident ray in radians
    :return: theta1: The angle of the outgoing ray in radians
    """

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


def parameter_gradient_descent(n0, e0, e2, theta0, d, freq, start=0, stop=None,
                               precision=1e-6, max_iter=1e4, gamma=0.01):
    """
    Function to perform a gradient descent search on the cost function for
    material parameter estimation. The gradient descent algorith is very similar
    to the one specified by Dorney et al. in reference [1].
    :param n0: The initial guess for the complex index of refraction. The
        imaginary part must be negative to cause extinction
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
        n_sol = n0  # initial guess
        n_iter = 0
        n_step = 100  # reset steps to a large value so it won't stop right away
        k_step = 100
        while (n_step > precision or k_step > precision) and n_iter < max_iter:
            prev_n = n_sol.real
            prev_k = n_sol.imag

            theta1 = get_theta_out(1.0, n_sol, theta0)
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

            n_sol = complex(new_n, new_k)  # update n_sol

            n_iter += 1

        if n_iter == max_iter:
            print('Max iterations reached at frequency %0.3f!' % freq[i])

        n_array[i] = n_sol  # store solution at that frequency

    return n_array


def scipy_optimize_parameters(data, n0, e0, d, stop_index):
    """
    Function that is a wrapper for scipy's optimization algorithm to determine
    the material parameters of the sample
    :param data: An instance of the THzProc Data class
    :param n0: the initial guess for the material's index of refraction.
        Expected to be an array that is [nr, ni].
    :param e0: The reference waveform
    :param d: The thickness of the sample in mm
    :param stop_index: The index that corresponds to the highes frequency that
        we are looking to solve for
    :return: The value of n that best matches our model to the data across the
        sample
    """
    from scipy import optimize

    theta0 = data.theta0

    n_array = np.zeros((data.y_step, data.x_step, stop_index), dtype=complex)

    for i in range(data.y_step):
        print('Row %d of %d' % (i+1, data.y_step))
        for j in range(data.x_step):
            e2 = data.freq_waveform[i, j, :]
            for k in range(stop_index):
                solution = \
                    optimize.fmin(half_space_mag_phase_equation, n0,
                                  args=(e0[:stop_index], e2[:stop_index],
                                        data.freq[:stop_index], d, theta0, k),
                                  disp=False)

                n_array[i, j, k] = complex(solution[0], solution[1])

    return n_array


def brute_force_search(e0, e2, freq, nr_list, ni_list, d, theta0, step,
                       stop_index, lb, ub, c=0.2998):
    """
    Manually searches over the given range of real and imaginary index of
    refraction values to build a 2D cost map of the solution space
    :return: the cost map for the given nr & ni values
    """

    # the cost map over given nr & ni list
    cost = np.zeros((len(nr_list), len(ni_list), stop_index//step))

    for i, nr in enumerate(nr_list):
        for j, ni in enumerate(ni_list):
            m = 0  # use a different counter for cost array
            n = np.array([nr, ni])

            for k in range(step//2, stop_index, step):
                raw_cost = \
                    half_space_mag_phase_equation(n, e0[k-lb:k+ub],
                                                  e2[k-lb:k+ub],
                                                  freq[k-lb:k+ub], d, theta0, c)

                # try to emulate least squares
                cost[i, j, m] = np.sum(raw_cost)
                m += 1

    return cost


def half_space_mag_phase_equation(n_in, e0, e2, freq, d, theta0, k=None,
                                  c=0.2998):
    """
    Function wrapper for the half space model that compares the magnitude and
    phase of the model that is derived from the reference signal (e0) to the
    experimental data (e2)
    :param n_in: The index of refraction to build the model with. Expected to be
        a length 2 array [nr, ni].
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
    n_out = n_in[0] + 1j * n_in[1]

    # determine the angle in the material for the given index of refraction
    # and theta0
    theta1 = get_theta_out(1.0, n_out, theta0)

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
    :param n: The complex index of refraction of the sample
    :param theta0: Angle of the THz beam in radians
    :param theta1: Angle of the THz beam in the material
    :param c: The speed of light in mm/ps (Default: 0.2998)
    """
    # transmission from free space into sample
    t01 = transmission_coefficient(1.0, n, theta0, theta1)
    # transmission from sample into free space
    t10 = transmission_coefficient(n, 1.0, theta1, theta0)

    # reflection at front surface
    r01 = reflection_coefficient(1.0, n, theta0, theta1)
    # reflection at back surface
    r10 = reflection_coefficient(n, 1.0, theta1, theta0)

    # t_delay also includes imaginary n value, so signal should decrease
    # factor of two accounts for back and forth travel
    t_delay = 2 * n*d / (c*np.cos(theta1))

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
