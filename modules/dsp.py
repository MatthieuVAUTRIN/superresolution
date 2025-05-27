import numpy as np
from scipy.signal import envelope, hilbert, firwin
from numba import njit


def get_envelope(bf, axis=0, iq: bool = False):
    env = np.zeros_like(bf, dtype=bf.dtype)

    if iq:
        env = np.abs(bf)
    else:
        env = envelope(bf, residual=None, axis=axis)
    return env


def get_iq(rf_data, axis=0):
    complex_rf_data = np.zeros_like(rf_data, dtype=np.complex64)

    complex_rf_data = hilbert(rf_data, axis=axis)

    return complex_rf_data


@njit()
def _zero_stuffing(rf_data, interpolation_factor, upsampled_samples):
    interpolation_factor = interpolation_factor - 1

    rf_stuffed = np.zeros(
        (upsampled_samples, rf_data.shape[1], rf_data.shape[2]),
        dtype=rf_data.dtype,
    )

    for i in range(rf_data.shape[2]):  # for each angle
        for j in range(rf_data.shape[1]):  # for each transducer
            for k in range(rf_data.shape[0]):  # for each sample
                idx_new = k + (interpolation_factor * k)
                rf_stuffed[idx_new, j, i] = rf_data[k, j, i]

    return rf_stuffed


def _get_filtered_matrices(rf_stuffed, original_fsam, interpolation_factor):
    interpolated_rf = np.zeros(
        (
            rf_stuffed.shape[0],
            rf_stuffed.shape[1],
            rf_stuffed.shape[2],
        ),
        dtype=rf_stuffed.dtype,
    )

    n_taps = 129
    cutoff_freq = original_fsam / 2

    filtering_function = (
        firwin(
            n_taps,
            cutoff_freq,
            fs=original_fsam * interpolation_factor,
            window="hamming",
        )
        * interpolation_factor
    )

    for i in range(0, rf_stuffed.shape[2]):  # for each angle
        for j in range(0, rf_stuffed.shape[1]):  # for each transducer
            interpolated_rf[:, j, i] = np.convolve(
                rf_stuffed[:, j, i],
                filtering_function,
                mode="same",
            )

    return interpolated_rf


def interpolate_rf(
    rf_matrix,
    original_fsam,
    interpolation_factor,
    upsampled_samples,
):
    zero_stuffed_rf = _zero_stuffing(rf_matrix, interpolation_factor, upsampled_samples)

    filtered_rf_matrix = _get_filtered_matrices(
        zero_stuffed_rf, original_fsam, interpolation_factor
    )

    return filtered_rf_matrix


@njit
def bit_reverse(n, bits):
    result = 0
    for i in range(bits):
        if n & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


@njit(cache=True, fastmath=True)
def fft(x):
    n = len(x)
    # Check if n is a power of 2
    if n & (n - 1) != 0:
        raise ValueError("Size of input must be a power of 2")

    # Calculate number of bits needed to represent n
    bits = 0
    temp_n = n
    while temp_n > 1:
        temp_n >>= 1
        bits += 1

    # Create output array
    result = np.zeros(n, dtype=np.complex64)

    # Bit reversal permutation
    for i in range(n):
        reversed_i = bit_reverse(i, bits)
        if i <= reversed_i:
            result[i] = x[reversed_i]
            result[reversed_i] = x[i]

    # Butterfly computations
    for stage in range(1, bits + 1):
        m = 1 << stage  # 2^stage
        half_m = m >> 1  # m/2

        for k in range(0, n, m):
            for j in range(half_m):
                # Calculate twiddle factor
                omega = np.exp(-2j * np.pi * j / m)

                # Apply butterfly operation
                t = omega * result[k + j + half_m]
                u = result[k + j]

                # Store results
                result[k + j] = u + t
                result[k + j + half_m] = u - t

    return result
