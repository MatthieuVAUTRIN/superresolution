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
