from scipy.signal import butter, lfilter, envelope, medfilt2d
from scipy import signal
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from numba import njit


def filter_hp(bf, fsam, f, nrebound, fixed=0):
    # subtract continuous componen
    # bf = bf - np.mean(bf, axis=2, keepdims=True)
    # butterworth high-pass filter
    if fixed == 0:
        b, a = butter(3, f * 0.6, "highpass", output="ba", fs=fsam)
    else:
        b, a = butter(3, fixed, "highpass", output="ba", fs=fsam)
    # apply filter along the time dimension
    bff = lfilter(b, a, bf, axis=2)
    # the first nelim images are eliminated because the filter border effect
    if nrebound > 0:
        bff[:, :, 0:nrebound] = 0
    return bff


def filter_svd(bf, nelim):
    if nelim == 0:
        return bf, None
    # get dimensions of data
    nz, nx, nt = bf.shape
    # reshape to 2D matrix (space-time)
    m = np.reshape(bf, (nz * nx, nt))
    # remove mean from each row
    # m = m - np.mean(m, axis=1, keepdims=True)
    # perform SVD decomposition
    U, S, Vt = np.linalg.svd(m, full_matrices=False)
    # create diagonal matrix from singular values
    Sf = np.diag(S)
    # set smallest singular values to zero (eliminating noise)
    Sf[:nelim, :nelim] = 0

    # reconstruct filtered matrix
    mf = U @ Sf @ Vt
    # reshape back to 3D
    bff = np.reshape(mf, (nz, nx, nt))

    del U, Vt, m, mf, Sf
    # return filtered data and singular values
    return bff, S


def svd_noise_profile(bf, noise_component=1):
    # get dimensions of data
    nz, nx, nt = bf.shape
    # reshape to 2D matrix (space-time)
    m = np.reshape(bf, (nz * nx, nt))
    # remove mean from each row
    # m = m - np.mean(m, axis=1, keepdims=True)
    # perform SVD decomposition
    U, S, Vt = np.linalg.svd(m, full_matrices=False)
    # create diagonal matrix from singular values
    Sf = np.diag(S)
    # set hifhest singular values to zero (keeping noise)
    Sf[:-noise_component, :-noise_component] = 0
    # reconstruct filtered matrix
    mf = U @ Sf @ Vt
    # reshape back to 3D
    bff = np.reshape(mf, (nz, nx, nt))

    return bff, S


def get_noise_profile(bf):

    noise_field, _ = svd_noise_profile(bf)

    noise_field = np.abs(noise_field)

    noise_field_avg = np.mean(noise_field, axis=1)

    # noise_field_smooth = np.zeros_like(noise_field_avg)

    # for i in range(noise_field_avg.shape[1]):
    #     noise_field_smooth[:, i] = moving_average_filter(
    #         noise_field_avg[:, i], window_size=15
    #     )

    noise_field_smooth = medfilt2d(noise_field_avg, kernel_size=(15, 15))

    # noise_field_smooth = ndimage.uniform_filter(noise_field_avg, size=5, mode="nearest")

    # noise_field_smooth_norm = (noise_field_smooth - np.min(noise_field_smooth)) / (
    #     np.max(noise_field_smooth) - np.min(noise_field_smooth)
    # )

    noise_field_smooth_image = np.zeros_like(noise_field)

    for i in range(noise_field.shape[1]):
        noise_field_smooth_image[:, i, :] = (
            noise_field_smooth  # noise_field_smooth_norm
        )

    eps = np.finfo(float).eps  # avoid division by zero
    noise_field_smooth_image = np.maximum(noise_field_smooth_image, eps)

    return noise_field_smooth_image


def filter_hp_svd(bf, fsam, f, nelim):
    bff = filter_hp(bf, fsam, f, 0)
    bff, S = filter_svd(bff, nelim)
    return bff, S


def _calculate_axis_doppler_power(time_signal, sampling_rate):

    n = len(time_signal)
    time_signal_fft = np.fft.fft(time_signal)
    freq_f = np.fft.fftfreq(n, 1 / sampling_rate)
    # energy Spectral Density (ESD) |Z_F(ω)|^2
    esd = np.abs(time_signal_fft) ** 2

    positive_freq_mask = freq_f >= 0
    negative_freq_mask = freq_f < 0

    pos_flow = esd[positive_freq_mask]
    neg_flow = esd[negative_freq_mask]

    return pos_flow, neg_flow


def get_doppler_flow_images(rf_matrix, sampling_rate, pd_masking: bool = False):

    num_depth, num_lateral, num_time = rf_matrix.shape
    gray_flow_image = 0

    if pd_masking:
        pd_image = get_doppler_power_image(rf_matrix)
        pd_image = 20 * np.log10(pd_image) - np.max(20 * np.log10(pd_image))
        gray_flow_image = np.zeros((num_depth, num_lateral, num_time))
    # pos_flow_image = np.zeros((num_depth, num_lateral))
    # neg_flow_image = np.zeros((num_depth, num_lateral))
    pos_flow_image = np.zeros((num_depth, num_lateral, int(num_time / 2)))
    neg_flow_image = np.zeros((num_depth, num_lateral, int(num_time / 2)))

    for i in range(num_depth):  # depth
        for j in range(num_lateral):  # lateral
            if pd_masking:
                if pd_image[i, j] < -40:
                    gray_flow_image[i, j, :] = rf_matrix[i, j, :]
                    continue

            time_signal = rf_matrix[i, j, :]
            pos_flow, neg_flow = _calculate_axis_doppler_power(
                time_signal, sampling_rate
            )

            # pos_flow_image[i, j] = pos_flow
            # neg_flow_image[i, j] = neg_flow
            pos_flow_image[i, j, :] = pos_flow
            neg_flow_image[i, j, :] = neg_flow

    return pos_flow_image, neg_flow_image, gray_flow_image


@njit(parallel=True)
def get_doppler_power_image(rf_matrix):
    temporal_samples = rf_matrix.shape[2]
    power_doppler_image = np.zeros(
        (rf_matrix.shape[0], rf_matrix.shape[1]), dtype=rf_matrix.dtype
    )

    for i in range(temporal_samples):  # frames
        for j in range(rf_matrix.shape[0]):  # depth
            for k in range(rf_matrix.shape[1]):  # lateral
                power_doppler_image[j, k] += rf_matrix[j, k, i] ** 2
    return 1 / temporal_samples * power_doppler_image


# phase difference for non nsi images
def filter_movement(rf_data, direction="up"):
    filtered_rf_data = np.zeros_like(rf_data, dtype=np.complex64)

    instantaneous_phase = np.angle(rf_data)

    for i in range(0, rf_data.shape[1]):  # tranducers
        for j in range(0, rf_data.shape[0]):  # samples depth
            for k in range(1, rf_data.shape[2]):  # frames

                phase_shift = (
                    instantaneous_phase[j, i, k] - instantaneous_phase[j, i, k - 1]
                )
                if direction == "up":
                    filtered_rf_data[j, i, k] = (
                        0 if phase_shift > 0 else rf_data[j, i, k]
                    )
                elif direction == "down":
                    filtered_rf_data[j, i, k] = (
                        0 if phase_shift < 0 else rf_data[j, i, k]
                    )

    return filtered_rf_data


def moving_average_filter(rf_data, window_size):

    i = 0
    rf_smooth = np.zeros_like(rf_data, dtype=rf_data.dtype)

    while i < len(rf_data) - window_size + 1:

        window = rf_data[i : i + window_size]

        window_average = np.sum(window) / window_size

        # rf_smooth.append(window_average)
        rf_smooth[i] = window_average

        i += 1

    rf_smooth[-window_size + 1 :] = rf_smooth[-window_size]
    # fill the rest of the moving averages with the last value
    # for i in range(window_size - 1):
    #     moving_averages.append(moving_averages[-1])

    return rf_smooth
