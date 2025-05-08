from scipy.signal import butter, lfilter, envelope, medfilt2d
import numpy as np
from scipy import ndimage


def filter_hp(bf, fsam, f, nrebound):
    # subtract continuous componen
    # bf = bf - np.mean(bf, axis=2, keepdims=True)
    # butterworth high-pass filter
    b, a = butter(3, f * 0.6, "highpass", output="ba", fs=fsam)
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

    return noise_field_smooth_image


def filter_hp_svd(bf, fsam, f, nelim):
    bff = filter_hp(bf, fsam, f, 0)
    bff, S = filter_svd(bff, nelim)
    return bff, S


def __calculate_axis_doppler_power(time_signal, sampling_rate, use_nsi):

    n = len(time_signal)
    time_signal_fft = np.fft.fft(time_signal)
    freq_f = np.fft.fftfreq(n, 1 / sampling_rate)
    # energy Spectral Density (ESD) |Z_F(ω)|^2
    esd = np.abs(time_signal_fft) ** 2

    positive_freq_mask = freq_f >= 0
    negative_freq_mask = freq_f < 0

    if use_nsi:
        # pos_flow = np.sum(envelope(esd[positive_freq_mask], residual=None))
        # neg_flow = np.sum(envelope(esd[negative_freq_mask], residual=None))
        pos_flow = envelope(esd[positive_freq_mask], residual=None)
        neg_flow = envelope(esd[negative_freq_mask], residual=None)
    else:
        # pos_flow = np.sum(esd[positive_freq_mask])
        # neg_flow = np.sum(esd[negative_freq_mask])
        pos_flow = esd[positive_freq_mask]
        neg_flow = esd[negative_freq_mask]

    return pos_flow, neg_flow


def get_doppler_flow_images(rf_matrix, sampling_rate, nsi: bool = False):

    num_depth, num_lateral, num_time = rf_matrix.shape
    # pos_flow_image = np.zeros((num_depth, num_lateral))
    # neg_flow_image = np.zeros((num_depth, num_lateral))
    pos_flow_image = np.zeros((num_depth, num_lateral, int(num_time / 2)))
    neg_flow_image = np.zeros((num_depth, num_lateral, int(num_time / 2)))

    for i in range(num_depth):  # depth
        for j in range(num_lateral):  # lateral

            time_signal = rf_matrix[i, j, :]
            pos_flow, neg_flow = __calculate_axis_doppler_power(
                time_signal, sampling_rate, use_nsi=nsi
            )

            # pos_flow_image[i, j] = pos_flow
            # neg_flow_image[i, j] = neg_flow
            pos_flow_image[i, j, :] = pos_flow
            neg_flow_image[i, j, :] = neg_flow

    return pos_flow_image, neg_flow_image


# phase difference for non nsi images
def filter_movement(rf_data, frames, direction="up"):
    filtered_rf_data = np.zeros_like(rf_data, dtype=np.complex64)

    instantaneous_phase = np.angle(rf_data)

    for i in range(0, rf_data.shape[1]):  # tranducers
        for j in range(0, rf_data.shape[0]):  # samples depth
            for k in range(1, frames):  # frames

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


# fgilter oevemnt by phase difference for nsi
def filter_movement_nsi(rf_data_zm, rf_data_dc1, rf_data_dc2, frames, direction="up"):

    rf_data = [rf_data_zm, rf_data_dc1, rf_data_dc2]

    filtered_rf_data_zm = np.zeros_like(rf_data_zm, dtype=np.complex64)
    filtered_rf_data_dc1 = np.zeros_like(rf_data_dc1, dtype=np.complex64)
    filtered_rf_data_dc2 = np.zeros_like(rf_data_dc2, dtype=np.complex64)

    filtered_rf_data = [filtered_rf_data_zm, filtered_rf_data_dc1, filtered_rf_data_dc2]

    instantaneous_phase_zm = np.angle(rf_data_zm)
    instantaneous_phase_dc1 = np.angle(rf_data_dc1)
    instantaneous_phase_dc2 = np.angle(rf_data_dc2)

    instantaneous_phase = [
        instantaneous_phase_zm,
        instantaneous_phase_dc1,
        instantaneous_phase_dc2,
    ]
    # instantaneous_phase = np.sum(instantaneous_phase, axis=0) / 3

    min_instantaneous_phase = float("inf")
    max_instantaneous_phase = float("-inf")

    for h in range(0, 3):
        for i in range(0, rf_data[0].shape[1]):  # tranducers
            for j in range(0, rf_data[0].shape[0]):  # samples depth
                for k in range(1, frames):  # frames
                    phase_shift = (
                        instantaneous_phase[h][j, i, k]
                        - instantaneous_phase[h][j, i, k - 1]
                    )
                    if phase_shift < min_instantaneous_phase:
                        min_instantaneous_phase = phase_shift
                    if phase_shift > max_instantaneous_phase:
                        max_instantaneous_phase = phase_shift
                    if direction == "up":
                        for l in range(len(rf_data)):
                            if phase_shift > 0:
                                filtered_rf_data[l][j, i, k] = rf_data[l][j, i, k]
                            else:
                                filtered_rf_data[l][j, i, k] = 0
                    elif direction == "down":
                        for l in range(len(rf_data)):
                            if phase_shift < -0:
                                filtered_rf_data[l][j, i, k] = rf_data[l][j, i, k]
                            else:
                                filtered_rf_data[l][j, i, k] = 0

    return (
        filtered_rf_data[0],
        filtered_rf_data[1],
        filtered_rf_data[2],
        min_instantaneous_phase,
        max_instantaneous_phase,
    )


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


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def svd_noise_equalization_3d(iq_data, N=1, window_length=15, visualize=False):
    """
    Perform SVD-based noise equalization for 3D ultrasound data.

    Parameters:
    -----------
    iq_data : 3D numpy array
        Ultrasound IQ data (complex values) with dimensions [depth, lateral, frames]
    N : int
        Number of smallest singular values to use for noise estimation
    window_length : int
        Length of the sliding window for smoothing
    visualize : bool
        If True, visualize the intermediate steps
    method : str
        'frame_by_frame': Apply SVD to each frame separately
        'volumetric': Reshape data to apply SVD to the entire volume

    Returns:
    --------
    equalized_data : 3D numpy array
        Noise-equalized ultrasound IQ data
    noise_profile : 3D numpy array
        The derived noise profile used for equalization
    """
    depth, lateral, frames = iq_data.shape
    equalized_data = np.zeros_like(iq_data, dtype=complex)

    # Reshape the 3D data to 2D for applying SVD to the entire volume
    # Combine depth and lateral dimensions
    reshaped_data = iq_data.reshape(depth * lateral, frames)
    # Apply SVD to the 2D representation of the volume
    U, s, Vh = np.linalg.svd(reshaped_data, full_matrices=False)
    # Create a noise field using only the N smallest singular values

    noise_field_2d = np.zeros_like(reshaped_data, dtype=complex)
    for i in range(len(s) - N, len(s)):
        noise_field_2d += s[i] * np.outer(U[:, i], Vh[i, :])

    # Reshape back to 3D
    noise_field = noise_field_2d.reshape(depth, lateral, frames)
    # Average the noise field along the lateral direction for each frame
    noise_profile = np.zeros_like(iq_data, dtype=float)
    for f in range(frames):
        frame_noise_field = noise_field[:, :, f]
        noise_avg = np.mean(np.abs(frame_noise_field), axis=1)
        # Smooth the averaged profile
        window = np.ones(window_length) / window_length
        smoothed_profile = signal.convolve(noise_avg, window, mode="same")
        # Replicate to match original frame dimensions
        noise_profile[:, :, f] = np.tile(smoothed_profile[:, np.newaxis], (1, lateral))
    if visualize:
        mid_frame = frames // 2
        plt.figure(figsize=(12, 10))
        plt.subplot(221)
        plt.imshow(np.abs(noise_field[:, :, mid_frame]), cmap="viridis")
        plt.title(f"(a) Derived noise field (N) for frame {mid_frame}")
        plt.colorbar()
        plt.subplot(222)
        plt.plot(np.mean(np.abs(noise_field[:, :, mid_frame]), axis=1))
        plt.title(f"(b) Averaged noise profile for frame {mid_frame}")
        plt.subplot(223)
        window = np.ones(window_length) / window_length
        smoothed = signal.convolve(
            np.mean(np.abs(noise_field[:, :, mid_frame]), axis=1),
            window,
            mode="same",
        )
        plt.plot(smoothed)
        plt.title(f"(c) Smoothed noise profile for frame {mid_frame}")
        plt.subplot(224)
        plt.imshow(noise_profile[:, :, mid_frame], cmap="viridis")
        plt.title(f"(d) Replicated noise profile for frame {mid_frame}")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # Apply equalization by dividing the original data by the noise profile
    epsilon = np.finfo(float).eps  # Avoid division by zero
    noise_profile = np.maximum(noise_profile, epsilon)

    # Element-wise division of complex data by real noise profile
    equalized_data = iq_data / noise_profile

    return equalized_data, noise_profile
