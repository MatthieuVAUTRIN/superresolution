from scipy.signal import butter, lfilter, envelope
import numpy as np


def filter_hp(bf, fsam, f, nrebound):
    # subtract continuous componen
    bf = bf - np.mean(bf, axis=2, keepdims=True)
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
    m = m - np.mean(m, axis=1, keepdims=True)
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
        pos_flow = np.sum(envelope(esd[positive_freq_mask], residual=None))
        neg_flow = np.sum(envelope(esd[negative_freq_mask], residual=None))
    else:
        pos_flow = np.sum(esd[positive_freq_mask])
        neg_flow = np.sum(esd[negative_freq_mask])

    return pos_flow, neg_flow


def get_doppler_flow_images(rf_matrix, sampling_rate, nsi: bool = False):

    num_depth, num_lateral, num_time = rf_matrix.shape
    pos_flow_image = np.zeros((num_depth, num_lateral))
    neg_flow_image = np.zeros((num_depth, num_lateral))

    for i in range(num_depth):  # depth
        for j in range(num_lateral):  # lateral

            time_signal = rf_matrix[i, j, :]
            pos_flow, neg_flow = __calculate_axis_doppler_power(
                time_signal, sampling_rate, use_nsi=nsi
            )

            pos_flow_image[i, j] = pos_flow
            neg_flow_image[i, j] = neg_flow

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
