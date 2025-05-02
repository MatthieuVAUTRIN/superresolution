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


def filter_movement_phase(bf, frames, direction="up"):

    bff = np.zeros_like(bf, dtype=np.complex64)
    instantaneous_phase = np.angle(bf)

    for h in range(0, bf.shape[2]):  # angles
        for i in range(0, bf.shape[1]):  # tranducers
            for j in range(0, bf.shape[0]):  # samples depth
                for k in range(1, frames):  # frames

                    phase_shift = (
                        instantaneous_phase[j, i, h, k]
                        - instantaneous_phase[j, i, h, k - 1]
                    )

                    if direction == "up":
                        bff[j, i, h, k] = 0 if phase_shift > 0 else bf[j, i, h, k]
                    elif direction == "down":
                        bff[j, i, h, k] = 0 if phase_shift < 0 else bf[j, i, h, k]
    return bff


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


def get_doppler_flow_images(rf_matrix, sampling_rate):

    num_depth, num_lateral, num_time = rf_matrix.shape
    pos_flow_image = np.zeros((num_depth, num_lateral))
    neg_flow_image = np.zeros((num_depth, num_lateral))

    for i in range(num_depth):  # depth
        for j in range(num_lateral):  # lateral

            time_signal = rf_matrix[i, j, :]
            pos_flow, neg_flow = __calculate_axis_doppler_power(
                time_signal, sampling_rate
            )

            pos_flow_image[i, j] = pos_flow
            neg_flow_image[i, j] = neg_flow

    return pos_flow_image, neg_flow_image
