import numpy as np
from scipy.signal import envelope, hilbert


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
