import numpy as np
from scipy.signal import envelope


def get_envelope(self, bf, frames, axis=0):
    lateral_resolution = bf.shape[1]
    samples = bf.shape[0]

    env = np.zeros((samples, lateral_resolution, frames), dtype=bf.dtype)
    env = envelope(bf, residual=None, axis=axis)
    return env
