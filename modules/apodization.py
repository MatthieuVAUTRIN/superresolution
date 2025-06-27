import numpy as np
import numpy as np
from numba import njit


@njit()
def find_object_effective_aperture(
    current_position, current_depth, aperture, array_length, pitch
):
    effective_array_length = current_depth / aperture
    first_transducer = -1
    transducers_in_effective_array = 0
    for j in range(0, array_length):
        if (
            j * pitch >= current_position - effective_array_length / 2
            and j * pitch <= current_position + effective_array_length / 2
        ):
            if first_transducer == -1:
                first_transducer = j
            transducers_in_effective_array += 1
    return effective_array_length, transducers_in_effective_array, first_transducer


@njit()
def generate_zm_apodization(
    first_transducer,
    transducers_in_effective_array,
    array_length,
    offset=0,
    mirror=False,
):

    result = np.zeros(array_length) + offset

    for transducer_idx in range(array_length):
        if (
            transducer_idx - first_transducer >= 0
        ):  # making sure we are in the effective array range

            transducer_relative_idx = transducer_idx - first_transducer

            if 0 <= transducer_relative_idx < transducers_in_effective_array / 2:
                result[transducer_idx] = (-1 if mirror else 1) + offset
            elif (
                transducers_in_effective_array / 2
                <= transducer_relative_idx
                < transducers_in_effective_array
            ):
                result[transducer_idx] = (1 if mirror else -1) + offset

    return result


@njit()
def generate_rectangle_apodization(
    current_position,
    effective_array_length,
    array_length,
    amount_kept,
    pitch,
    offset=0,
):
    result = np.zeros(array_length) + offset
    half_array = effective_array_length / 2
    amount_trimmed = (1 - amount_kept) * half_array

    for transducer_idx in range(array_length):

        position = transducer_idx * pitch
        lower_bound = current_position - half_array + amount_trimmed
        upper_bound = current_position + half_array - amount_trimmed

        if lower_bound <= position <= upper_bound:
            result[transducer_idx] = 1 + offset
        else:
            result[transducer_idx] = 0 + offset

    return result


@njit()
def generate_hamming_apodization(
    current_position,
    effective_array_length,
    array_length,
    amount_kept,
    pitch,
    offset=0,
):
    result = np.zeros(array_length) + offset
    half_array = effective_array_length / 2
    amount_trimmed = (1 - amount_kept) * half_array
    eps = 1e-9  # to avoid division by zero

    for transducer_idx in range(array_length):
        position = transducer_idx * pitch
        lower_bound = current_position - half_array + amount_trimmed
        upper_bound = current_position + half_array - amount_trimmed

        if lower_bound <= position <= upper_bound:
            result[transducer_idx] = (
                0.54
                - 0.46
                * np.cos(
                    2
                    * np.pi
                    * (position - lower_bound)
                    / ((upper_bound - lower_bound) + eps)
                )
            ) + offset

        else:
            result[transducer_idx] = 0 + offset
    return result


@njit()
def apodization_wrapper(
    apodization_function,
    first_transducer,
    transducers_in_effective_array,
    effective_array_length,
    array_length,
    current_position,
    amount_kept,
    pitch,
    nsi_dc_offset,
    offset=0,
    mirror=False,
):
    match apodization_function:
        case "zm":
            return generate_zm_apodization(
                first_transducer,
                transducers_in_effective_array,
                array_length,
                offset=0,
                mirror=False,
            )
        case "dc1":
            return generate_zm_apodization(
                first_transducer,
                transducers_in_effective_array,
                array_length,
                offset=nsi_dc_offset,
                mirror=False,
            )
        case "dc2":
            return generate_zm_apodization(
                first_transducer,
                transducers_in_effective_array,
                array_length,
                offset=nsi_dc_offset,
                mirror=True,
            )
        case "rectangle":
            return generate_rectangle_apodization(
                current_position,
                effective_array_length,
                array_length,
                amount_kept,
                pitch,
                offset=0,
            )
        case "hamming":
            return generate_hamming_apodization(
                current_position,
                effective_array_length,
                array_length,
                amount_kept,
                pitch,
                offset=0,
            )
        case "":
            return np.ones(array_length)
        case _:
            return np.ones(array_length)
