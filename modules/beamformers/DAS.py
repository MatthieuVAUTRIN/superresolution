import json
import timeit
import os
import math

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
        case "":
            return np.ones(array_length)
        case _:
            return np.ones(array_length)


class Das:
    def __init__(
        self,
        nsi_dc_offset,
        params,
        sampling_rate,
        samples,
        using_real_data,
        interpolate_data,
        bf_dtype,
        debug,
    ):
        self.nsi_dc_offset = nsi_dc_offset
        self.nx = params["image"]["nx"]
        self.nz = params["image"]["nz"]
        self.dx = params["image"]["dx"]
        self.dz = params["image"]["dz"]
        self.x0 = params["image"]["x0"]
        self.z0 = params["image"]["z0"]
        # mutliplying delay by interpolation rate to make sure its correct
        self.delay0 = (
            params["rf"]["delay0"] * params["rf"]["interpolation_factor"]
            if interpolate_data
            else params["rf"]["delay0"]
        )
        self.pitch = params["array"]["pitch"]
        self.c = params["rf"]["C"]
        self.array_length = params["array"]["length"]
        self.samples = samples
        self.sampling_rate = sampling_rate
        self.aperture = params["array"]["aperture"]
        self.using_real_data = using_real_data
        self.bf_dtype = bf_dtype
        self.debug = debug
        self.angles = params["rf"]["angles"]

    @staticmethod
    @njit(parallel=True)
    def _beamforming(
        rf,
        nx,
        nz,
        dx,
        dz,
        x0,
        z0,
        delay0,
        pitch,
        c,
        array_length,
        samples,
        sampling_rate,
        aperture,
        apodization_function,
        nsi_dc_offset,
        using_real_data,
        frames,
        bf_dtype,
        debug,
        angles,
    ):

        # var initrt
        fixed_delay = delay0 / c

        bf = np.zeros((nz, nx, len(angles), frames), dtype=bf_dtype)

        cos_angles = [np.cos(angle * np.pi / 180) for angle in angles]
        sin_angles = [np.sin(angle * np.pi / 180) for angle in angles]

        t0_delay = [fixed_delay for _ in range(len(angles))]

        if using_real_data:
            t0_delay = [
                t0_delay[i] + ((array_length * pitch) * sin_angle / c) * (angles[i] < 0)
                for i, sin_angle in enumerate(sin_angles)
            ]
            min_t0_delay = min(t0_delay)
            # t0_delay = [t0_delay[i] - min_t0_delay for i in range(len(t0_delay))]

        if debug:
            print("\n" + "=" * 60)
            print(f"✓ BEAMFORMING CONFIGURATION:")
            print(f"  • Resolution: {nx} x {nz}")
            print(f"  • Acquisition depth:", 2, "m", f"({nz} samples)")
            print(f"  • Fixed delay:", fixed_delay, "s", delay0, "m")
            print(f"  • Delays: ", t0_delay, " - Min delay: ", min_t0_delay, "s")
            print(
                f"  • Array length:",
                array_length * pitch,
                "m",
                f"({nx} samples)",
            )
            print(f"  • Apodization: {apodization_function}")
            print(f"  • Processing {len(angles)} angle(s)")
            print("-" * 60)

        for h in range(0, frames):  # for each frame
            if debug:
                print(f"  ↳ Processing frame", h + 1, f"[{h+1}/{frames}]")

            for i in range(0, len(angles)):  # for each angle
                if debug:
                    print(
                        f"    ↳ Processing angle",
                        angles[i],
                        "°",
                        f"[{i+1}/{len(angles)}]",
                    )

                for j in range(0, nx):  # for each lateral res sample

                    for k in range(0, nz):  # for vertical res sample

                        # account for intial depth / same goes for x0
                        depth = dz * k + z0
                        position = dx * j + x0

                        (
                            effective_array_length,
                            transducers_in_effective_array,
                            first_transducer,
                        ) = find_object_effective_aperture(
                            position, depth, aperture, array_length, pitch
                        )
                        apodization = apodization_wrapper(
                            apodization_function,
                            first_transducer,
                            transducers_in_effective_array,
                            effective_array_length,
                            array_length,
                            position,
                            0.5,
                            pitch,
                            nsi_dc_offset,
                        )

                        for l in range(
                            first_transducer,
                            first_transducer + transducers_in_effective_array,
                        ):  # for transducers in effective array

                            delay = (
                                (depth * cos_angles[i] + position * sin_angles[i])
                                + (np.sqrt(depth**2 + (l * pitch - position) ** 2))
                            ) / c - t0_delay[
                                i
                            ]  # delay in s

                            exact_delay_idx_float = delay * sampling_rate

                            # for better accuracr, lets apply linear interpolation on our delay values
                            idx_floor = int(np.floor(exact_delay_idx_float))  # x0
                            idx_ceil = idx_floor + 1  # x1

                            if 0 <= idx_floor and idx_ceil < samples:

                                weight_ceil = (
                                    exact_delay_idx_float - idx_floor
                                )  # wieght for y1
                                weight_floor = (
                                    idx_ceil - exact_delay_idx_float
                                )  # weight for y0

                                val_floor = rf[idx_floor, l, i, h]  # y0
                                val_ceil = rf[idx_ceil, l, i, h]  # y1

                                interpolated_value = (weight_floor * val_floor) + (
                                    weight_ceil * val_ceil
                                )  # y

                                bf[k, j, i, h] += interpolated_value * apodization[l]

        if debug:
            print("-" * 60)
            print(f"✓ BEAMFORMING COMPLETE")
            print(f"  • Output shape: ", bf.shape)
            print("=" * 60 + "\n")
        return bf

    def beamforming(self, rf, frames, apodization_function):
        return self._beamforming(
            rf,
            self.nx,
            self.nz,
            self.dx,
            self.dz,
            self.x0,
            self.z0,
            self.delay0,
            self.pitch,
            self.c,
            self.array_length,
            self.samples,
            self.sampling_rate,
            self.aperture,
            apodization_function,
            self.nsi_dc_offset,
            self.using_real_data,
            frames,
            self.bf_dtype,
            self.debug,
            self.angles,
        )
