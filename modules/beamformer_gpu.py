import json
import timeit
import os
import math

import numpy as np
import numpy as np
from numba import njit
from numba import cuda
from numba import float32


@cuda.jit(device=True)
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


@cuda.jit(device=True)
def generate_zm_apodization(
    apo_func,
    first_transducer,
    transducers_in_effective_array,
    array_length,
    offset,
    mirror,
):

    # result = np.zeros(array_length) + offset
    for i in range(array_length):
        apo_func[i] = offset

    for transducer_idx in range(array_length):
        if (
            transducer_idx - first_transducer >= 0
        ):  # making sure we are in the effective array range

            transducer_relative_idx = transducer_idx - first_transducer

            if 0 <= transducer_relative_idx < transducers_in_effective_array / 2:
                apo_func[transducer_idx] = (-1 if mirror else 1) + offset
            elif (
                transducers_in_effective_array / 2
                <= transducer_relative_idx
                < transducers_in_effective_array
            ):
                apo_func[transducer_idx] = (1 if mirror else -1) + offset

    # return apo_func


@cuda.jit(device=True)
def generate_rectangle_apodization(
    apo_func,
    current_position,
    effective_array_length,
    array_length,
    amount_kept,
    pitch,
    offset,
):
    # result = cuda.local.array(array_length, dtype=float32)

    for i in range(array_length):
        apo_func[i] = offset

    half_array = effective_array_length / 2
    amount_trimmed = (1 - amount_kept) * half_array

    for transducer_idx in range(array_length):

        position = transducer_idx * pitch
        lower_bound = current_position - half_array + amount_trimmed
        upper_bound = current_position + half_array - amount_trimmed

        if lower_bound <= position <= upper_bound:
            apo_func[transducer_idx] = 1 + offset

    # return apo_func


@cuda.jit(device=True)
def apodization_wrapper(
    apo_func,
    apo_name,
    first_transducer,
    transducers_in_effective_array,
    effective_array_length,
    array_length,
    current_position,
    amount_kept,
    pitch,
    nsi_dc_offset,
    offset,
):
    if apo_name == 231:
        generate_zm_apodization(
            apo_func,
            first_transducer,
            transducers_in_effective_array,
            array_length,
            0,
            False,
        )
    elif apo_name == 248:
        generate_zm_apodization(
            apo_func,
            first_transducer,
            transducers_in_effective_array,
            array_length,
            nsi_dc_offset,
            False,
        )
    elif apo_name == 249:
        generate_zm_apodization(
            apo_func,
            first_transducer,
            transducers_in_effective_array,
            array_length,
            nsi_dc_offset,
            True,
        )
    elif apo_name == 949:
        generate_rectangle_apodization(
            apo_func,
            current_position,
            effective_array_length,
            array_length,
            amount_kept,
            pitch,
            0,
        )


class BeamformerGPU:
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
    @cuda.jit
    def _beamforming_gpu(
        rf,
        bf,
        nx,
        nz,
        dx,
        dz,
        x0,
        z0,
        pitch,
        c,
        array_length,
        samples,
        sampling_rate,
        aperture,
        apodization_function,
        nsi_dc_offset,
        frames,
        angles,
        cos_angles,
        sin_angles,
        t0_delay,
    ):

        lat_res, ver_res, trans_idx = cuda.grid(3)

        for h in range(0, frames):  # for each frame

            for i in range(0, len(angles)):  # for each angle

                if lat_res < nx and ver_res < nz:

                    # account for intial depth / same goes for x0
                    depth = dz * ver_res + z0
                    position = dx * lat_res + x0

                    (
                        effective_array_length,
                        transducers_in_effective_array,
                        first_transducer,
                    ) = find_object_effective_aperture(
                        position, depth, aperture, array_length, pitch
                    )

                    if (
                        first_transducer
                        < trans_idx
                        < first_transducer + transducers_in_effective_array
                    ):

                        apodization = cuda.local.array(128, dtype=float32)

                        apodization_wrapper(
                            apodization,
                            apodization_function,
                            first_transducer,
                            transducers_in_effective_array,
                            effective_array_length,
                            array_length,
                            position,
                            0.5,
                            pitch,
                            nsi_dc_offset,
                            0,
                        )

                        delay = (
                            (depth * cos_angles[i] + position * sin_angles[i])
                            + (
                                math.sqrt(
                                    depth**2 + (trans_idx * pitch - position) ** 2
                                )
                            )
                        ) / c - t0_delay[
                            i
                        ]  # delay in s

                        exact_delay_idx_float = delay * sampling_rate

                        # for better accuracr, lets apply linear interpolation on our delay values
                        idx_floor = int(math.floor(exact_delay_idx_float))  # x0
                        idx_ceil = idx_floor + 1  # x1

                        if 0 <= idx_floor and idx_ceil < samples:

                            weight_ceil = (
                                exact_delay_idx_float - idx_floor
                            )  # wieght for y1
                            weight_floor = (
                                idx_ceil - exact_delay_idx_float
                            )  # weight for y0

                            val_floor = rf[idx_floor, trans_idx, i, h]  # y0
                            val_ceil = rf[idx_ceil, trans_idx, i, h]  # y1

                            interpolated_value = (weight_floor * val_floor) + (
                                weight_ceil * val_ceil
                            )  # y

                            value = interpolated_value * apodization[trans_idx]

                            cuda.atomic.add(
                                bf.real,
                                (ver_res, lat_res, i, h),
                                value.real,
                            )
                            cuda.atomic.add(
                                bf.imag,
                                (ver_res, lat_res, i, h),
                                value.imag,
                            )

    def beamforming_gpu(self, rf, frames, apodization_function):

        apo_int = [ord(c.lower()) for c in apodization_function]
        apo_int = np.int16(np.sum(apo_int))

        bf = np.zeros((self.nz, self.nx, len(self.angles), frames), dtype=self.bf_dtype)

        # define 3D grid dimensions
        threads_per_block = (8, 8, 8)
        blocks_x = (self.nx + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_y = (self.nz + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_z = (128 + threads_per_block[2] - 1) // threads_per_block[2]

        rf_device = cuda.to_device(rf)
        bf_device = cuda.to_device(bf)

        # init angles and delays
        fixed_delay = self.delay0 / self.c

        cos_angles = [np.cos(angle * np.pi / 180) for angle in self.angles]
        sin_angles = [np.sin(angle * np.pi / 180) for angle in self.angles]

        t0_delay = [fixed_delay for _ in range(len(self.angles))]

        if self.using_real_data:
            t0_delay = [
                t0_delay[i]
                + ((self.array_length * self.pitch) * sin_angle / self.c)
                * (self.angles[i] < 0)
                for i, sin_angle in enumerate(sin_angles)
            ]

        angles_device = cuda.to_device(np.array(self.angles, dtype=np.float32))
        cos_angles_device = cuda.to_device(np.array(cos_angles, dtype=np.float32))
        sin_angles_device = cuda.to_device(np.array(sin_angles, dtype=np.float32))
        t0_delay_device = cuda.to_device(np.array(t0_delay, dtype=np.float32))

        self._beamforming_gpu[(blocks_x, blocks_y, blocks_z), threads_per_block](
            rf_device,
            bf_device,
            self.nx,
            self.nz,
            self.dx,
            self.dz,
            self.x0,
            self.z0,
            self.pitch,
            self.c,
            self.array_length,
            self.samples,
            self.sampling_rate,
            self.aperture,
            apo_int,
            self.nsi_dc_offset,
            frames,
            angles_device,
            cos_angles_device,
            sin_angles_device,
            t0_delay_device,
        )

        return bf_device.copy_to_host()
