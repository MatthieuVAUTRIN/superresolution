import json
import timeit
import os
import math

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import concurrent.futures
from numba import njit
from scipy.signal import envelope, firwin, hilbert
from scipy import fft
from scipy.io import loadmat

from modules.beamformer import Beamformer
from modules.beamformer_gpu import BeamformerGPU
from modules.multiprocessing import MultiprocessingHandler


class Beamforming(Beamformer, BeamformerGPU):

    def __init__(
        self,
        # beamformer parameters
        nsi_dc_offset,
        params,
        debug,
        angles_debug,
        # Technical parameters
        MAX_WORKERS,
        frames_to_process,
        batch_size,
        # beamforming parameters
        rf_matrix_path,
        using_simulation_data,
        using_matlab_data,
        convert_complex,
        interpolate_data,
        run_nsi_apodization,
        other_apodization,
        # saving parameters
        bf_data_path,
        folder_name,
        beamformed_apodized_matrix_name,
    ):
        self.params = params
        self.convert_complex = convert_complex
        self.interpolate_data = interpolate_data
        self.run_nsi_apodization = run_nsi_apodization
        self.other_apodization = other_apodization
        # technical parameters
        self.MAX_WORKERS = MAX_WORKERS
        self.frames_to_process = frames_to_process
        self.batch_size = batch_size

        self.mp_handler = MultiprocessingHandler(
            self.MAX_WORKERS,
            self.frames_to_process,
            self.batch_size,
        )

        # imports
        self.rf_matrix_path = rf_matrix_path
        self.using_simulation_data = using_simulation_data
        self.using_matlab_data = using_matlab_data
        # saving
        self.bf_data_path = bf_data_path
        self.folder_name = folder_name
        self.beamformed_apodized_matrix_name = beamformed_apodized_matrix_name

        self.angles_debug = angles_debug

        (
            self.samples,
            self.sampling_rate,
            self.bf_dtype,
            # self.frame_iterations,
            self.samples_for_acquisition_depth,
            self.samples_for_array_length,
            self.apodization_functions,
        ) = self._init_beamforming()

        Beamformer.__init__(
            self,
            nsi_dc_offset=nsi_dc_offset,
            params=params,
            sampling_rate=self.sampling_rate,
            samples=self.samples,
            using_real_data=not self.using_simulation_data,
            interpolate_data=interpolate_data,
            bf_dtype=self.bf_dtype,
            debug=debug,
        )
        BeamformerGPU.__init__(
            self,
            nsi_dc_offset=nsi_dc_offset,
            params=params,
            sampling_rate=self.sampling_rate,
            samples=self.samples,
            using_real_data=not self.using_simulation_data,
            interpolate_data=interpolate_data,
            bf_dtype=self.bf_dtype,
            debug=debug,
        )

    def __get_samples_depth_lateral_resolution(
        self, acq_depth, array_len, pitch, dx, dz
    ):
        samples_for_acquisition_depth = int(acq_depth / dz)
        samples_for_array_length = int(array_len * pitch / dx)

        return samples_for_acquisition_depth, samples_for_array_length

    @staticmethod
    def save_npy_data(
        data,
        filename,
        destination,  # {bf_data_path}/{folder_name}/beamformed_final/"
    ):
        np.save(f"{destination}{filename}", data)
        print(
            f"Beamformed data of shape {data.shape} saved at: {destination}{filename}.npy"
        )

    def _init_beamforming(self):

        if self.convert_complex:
            bf_dtype = np.complex64
        else:
            bf_dtype = np.float32

        if self.interpolate_data:
            samples = self.params["rf"]["upsampled_samples"]
            sampling_rate = self.params["rf"]["upsampled_sampling_rate"]
        else:
            samples = self.params["rf"]["samples"]
            sampling_rate = self.params["rf"]["sampling_rate"]

        apo_funcs = (
            ["zm", "dc1", "dc2"]
            if self.run_nsi_apodization
            else [self.other_apodization]
        )

        samples_for_acquisition_depth, samples_for_array_length = (
            self.__get_samples_depth_lateral_resolution(
                self.params["array"]["acq_depth"][1],
                self.params["array"]["length"],
                self.params["array"]["pitch"],
                self.params["image"]["dx"],
                self.params["image"]["dz"],
            )
        )

        if self.params["image"]["nx"]:
            samples_for_array_length = self.params["image"]["nx"]
        if self.params["image"]["nz"]:
            samples_for_acquisition_depth = self.params["image"]["nz"]

        print(f"Samples for acquisition depth: {samples_for_acquisition_depth}")
        print(f"Samples for array length: {samples_for_array_length}")

        # create numpy matrices for batch processing
        for apo_func in apo_funcs:
            apodized_bf_image = np.zeros(
                (
                    samples_for_acquisition_depth,
                    samples_for_array_length,
                    self.frames_to_process,
                ),
                dtype=bf_dtype,
            )
            self.save_npy_data(
                apodized_bf_image,
                f"{self.beamformed_apodized_matrix_name}_{apo_func}",  # file name
                destination=f"{self.bf_data_path}{self.folder_name}/beamformed_apodized/",  # saving folder path
            )
        return (
            samples,
            sampling_rate,
            bf_dtype,
            # frame_iterations,
            samples_for_acquisition_depth,
            samples_for_array_length,
            apo_funcs,
        )

    def process_batch_multiprocessing(
        self, batch_idx, start_frame, current_batch_size, frame_iterations, cuda
    ):
        """
        Processes a single batch of RF data frames in a separate process.

        Returns:
            tuple: (batch_idx, True) if successful, (batch_idx, False) otherwise.
        """
        try:

            # Load rf data for the current batch

            rf_matrix, debug_statement = self.mp_handler.load_rf_data(
                self.rf_matrix_path,
                [start_frame, start_frame + current_batch_size],
                self.using_simulation_data,
                self.using_matlab_data,
            )

            print(
                f"{'='*60}\nProcessing batch {batch_idx} [{batch_idx}/{frame_iterations}] (Process ID: {os.getpid()})"
                f"\n  • Processing from frame {start_frame} to frame {start_frame + current_batch_size}"
                f"\n  • Processing {current_batch_size} frame(s)"
                f"\n  • RF matrix: {debug_statement}"
                f"\n  • Running {'NSI apodizations' if self.run_nsi_apodization else f'{self.other_apodization} apodization'}"
                f"\n{'='*60}\n"
            )

            actual_loaded_frames = rf_matrix.shape[3]

            if actual_loaded_frames != current_batch_size:
                print(
                    f"Warning: Loaded {actual_loaded_frames} frames for batch {batch_idx}, expected {current_batch_size}."
                )
                current_batch_size = actual_loaded_frames

            # Beamforming
            bf_matrices = []

            for apo_func in self.apodization_functions:
                if cuda:
                    rf_matrix = rf_matrix.copy()
                    bf_apodized_matrix = self.beamforming_gpu(
                        rf_matrix, actual_loaded_frames, apo_func
                    )
                else:
                    bf_apodized_matrix = self.beamforming(
                        rf_matrix, actual_loaded_frames, apo_func
                    )
                bf_matrices.append(bf_apodized_matrix)

            if self.angles_debug and not self.run_nsi_apodization:
                self.save_npy_data(
                    bf_matrices[0],
                    f"{self.beamformed_apodized_matrix_name}_{self.apodization_functions[0]}_angles",
                    destination=f"{self.bf_data_path}{self.folder_name}/",
                )

            # summing all angles in bf matries
            bf_apodized_matrices_summed = [
                np.sum(bf_apodized_matrices_loaded, axis=2)
                for bf_apodized_matrices_loaded in bf_matrices
            ]

            # append to saved matrix using memory mapping
            output_folder = os.path.join(
                self.bf_data_path, self.folder_name, "beamformed_apodized"
            )

            for j, apo_func in enumerate(self.apodization_functions):
                memmap_path = os.path.join(
                    output_folder,
                    f"{self.beamformed_apodized_matrix_name}_{apo_func}.npy",
                )

                memmap_bf_matrix = np.lib.format.open_memmap(
                    memmap_path,
                    dtype=self.bf_dtype,
                    mode="r+",
                    shape=(
                        self.samples_for_acquisition_depth,
                        self.samples_for_array_length,
                        self.frames_to_process,
                    ),
                )
                memmap_bf_matrix[
                    :, :, start_frame : start_frame + actual_loaded_frames
                ] = bf_apodized_matrices_summed[j]
                memmap_bf_matrix.flush()  # making sure data is written to disk
                del memmap_bf_matrix

            del bf_apodized_matrices_summed
            del bf_matrices
            del rf_matrix

            print(
                "=" * 60
                + "\n"
                + f"Finished processing batch {batch_idx} (Process ID: {os.getpid()})\n"
                + "=" * 60
                + "\n"
            )
            return (batch_idx, True)  # return process status for easier debuging

        except Exception as e:
            print(
                f"!!! Error in process {os.getpid()} processing batch {batch_idx} (frames {start_frame} to {start_frame + current_batch_size}): {e}"
            )
            import traceback

            traceback.print_exc()
            return (batch_idx, False)

    def run_beamforming(self, cuda=False):
        self.mp_handler.run_multiprocessing(self.process_batch_multiprocessing, cuda)
