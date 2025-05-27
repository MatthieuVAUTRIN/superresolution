import os

import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

from modules.handlers.multiprocessing import MultiprocessingHandler
from modules import filters
from modules import dsp


class Preprocessing:
    def __init__(
        self,
        params,
        # multiprocessing parameters
        MAX_WORKERS,
        frames_to_process,
        batch_size,
        # preprocessing parameters
        rf_matrix_path,
        using_simulation_data,
        using_matlab_data,
        convert_complex,
        interpolate_data,
        # filter parameters
        apply_hp,
        apply_hp_svd,
        apply_noise_profile,
        apply_directional_filter,
        directional_filter_direction,
        # saving parameters
        rf_data_path,
        folder_name,
        processed_rf_matrix_name,
    ):
        self.params = params
        # technical parameters
        self.MAX_WORKERS = MAX_WORKERS
        self.frames_to_process = frames_to_process
        self.batch_size = batch_size

        self.mp_handler = MultiprocessingHandler(
            self.MAX_WORKERS,
            self.frames_to_process,
            self.batch_size,
        )

        # saving parameters
        self.rf_data_path = rf_data_path
        self.folder_name = folder_name
        self.processed_rf_matrix_name = processed_rf_matrix_name

        # preprocessing parameters
        self.rf_matrix_path = rf_matrix_path
        self.using_simulation_data = using_simulation_data
        self.using_matlab_data = using_matlab_data
        self.convert_complex = convert_complex
        self.interpolate_data = interpolate_data

        self.rf_dtype, self.samples, self.sampling_rate = self._init_preprocessing()

        # filter parameters
        self.apply_hp = apply_hp
        self.apply_hp_svd = apply_hp_svd
        self.apply_noise_profile = apply_noise_profile
        self.apply_directional_filter = apply_directional_filter
        self.directional_filter_direction = directional_filter_direction

    def _init_preprocessing(self):

        if self.convert_complex:
            rf_dtype = np.complex64
        else:
            rf_dtype = np.float32

        if self.interpolate_data:
            samples = self.params["rf"]["upsampled_samples"]
            sampling_rate = self.params["rf"]["upsampled_sampling_rate"]
        else:
            samples = self.params["rf"]["samples"]
            sampling_rate = self.params["rf"]["sampling_rate"]

        # create numpy matrix for batch processing
        rf_matrix_preprocessed = np.lib.format.open_memmap(
            f"{self.rf_data_path}{self.folder_name}/{self.processed_rf_matrix_name}.npy",
            dtype=rf_dtype,
            mode="w+",
            shape=(
                samples,
                self.params["array"]["length"],
                len(self.params["rf"]["angles"]),
                self.frames_to_process,
            ),
        )
        print(
            f"Memmap file with shape: {rf_matrix_preprocessed.shape} saved at: {self.rf_data_path}{self.folder_name}/{self.processed_rf_matrix_name}.npy"
        )

        return rf_dtype, samples, sampling_rate

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
                f"\n{'='*60}\n"
            )

            actual_loaded_frames = rf_matrix.shape[3]

            if self.convert_complex:
                rf_matrix = dsp.get_iq(rf_matrix, axis=0)

            # Interpolate data
            if self.interpolate_data:
                rf_matrix_tmp = np.zeros(
                    (
                        self.params["rf"]["upsampled_samples"],
                        rf_matrix.shape[1],
                        rf_matrix.shape[2],
                        actual_loaded_frames,
                    ),
                    dtype=self.rf_dtype,
                )

                for j in range(actual_loaded_frames):

                    rf_matrix_tmp[:, :, :, j] = dsp.interpolate_rf(
                        rf_matrix[:, :, :, j],
                        self.params["rf"]["sampling_rate"],
                        self.params["rf"]["interpolation_factor"],
                        self.samples,
                    )

                rf_matrix = rf_matrix_tmp

                del rf_matrix_tmp

            if self.apply_hp:
                for i in range(0, len(self.params["rf"]["angles"])):
                    rf_matrix[:, :, i, :] = filters.filter_hp(
                        rf_matrix[:, :, i, :],
                        self.sampling_rate,
                        self.params["rf"]["frequency"],
                        1,
                        500,
                    )

            if self.apply_hp_svd:
                min_singular_value = 0
                for i in range(0, len(self.params["rf"]["angles"])):
                    rf_matrix[:, :, i, :], S = filters.filter_hp_svd(
                        rf_matrix[:, :, i, :],
                        self.sampling_rate,
                        self.params["rf"]["frequency"],
                        2,
                    )
                    kneedle = KneeLocator(
                        range(1, len(S) + 1), S, curve="convex", direction="decreasing"
                    )
                    min_singular_value += kneedle.elbow

                print(
                    f"Optimal number of singular values for SVD HPF: {min_singular_value / len(self.params['rf']['angles'])}"
                )

            if self.apply_noise_profile:
                for i in range(0, len(self.params["rf"]["angles"])):
                    rf_matrix[:, :, i, :], _ = filters.svd_noise_equalization_3d(
                        rf_matrix[:, :, i, :], window_length=128
                    )

            if self.apply_directional_filter:
                for i in range(0, len(self.params["rf"]["angles"])):

                    (
                        rf_matrix[:, :, i, : int(self.frames_to_process / 2)],
                        rf_matrix[:, :, i, int(self.frames_to_process / 2) :],
                    ) = filters.get_doppler_flow_images(
                        rf_matrix[:, :, i, :],
                        self.sampling_rate,
                        False,
                    )

            if actual_loaded_frames != current_batch_size:
                print(
                    f"Warning: Loaded {actual_loaded_frames} frames for batch {batch_idx}, expected {current_batch_size}."
                )
                current_batch_size = actual_loaded_frames

            output_folder = os.path.join(self.rf_data_path, self.folder_name)

            memmap_path = os.path.join(
                output_folder, f"{self.processed_rf_matrix_name}.npy"
            )
            memmap_rf_matrix = np.lib.format.open_memmap(
                memmap_path,
                dtype=self.rf_dtype,
                mode="r+",
                shape=(
                    self.samples,
                    self.params["array"]["length"],
                    len(self.params["rf"]["angles"]),
                    self.frames_to_process,
                ),
            )
            memmap_rf_matrix[
                :, :, :, start_frame : start_frame + actual_loaded_frames
            ] = rf_matrix
            memmap_rf_matrix.flush()  # making sure data is written to disk

            del memmap_rf_matrix  # force garbage collection
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

    def run_preprocessing(self):
        self.mp_handler.run_multiprocessing(self.process_batch_multiprocessing)
