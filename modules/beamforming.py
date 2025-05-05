import json
import timeit
import os
import math

import numpy as np
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from numba import njit
from scipy.signal import envelope, firwin, hilbert
from scipy import fft
from scipy.io import loadmat

from modules.beamformer import Beamformer


class Beamforming(Beamformer):

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
            self.frame_iterations,
            self.samples_for_acquisition_depth,
            self.samples_for_array_length,
            self.apodization_functions,
        ) = self.__initialize_beamforming()

        super().__init__(
            nsi_dc_offset,
            params,
            self.sampling_rate,
            self.samples,
            not using_simulation_data,
            interpolate_data,
            self.bf_dtype,
            debug,
        )

    def load_rf_data(
        self,
        desired_frames: list,
    ):
        debug_statement = ""

        if self.using_simulation_data:

            rf_matrix_summed = np.load(self.rf_matrix_path)
            if rf_matrix_summed.ndim == 3:
                rf_matrix_summed = rf_matrix_summed[:, :, :, np.newaxis]
            rf_matrix_summed = rf_matrix_summed[
                :, :, :, desired_frames[0] : desired_frames[1]
            ]
            debug_statement += (
                f"Imported RF matrix with shape:  {rf_matrix_summed.shape}"
            )

        else:
            if self.using_matlab_data:
                rf_matrix_summed = loadmat(self.rf_matrix_path)
                rf_matrix_summed = list(rf_matrix_summed.values())[-1]

                if rf_matrix_summed.ndim == 3:
                    rf_matrix_summed = rf_matrix_summed[:, :, :, np.newaxis]
                rf_matrix_summed = rf_matrix_summed[
                    :, :, :, desired_frames[0] : desired_frames[1]
                ]
                rf_matrix_summed = np.float32(rf_matrix_summed)
                debug_statement += (
                    f"Imported RF matrix with shape: {rf_matrix_summed.shape}"
                )
            else:
                rf_matrix_summed = np.lib.format.open_memmap(
                    self.rf_matrix_path, mode="r"
                )

                rf_matrix_summed = rf_matrix_summed[
                    :, :, :, desired_frames[0] : desired_frames[1]
                ]
                if rf_matrix_summed.ndim == 3:
                    rf_matrix_summed = rf_matrix_summed[:, :, :, np.newaxis]
                debug_statement += (
                    f"Imported RF matrix with shape:  {rf_matrix_summed.shape}"
                )

        return rf_matrix_summed, debug_statement

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

    def __initialize_beamforming(self):
        # load right parameters for batch processing
        if self.frames_to_process % self.batch_size == 0:
            frame_iterations = self.frames_to_process // self.batch_size
        else:
            frame_iterations = self.frames_to_process // self.batch_size + 1

        print(f"Frame iterations: {frame_iterations}")
        print(f"Batch processing with {self.batch_size} frames")

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
            frame_iterations,
            samples_for_acquisition_depth,
            samples_for_array_length,
            apo_funcs,
        )

    def process_batch_multiprocessing(
        self,
        batch_idx,
        start_frame,
        current_batch_size,
    ):
        """
        Processes a single batch of RF data frames in a separate process.

        Returns:
            tuple: (batch_idx, True) if successful, (batch_idx, False) otherwise.
        """
        try:

            # Load rf data for the current batch

            rf_matrix_summed, debug_statement = self.load_rf_data(
                [start_frame, start_frame + current_batch_size],
            )

            print(
                f"{'='*60}\nProcessing batch {batch_idx} [{batch_idx}/{self.frame_iterations}] (Process ID: {os.getpid()})"
                f"\n  • Processing from frame {start_frame} to frame {start_frame + current_batch_size}"
                f"\n  • Processing {current_batch_size} frame(s)"
                f"\n  • RF matrix: {debug_statement}"
                f"\n  • Running {'NSI apodizations' if self.run_nsi_apodization else f'{self.other_apodization} apodization'}"
                f"\n{'='*60}\n"
            )

            actual_loaded_frames = rf_matrix_summed.shape[3]

            if actual_loaded_frames != current_batch_size:
                print(
                    f"Warning: Loaded {actual_loaded_frames} frames for batch {batch_idx}, expected {current_batch_size}."
                )
                current_batch_size = actual_loaded_frames

            # Beamforming
            bf_matrices = []

            for apo_func in self.apodization_functions:
                bf_apodized_matrix = super().beamforming(
                    rf_matrix_summed, actual_loaded_frames, apo_func
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
            del rf_matrix_summed

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

    def run_beamforming(self):
        # run multiprocessing
        start_time = timeit.default_timer()

        print(f"\nStarting process pool with {self.MAX_WORKERS} workers.")
        print(f"Total frames to process: {self.frames_to_process}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total batches: {self.frame_iterations}")
        futures = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.MAX_WORKERS
        ) as executor:
            last_batch_flag = False

            for i in range(0, self.frames_to_process, self.batch_size):
                if last_batch_flag:
                    break  # for safety
                current_batch_start_frame = i

                # last batch nb of frames if batch size isnt multiple of frames_to process - spot last batch
                if (
                    current_batch_start_frame + self.batch_size
                    >= self.frames_to_process
                ):
                    current_batch_actual_size = (
                        self.frames_to_process - current_batch_start_frame
                    )
                    last_batch_flag = True
                    print(
                        f"Last batch identified: Start Frame {current_batch_start_frame}, Size {current_batch_actual_size}"
                    )
                else:
                    current_batch_actual_size = self.batch_size

                    # set last batch if we have batch size multiple of frames to process
                    last_batch_flag = (
                        current_batch_start_frame + self.batch_size
                        == self.frames_to_process
                    )

                if current_batch_actual_size <= 0:
                    print(
                        f"Skipping batch starting at frame {current_batch_start_frame} due to zero or negative size."
                    )
                    continue

                batch_index = int(i / self.batch_size) + 1

                # submit task to process pool
                futures.append(
                    executor.submit(
                        self.process_batch_multiprocessing,
                        # batch args
                        batch_idx=batch_index,
                        start_frame=current_batch_start_frame,
                        current_batch_size=current_batch_actual_size,
                    )
                )

            # evluate all processes
            print("\nWaiting for all batches to complete...\n")
            results = []
            num_success = 0
            num_failed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_num, success = future.result()  # get function output
                    if success:
                        num_success += 1
                    else:
                        num_failed += 1
                        print(f"Batch {batch_num} failed.")
                    results.append(success)
                except Exception as exc:
                    print(
                        f"A batch generated an exception during execution/retrieval: {exc}"
                    )
                    num_failed += 1
                    results.append(False)

        # end of multiprocessing
        end_time = timeit.default_timer()

        print("\n--- Processing Summary ---\n")
        print(f"Total Batches: {self.frame_iterations}")
        print(f"Successful Batches: {num_success}")
        print(f"Failed Batches: {num_failed}")
        if num_failed == 0:
            print("\nAll batches processed successfully.")
        else:
            print(f"\n{num_failed} batch(es) failed. Check logs above.")
        print(f"\nTotal Beamforming took: {end_time - start_time:.2f} seconds")
