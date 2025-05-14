import timeit

import numpy as np
import concurrent.futures
from scipy.io import loadmat


class MultiprocessingHandler:

    def __init__(
        self,
        # Technical parameters
        MAX_WORKERS,
        frames_to_process,
        batch_size,
    ):

        # technical parameters
        self.MAX_WORKERS = MAX_WORKERS
        self.frames_to_process = frames_to_process
        self.batch_size = batch_size

    def run_multiprocessing(self, func, cuda=False):

        if cuda:
            WORKERS = min(self.MAX_WORKERS, 1)
        else:
            WORKERS = self.MAX_WORKERS

        if self.frames_to_process % self.batch_size == 0:
            frame_iterations = self.frames_to_process // self.batch_size
        else:
            frame_iterations = self.frames_to_process // self.batch_size + 1

        # run multiprocessing
        start_time = timeit.default_timer()

        print(f"\nStarting process pool with {WORKERS} workers.")
        if cuda:
            print("CUDA enabled, processing on GPU.")
        print(f"Total frames to process: {self.frames_to_process}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total batches: {frame_iterations}")
        futures = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
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
                        func,
                        # batch args
                        batch_idx=batch_index,
                        start_frame=current_batch_start_frame,
                        current_batch_size=current_batch_actual_size,
                        frame_iterations=frame_iterations,
                        cuda=cuda,
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
        print(f"Total Batches: {frame_iterations}")
        print(f"Successful Batches: {num_success}")
        print(f"Failed Batches: {num_failed}")
        if num_failed == 0:
            print("\nAll batches processed successfully.")
        else:
            print(f"\n{num_failed} batch(es) failed. Check logs above.")
        print(f"\nTotal Beamforming took: {end_time - start_time:.2f} seconds")

    @staticmethod
    def load_rf_data(
        rf_matrix_path,
        desired_frames: list,
        using_simulation_data,
        using_matlab_data,
    ):
        debug_statement = ""

        if using_simulation_data:

            rf_matrix = np.load(rf_matrix_path)

            if rf_matrix.ndim == 3:
                rf_matrix = rf_matrix[:, :, :, np.newaxis]

            rf_matrix = rf_matrix[:, :, :, desired_frames[0] : desired_frames[1]]
            debug_statement += f"Imported RF matrix with shape:  {rf_matrix.shape}"

        else:
            if (
                using_matlab_data
            ):  # real data - matlab #!loads entire file high RAM usage
                rf_matrix = loadmat(rf_matrix_path)
                rf_matrix = list(rf_matrix.values())[-1]

                if rf_matrix.ndim == 3:
                    rf_matrix = rf_matrix[:, :, :, np.newaxis]

                rf_matrix = rf_matrix[:, :, :, desired_frames[0] : desired_frames[1]]
                rf_matrix = np.float32(rf_matrix)
                debug_statement += f"Imported RF matrix with shape: {rf_matrix.shape}"
            else:  # real data - numpy
                rf_matrix = np.lib.format.open_memmap(rf_matrix_path, mode="r")
                rf_matrix = rf_matrix[:, :, :, desired_frames[0] : desired_frames[1]]

                if rf_matrix.ndim == 3:
                    rf_matrix = rf_matrix[:, :, :, np.newaxis]

                debug_statement += f"Imported RF matrix with shape:  {rf_matrix.shape}"

        return rf_matrix, debug_statement
