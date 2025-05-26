import numpy as np
from scipy.io import loadmat

import os
import concurrent.futures

from convert_npy_mat import convert_npy_to_mat


def memory_concat(
    path, path_idx, file_frames, mat_dtype, mat_shape, output_matrix_name, output_folder
):
    try:
        output_path = os.path.join(output_folder, output_matrix_name)

        memmap = np.lib.format.open_memmap(
            output_path + ".npy",
            dtype=mat_dtype,
            mode="r+",
            shape=mat_shape,
        )

        curr_mat = loadmat(path)
        curr_mat = list(curr_mat.values())[-1]

        print(f"=" * 60)
        print(f"Processing file {path_idx + 1} - PID: {os.getpid()}")
        print(f"Current matrix shape: {curr_mat.shape}")
        print(
            f"Concatenating from {path_idx * file_frames} to {(path_idx + 1) * file_frames}"
        )
        print(f"=" * 60)

        memmap[:, :, :, path_idx * file_frames : (path_idx + 1) * file_frames] = (
            curr_mat
        )

        memmap.flush()
        del memmap
        del curr_mat

        print(
            "=" * 60
            + "\n"
            + f"Finished processing path {path_idx} (Process ID: {os.getpid()})\n"
            + "=" * 60
            + "\n"
        )
        return True

    except Exception as e:

        print(f"Error processing in PID - {os.getpid()}, file {path}: {e}")
        return False


if __name__ == "__main__":

    WORKERS = 8

    path_list = [
        "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0001.mat",
        "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0002.mat",
        "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0003.mat",
        "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0004.mat",
        "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0005.mat",
        "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0006.mat",
        "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0007.mat",
        "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0008.mat",
        # "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0009.mat",
        # "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0010.mat",
        # "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0011.mat",
    ]
    output_matrix_name = "1280_concat_rf"
    output_folder = "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/concat/"

    output_path = os.path.join(output_folder, output_matrix_name)

    # save first matrix for memmap to operate
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping concatenation!")
        exit(0)

    rf_matrix = loadmat(path_list[0])
    rf_matrix = list(rf_matrix.values())[-1]

    matrix_shape = rf_matrix.shape
    frames = matrix_shape[3]
    matrix_dtype = rf_matrix.dtype

    tmp_matrix = np.zeros(
        (
            matrix_shape[0],  # depth
            matrix_shape[1],  # lat distance
            matrix_shape[2],  # angles
            len(path_list) * frames,  # frames
        ),
        dtype=matrix_dtype,
    )

    output_rf_matrix_shape = tmp_matrix.shape

    np.save(output_path + ".npy", tmp_matrix)

    print(
        f"Output file of shape, {tmp_matrix.shape} at {output_path} created. Starting concatenation..."
    )

    del tmp_matrix, rf_matrix

    futures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for idx, path in enumerate(path_list):
            futures.append(
                executor.submit(
                    memory_concat,
                    path=path,
                    path_idx=idx,
                    file_frames=frames,
                    mat_dtype=matrix_dtype,
                    mat_shape=output_rf_matrix_shape,
                    output_matrix_name=output_matrix_name,
                    output_folder=output_folder,
                )
            )

    nb_success = 0
    nb_fail = 0

    for future in concurrent.futures.as_completed(futures):
        try:
            success = future.result()
            if success:
                nb_success += 1
            else:
                nb_fail += 1

        except Exception as e:
            print(f"Error in future: {e}")

    print("\n--- Processing Summary ---")
    print(f"Total Paths: {len(path_list)}")
    print(f"Successful Paths: {nb_success}")
    print(f"Failed Paths: {nb_fail}")
    if nb_fail == 0:
        print("\nAll batches processed successfully.")
    print("\n")
    # memory_concat(path_list, output_matrix_name, output_folder)

    # convert_npy_to_mat(output_path + ".npy", "", same_folder=True)
    # print(f"Concatenation completed. Output saved to {output_path}")
