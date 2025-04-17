from scipy.io import savemat
import numpy as np


def convert_npy_to_mat(npy_file_path, mat_file_path, same_folder=True):
    data = np.load(npy_file_path)

    if same_folder:
        mat_file_path = npy_file_path.replace(".npy", ".mat")

    savemat(mat_file_path, {"data": data})
    print(f"Converted {npy_file_path} to {mat_file_path}")


if __name__ == "__main__":

    npy_folder_path = "/media/matmat/IZME/001_bf_rectangle.npy"
    mat_folder_path = "/home/matmat/Documents/urban_lab/002_Superresolution/superresolution/data_bf/pig_test/beamformed_apodized/0001_bf_bm_rectangle.mat"

    convert_npy_to_mat(npy_folder_path, mat_folder_path)
