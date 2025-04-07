from scipy.io import savemat
import numpy as np


def convert_npy_to_mat(npy_file_path, mat_file_path):
    data = np.load(npy_file_path)

    savemat(mat_file_path, {"data": data})
    print(f"Converted {npy_file_path} to {mat_file_path}")


if __name__ == "__main__":

    npy_folder_path = "/home/matmat/Documents/urban_lab/002_Superresolution/superresolution/data_bf/npy/pig/beamformed_apodized/001_pig_rectangle.npy"
    mat_folder_path = "/home/matmat/Documents/urban_lab/002_Superresolution/superresolution/data_bf/npy/pig/beamformed_apodized/bf_pig_data.mat"

    convert_npy_to_mat(npy_folder_path, mat_folder_path)
