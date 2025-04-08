import numpy as np


def are_bf_images_equal(path_image1, path_image2):

    image1 = np.load(path_image1)
    image2 = np.load(path_image2)

    if image1.shape != image2.shape:
        return False

    return np.array_equal(image1, image2)


if __name__ == "__main__":

    path_image1 = "/home/matmat/Documents/urban_lab/002_Superresolution/superresolution/data_bf/npy/pig/beamformed_final/pig_nsi_15frames_multiprocess_env_nsi.npy"
    path_image2 = "/home/matmat/Documents/urban_lab/002_Superresolution/superresolution/data_bf/npy/pig/beamformed_final/pig_nsi_15frames_env_nsi.npy"

    result = are_bf_images_equal(path_image1, path_image2)

    print(f"Are the BF images equal? {result}")
