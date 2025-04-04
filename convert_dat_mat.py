import os


def convert_dat_to_mat(dat_folder_path, mat_folder_path):
    """
    Convert .dat files in a folder to .mat files and save them in another folder.

    Parameters:
    - dat_folder_path: Path to the folder containing .dat files.
    - mat_folder_path: Path to the folder where .mat files will be saved.
    """

    if not os.path.exists(mat_folder_path):
        os.makedirs(mat_folder_path)

    for filename in os.listdir(dat_folder_path):
        if filename.endswith(".dat"):
            dat_file_path = os.path.join(dat_folder_path, filename)
            mat_file_name = filename.replace(".dat", ".mat")
            mat_file_path = os.path.join(mat_folder_path, mat_file_name)

            os.rename(dat_file_path, mat_file_path)
            print(f"Converted {dat_file_path} to {mat_file_path}")


if __name__ == "__main__":

    dat_folder_path = (
        "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/rawdata/"
    )
    mat_folder_path = (
        "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/pig/"
    )

    convert_dat_to_mat(dat_folder_path, mat_folder_path)
