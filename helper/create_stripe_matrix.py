import numpy as np
from scipy.io import loadmat


def create_stripe_matrix(
    rf_path, output_path, stripe_width, matlab: bool, direction="horizontal"
):
    # creates horizontal stripes of the rf matrix
    #! make sure the stripe width is divisible by rf shape[1]

    if matlab:
        rf = loadmat(rf_path)
        rf = list(rf.values())[-1]
    else:
        rf = np.lib.format.open_memmap(rf_path, mode="r", dtype=np.float32)

    matrix_name = rf_path.split("/")[-1].split(".")[0]

    if direction == "horizontal":
        slice_shape = rf.shape[0]
        memmap_shape = (stripe_width, rf.shape[1], rf.shape[2], rf.shape[3])

    elif direction == "vertical":
        slice_shape = rf.shape[1]
        memmap_shape = (rf.shape[0], stripe_width, rf.shape[2], rf.shape[3])

    for i in range(0, int(slice_shape / stripe_width)):
        stripe = np.lib.format.open_memmap(
            output_path + f"{matrix_name}_stripe_{i}.npy",
            mode="w+",
            dtype=np.float32,
            shape=memmap_shape,
        )
        stripe[:, :, :, :] = (
            rf[i * stripe_width : (i + 1) * stripe_width, :, :, :]
            if direction == "horizontal"
            else rf[:, i * stripe_width : (i + 1) * stripe_width, :, :]
        )
        stripe.flush()
        print(f"Stripe {i} created at {output_path + matrix_name}_stripe_{i}.npy")

    print(f"Stripe matrix created at {output_path}")


if __name__ == "__main__":

    output_path = "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0002_stripes/"
    rf_path = "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/rf0002.mat"

    stripe_width = 256

    create_stripe_matrix(rf_path, output_path, stripe_width, matlab=True)
