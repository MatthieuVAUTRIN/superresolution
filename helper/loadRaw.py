import scipy.io
import numpy as np
import matplotlib.pyplot as plt


path = (
    "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data/eloHelp/raw_144343"
)

mat = scipy.io.loadmat(path + ".mat")
sizeData = mat["sizeData"].flatten()

with open(path + ".dat", "rb") as fid:
    dat = np.fromfile(fid)
# fid.close()
a = np.reshape(dat, sizeData)
plt.imshow(a[:, :, 999], cmap="gray")
