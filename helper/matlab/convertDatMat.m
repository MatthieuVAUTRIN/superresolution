function convertDatMat()

    savePath = '/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/';
    folderRaw='/home/matmat/Documents/urban_lab/002_Superresolution/raw_data/acquisitionLong/beginning/';

    nb_angles = 16;

    dirData = dir([folderRaw, '/*.dat']);

    for i = 1:length(dirData)
        datName = dirData(i).name;

        disp(['Data path: ' folderRaw datName]);

        [D,frame]=loadDat([folderRaw datName]);

        shape = size(D);

        D = reshape(D, [shape(1), shape(2), nb_angles, shape(3)/nb_angles]);

        datName = erase(datName, '.dat');

        disp(['Data saved at: ' savePath datName '.mat'])

        save([savePath datName '.mat'], 'D');

    end
    