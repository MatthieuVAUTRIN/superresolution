%exampleLoadRF
savePath = '/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/006_mouse_long/';
folderRaw='/home/matmat/Documents/urban_lab/002_Superresolution/raw_data/acquisitionLong/beginning/';

dirData = dir([folderRaw, '/*.dat']);

load([folderRaw, '/orderInfo']);

for i = 1:length(dirData)
    datName = dirData(i).name;

    disp(['Data path: ' folderRaw datName]);

    [D,frame]=loadDat([folderRaw datName]);

    rf = reorderRF(D, orderInfo);

    datName = erase(datName, '.dat');

    disp(['Data saved at: ' savePath datName '.mat'])

    save([savePath datName '.mat'], 'rf');

end


