
function saveAcquisitionData(acquisitionData, filePath, fileNamePrefix, paramPath)

    RF = acquisitionData.buffers.RF.D;

    save([filePath fileNamePrefix "_RF.mat"], 'RF');

    BM = acquisitionData.buffers.BM.D;

    save([filePath fileNamePrefix "_BM.mat"], 'BM');

    copyfile(paramPath, [filePath "params_" fileNamePrefix ".m"]);
