function a=loadRaw(namefile)
    load([namefile '.mat'],'sizeData')
    fid=fopen([namefile '.dat'],'r+b');
    dat=fread(fid,'int16=>int16');
    fclose(fid);
    a=reshape(dat,sizeData);
    end