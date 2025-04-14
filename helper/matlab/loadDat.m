function [D,frame]=loadDat(namefile)
fid=fopen(namefile,'r+b');
frame=fread(fid,1,'int32');
nch=fread(fid,1,'*int32');
datatype=fread(fid,[1,nch],'*char');
ndims=fread(fid,1,'int32');
dims=fread(fid,ndims,'int32');
D=fread(fid,prod(dims),['*' datatype]);
D=reshape(D,dims');
fclose(fid);
end

