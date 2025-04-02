

function [bff,s]=filterSVD(bf,nelim)

if nelim==0
    bff=bf;
    return;
end

[nz,nx,nt]=size(bf);         % get dimensions of data nz=depth, nx=lateral, nt=time
m=reshape(bf,nz*nx,nt);      % convert in a 2D matrix space - time

for i=1:nz*nx
    m(i,:)=m(i,:)-mean(m(i,:));
end

[U,S,V]=svd(m,'econ');       % perform SVD decomposition
Sf=S;                        % Sf is the diagonal (singular values)
Sf(1:nelim,1:nelim)=0;       % the first nelim values are set to 0 (i.e. eliminated)
mf=U*Sf*V';                  % reconstruction of the filtered space - time matrix
bff=reshape(mf,nz,nx,nt);    % come back to a 3D matrix z-x-t --> output of the filter
s=diag(S);                   % singular values is an optional output. 
end