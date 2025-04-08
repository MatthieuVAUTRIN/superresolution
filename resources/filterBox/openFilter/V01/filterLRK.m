function a=filterLRK(bfr,nelim)

persistent G 

[nz,nx,nt]=size(bfr);

if(nargin==2)
  G=single(randn(nx*nz,nelim));
  return;
end


bf0=mean(bfr,3);
for it=1:nt
    bfr(:,:,it)=bfr(:,:,it)-bf0;
end

bfr=reshape(bfr,[nz*nx,nt])';
a=bfr-lowRankInternal(bfr,G);
a=a';
a=reshape(a,nz,nx,nt);
end

function [AL,X]=lowRankInternal(A,G)
[m,n]=size(A);
X=A*A';
H=X*(A*G);
[Q,R]=qr(H,0);
AL=Q*(Q'*A);
end 



