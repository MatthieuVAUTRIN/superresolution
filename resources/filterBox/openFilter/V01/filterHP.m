
% high pass filter
% bff=filterHP(bf,fs,fcut,nrebound)
%  bf(nz,nx,nt)  3D matrix of beamformed images (complex)
%  fsam          sampling frequency
%  fcut          cutoff frequency
%  nrebound      number of eliminated images (use 0 if combine with SVD)

function bff=filterHP(bf,fsam,fcut,nrebound)

nt=size(bf,3);

bm=mean(bf,3);                       % substratc continius component
for i=1:nt                           % in order to eliminate frequency 0
    bf(:,:,i)=bf(:,:,i)-bm;
end

[B,A]=butter(3,fcut/fsam*2,'high');  % standard recursive high pass filter
bff=filter(B,A,bf,[],3);
bff(:,:,1:nrebound)=0;               % the first nelim images  are eliminated 
                                     % because the filter border effect.
