% some examples.

%load data example
load originalBF.mat

% only HP, in such case the cut-off and nrebound must be high
bff1=filterHP(bf,500,25,15);
a1=mean(abs(bff1).^2,3);
figure(1);
imagesc(a1.^0.25);

% only SVD
bff2=filterSVD(bf,25);
a2=mean(abs(bff2).^2,3);
figure(2);
imagesc(a2.^0.25);

% combination, use low values in HP filter
bff3=filterHP(bf,500,10,0);
bff3=filterSVD(bff3,25);
a3=mean(abs(bff3).^2,3);
figure(3);
imagesc(a3.^0.25);


% LRK is a fast implementation of the SVD filter with some approximations
% this filter has a very fast implementation in GPU then it is
% possible  to use it in real time. It is possible that it introduces some
% extra noise when the number of temporal samples is small

% you need an init step before use it.(here we init with the size of the bf
% matrix to filter and 25 SVD values to eliminate.
filterLRK(bf,25); 
                     
% now we can use the filter 
bff4=filterLRK(bf);
a4=mean(abs(bff4).^2,3); 
figure(4);
imagesc(a4.^0.25);


