function c=correlation(b,a0, a1)
[nz,nx,nt]=size(b); 

a=zeros(nt,1); a(a0:a1)=1;

a=a-mean(a); 
a=a./sqrt(sum(a.^2));
 
c=zeros(nz,nx);
 
 for iz=1:nz
     for ix=1:nx
          s=squeeze(b(iz,ix,:));
          s=s-mean(s);
          s=s/sqrt(sum(s.^2));
          c(iz,ix)=sum(s.*a);
     end
 end

 