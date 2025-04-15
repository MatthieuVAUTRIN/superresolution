function rf = reorderRF(rfmix, order)


rfmix=reshape(rfmix,[order.nchmod, order.nSamples, order.nmod, order.nAngles ,order.nImagesBurst]);

rf=zeros(order.nSamples,order.nmod*order.nchmod,order.nAngles,order.nImagesBurst, 'int16');

for iAngle=1:order.nAngles
    for iImage=1:order.nImagesBurst
        rf2=reorder1(rfmix(:,:,:,iAngle,iImage),order.nSamples,order.nmod,order.nchmod);
        rf(:,:,iAngle,iImage)=rf2(:,order.pin2channel);
    end
end
end

function rf=reorder1(rfmix,ndat,nmod,nchmod)

rf=zeros(ndat,nmod*nchmod,'int16');
for imod=1:nmod
    for it=1:ndat
        for ichmod=1:nchmod
            rf(it,(imod-1)*nchmod+ichmod)=rfmix(ichmod,it,imod);
        end
    end
end

end