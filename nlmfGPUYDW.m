function [nlmf] = nlmfGPUYDW(domain,DoS,k,npasses,gpuFlag)
%     [nx,ny,nz] = size(domain);
%     gpuDevice
    if gpuFlag
        DoS=gpuArray(single(DoS));
        domainG=gpuArray(single(domain));
        meanKernelG=gpuArray.ones(k,k,k,'single')./(k^3);
        neighbourKernel=gpuArray.zeros(k,k,k,(k^3),'single');
    else
        domainG=single(domain);
        meanKernelG=ones(k,k,k,'single')./(k^3);
        neighbourKernel=zeros(k,k,k,(k^3),'single');    
    end
    domainG=domainG./(max(domainG,[],'all'));
%     k=3;
    dl=floor(k/2);
    n=1;
    for i=1:k
        for j=1:k
            for k=1:k
                neighbourKernel(i,j,k,n)=1;
                n=n+1;
            end
        end
    end  
    for n=1:npasses
        disp(['Filter Pass: ', num2str(n)])
        disp(['Convolving Mean Filter'])

        meanG=convn(domainG,meanKernelG,'same');
        
        disp(['Convolving Neighbours'])

        meanNeighbours=convn(meanG,neighbourKernel,'full');
        meanNeighbours=meanNeighbours(1+dl:end-dl,1+dl:end-dl,1+dl:end-dl,:);

        domainNeighbours=convn(domainG,neighbourKernel,'full');
        domainNeighbours=domainNeighbours(1+dl:end-dl,1+dl:end-dl,1+dl:end-dl,:);

        disp(['Replicating Arrays'])

        domainG=repmat(domainG,[1,1,1,k^3]); % loop instead?
        meanG=repmat(meanG,[1,1,1,k^3]); % loop instead?
        
        disp(['Calculating Non-Local Mean'])

        expTerm=exp(-1.*(meanG-meanNeighbours).^2.*DoS);
        nlmfSums=sum(expTerm.*domainNeighbours,4);
        nlmfWeights=sum(expTerm,4);
        domainG=nlmfSums./nlmfWeights;

    end
    nlmf=domainG;
end