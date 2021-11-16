function [nlmf] = serialGPUNLMF(domain,locSize,DoS,k,npasses,gpuFlag)
    Nx=size(domain,1);
    Ny=size(domain,2);
    Nz=size(domain,3);
    if gpuFlag
%         nlmf=gpuArray.zeros(size(domain),'single');
        gpuDevice(1)
    end
    nlmf=zeros(size(domain),'single');
    dL=floor(k/2);
    kernelSize=k;
    npx=ceil(Nx/locSize);
    npy=ceil(Ny/locSize);
    npz=ceil(Nz/locSize);
    n=0;
    tic
    for m=1:npasses
    for i=1:npx
        for j=1:npy
            for k=1:npz

                ox=(i-1)*locSize+1;
                oy=(j-1)*locSize+1;
                oz=(k-1)*locSize+1;
%                 %use original location
%                 TX=[ox-dL:ox+locSize+dL-1];
%                 TY=[oy-dL:oy+locSize+dL-1];
%                 TZ=[oz-dL:oz+locSize+dL-1];

                DX=[max(ox-dL,1):min(ox+locSize+dL-1,Nx)];
                DY=[max(oy-dL,1):min(oy+locSize+dL-1,Ny)];
                DZ=[max(oz-dL,1):min(oz+locSize+dL-1,Nz)];
                borderFlags=double(~[ox-dL<1, ox+locSize+dL-1>Nx, oy-dL<1, oy+locSize+dL-1>Ny, oz-dL<1, oz+locSize+dL-1>Nz]);
                boundBoxD=domain(DX,DY,DZ);

                locnlmf=nlmfGPUYDW(boundBoxD,DoS,kernelSize,1,gpuFlag);
                if gpuFlag
                    locnlmf=gather(locnlmf);
                end
                nlmf(DX(1+borderFlags(1)+dL-1:end-borderFlags(2)-dL+1),DY(1+borderFlags(3)+dL-1:end-borderFlags(4)-dL+1),DZ(1+borderFlags(5)+dL-1:end-borderFlags(6)-dL+1))=locnlmf(1+borderFlags(1)+dL-1:end-borderFlags(2)-dL+1,1+borderFlags(3)+dL-1:end-borderFlags(4)-dL+1,1+borderFlags(5)+dL-1:end-borderFlags(6)-dL+1);
                n=n+1;
                disp(['Pass: ', num2str(m), ', ', num2str(n), ' of ', num2str(npx*npy*npz), ' time: ', num2str(toc)])
            end
        end
    end
    end

end