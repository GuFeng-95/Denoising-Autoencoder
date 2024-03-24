clc, clear, close all

%%
%% DATA
n=1:3;
an=0.03./(1+log(n));
bn=0.05./(1+log(n));
betan=0.8./(1+log(n));
gamman=0.8+n-1;
alfan=ones(size(n)); 
SNR=1.25;
fsamp=1/0.3;
tvec=0:(1/fsamp):10;


N=numel(n);

addpath('.\functions\')
%%
%% creating MESH
dx=0.01;
y=-0.64:dx:0.63;
x=0:dx:3.19;
[X,Y]=meshgrid(x,y);


q0=exp(-Y.^2./0.7);

% figure
% pcolor(X,Y,q0)
% shading interp
% axis equal

Q=zeros(size(q0,1),size(q0,2),length(tvec));

cont=0;
for t=tvec
    cont=cont+1;
    q=q0;
    for n=1:N
        dn=an(n)*X+bn(n);
        for m=-20:20
            q=q+alfan(n)*(-1)^m*exp(-(X-betan(n)*m-gamman(n)*t).^2./dn(n)-Y.^2./dn(n));
        end
    end
    figure(2)
    pcolor(X,Y,q)
    axis equal
    colorbar
    shading interp
    pause(0.2)
    Q(:,:,cont)=q;
end


QN=Q+(std(Q(:)).^2/SNR).^0.5*randn(size(Q));

    figure(3)
    subplot(2,1,1)
    pcolor(X,Y,Q(:,:,1))
    subplot(2,1,2)
    pcolor(X,Y,QN(:,:,1))
    for i=1:2
        subplot(2,1,i)
        axis equal
        h=colorbar;
        set(h,'ticklabelinterpreter','latex','fontsize',12)
        shading interp
        caxis([-1 3])
        colormap jet
    end
    set(gcf,'units','centimeters','position',[1 1 16 9])
    xlabel('$X$','interpreter','latex','fontsize',16)
    ylabel('$Y$','interpreter','latex','fontsize',16)
    set(gca,'ticklabelinterpreter','latex','fontsize',12)
    print('patterns.png','-dpng','-r300')
    
save('.\DATA\Patterns8','Q','QN','X','Y','an','bn','betan','gamman','alfan','tvec','SNR','fsamp');
