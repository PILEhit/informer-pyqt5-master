% sdp make 72维度数据
% EMD
% 数据规范化
k1 = 5;%雪花放大系数
k2 = pi/6;
T = 1;
L = 1;%取值间隔
a = 1;%时滞系数
len = length(data2(1,:));
for j=1:length(data2(:,1))
    data = data2(j,:);
    max_data = max(data);
    min_data = min(data);
    r=zeros(1,len);%极坐标转换的极径
    theta1=zeros(1,len);theta2=zeros(1,len);theta3=zeros(1,len);
    theta4=zeros(1,len);theta5=zeros(1,len);theta6=zeros(1,len);
    %顺时针极角组合
    phi1=zeros(1,len);phi2=zeros(1,len);phi3=zeros(1,len);
    phi4=zeros(1,len);phi5=zeros(1,len);phi6=zeros(1,len);
    %逆时针极角组合
    %(L*fix(i/L)+1)

    for i=fix(1+0*len/T):L:fix(1*len/T)-a-1
        r(i)=k1*(data(i)-min_data)/(max_data-min_data);
        theta1(i) = (2*pi/6)*1+k2*((data(i+a)-min_data)/(max_data-min_data));
        theta2(i) = (2*pi/6)*2+k2*((data(i+a)-min_data)/(max_data-min_data));
        theta3(i) = (2*pi/6)*3+k2*((data(i+a)-min_data)/(max_data-min_data));
        theta4(i) = (2*pi/6)*4+k2*((data(i+a)-min_data)/(max_data-min_data));
        theta5(i) = (2*pi/6)*5+k2*((data(i+a)-min_data)/(max_data-min_data));
        theta6(i) = (2*pi/6)*6+k2*((data(i+a)-min_data)/(max_data-min_data));
        phi1(i) = (2*pi/6)*1-k2*((data(i+a)-min_data)/(max_data-min_data));
        phi2(i) = (2*pi/6)*2-k2*((data(i+a)-min_data)/(max_data-min_data));
        phi3(i) = (2*pi/6)*3-k2*((data(i+a)-min_data)/(max_data-min_data));
        phi4(i) = (2*pi/6)*4-k2*((data(i+a)-min_data)/(max_data-min_data));
        phi5(i) = (2*pi/6)*5-k2*((data(i+a)-min_data)/(max_data-min_data));
        phi6(i) = (2*pi/6)*6-k2*((data(i+a)-min_data)/(max_data-min_data));
    end
    figure(j);
    t1=polarplot(theta1,r,'.');hold on;
    t2=polarplot(theta2,r,'.');t3=polarplot(theta3,r,'.');t4=polarplot(theta4,r,'.');t5=polarplot(theta5,r,'.');t6=polarplot(theta6,r,'.');
    p1=polarplot(phi1,r,'.');p2=polarplot(phi2,r,'.');p3=polarplot(phi3,r,'.');p4=polarplot(phi4,r,'.');p5=polarplot(phi5,r,'.');p6=polarplot(phi6,r,'.');
    set(gca,'FontSize',20); % 还是set好使
    % t1.Fontsize=20;
    hold off
    axis off %坐标轴隐藏
    saveas(j,['D:\网课5\不务正业\温度映射\sdp\',num2str(j-1),'.png']);
    close(j)
    img = imread(['D:\网课5\不务正业\温度映射\sdp\',num2str(j-1),'.png']);
     %修改像素为189*189
     re_img = imresize(img,[8,9]);
     imwrite(re_img,['D:\网课5\不务正业\温度映射\reshape_sdp\',num2str(j-1),'.png']);
end

    
