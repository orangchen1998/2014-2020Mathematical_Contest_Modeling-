% A2018_3_1.m 问题三的求解及灵敏度分析

global AA;
%传入系数

figure(1)
plot(xx1,zz1,'k',xx1,zz1,'kO','linewidth',2)
xlabel('满足约束条件的个数'),ylabel('适应度'),legend('适应度曲线'),box off
%xx1=xx1(1:40);zz1=zz1(1:40);xx2=xx2(1:40);zz2=zz2(1:40); % 截取前40个数据
%xx3=xx3(1:40);zz3=zz3(1:40);xx4=xx4(1:40);zz4=zz4(1:40);xx5=xx5(1:40);zz5=zz5(1:40);
figure(2),hold on
plot(xx2,zz2-zz1,'k','linewidth',2)
xlabel('满足约束条件的个数'),ylabel('适应度'),legend('系数变动2%时适应度差值'),box off
figure(3),hold on
plot(xx3,zz3-zz1,'k','linewidth',2)
xlabel('满足约束条件的个数'),ylabel('适应度'),legend('系数变动2%时适应度差值'),box off
figure(4),hold on
plot(xx4,zz4-zz1,'k','linewidth',2),plot(xx4,zeros(1,20),'k-.','linewidth',2)
xlabel('满足约束条件的个数'),ylabel('适应度'),legend('系数变动2%时适应度差值'),box off
figure(5),hold on
plot(xx5,zz5-zz1,'k','linewidth',2),plot(xx5,zeros(1,20),'k-.','linewidth',2)
xlabel('满足约束条件的个数'),ylabel('适应度'),legend('系数变动2%时适应度差值'),box off
st1=std(zz2-zz1);st2=std(zz3-zz1);st3=std(zz4-zz1);st4=std(zz5-zz1);
disp(['织物二层厚度:',num2str(sol1(1,1)),'织物四层厚度',num2str(sol1(1,2))])
disp(['60分钟体表温度系数变动2%时适应度差值标准差为:',num2str(st1)])
disp(['55分钟体表温度系数变动2%时适应度差值标准差为:',num2str(st2)])
disp(['织物二层厚度系数变动2%时适应度差值标准差为:',num2str(st3)])
disp(['织物四层厚度系数变动2%时适应度差值标准差为:',num2str(st4)])