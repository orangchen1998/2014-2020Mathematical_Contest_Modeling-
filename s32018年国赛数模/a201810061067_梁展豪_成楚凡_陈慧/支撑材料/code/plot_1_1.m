%plot_1_1画出各层温度随时间变化曲线
%A=xlsread('c:\users\f vadim\desktop\data.xlsx');
[m,n]=size(A);
i=1:m;
figure(1);
plot(i,A(:,1),'k','linewidth',1.5)
xlabel('时间/s');ylabel('温度/摄氏度');
figure(2);
plot(i,A(:,500),'k','linewidth',1.5)
xlabel('时间/s');ylabel('温度/摄氏度');
figure(3);
plot(i,A(:,500+360),'k','linewidth',1.5)
xlabel('时间/s');ylabel('温度/摄氏度');
figure(4);
plot(i,A(:,500+360+600),'k','linewidth',1.5)
xlabel('时间/s');ylabel('温度/摄氏度');