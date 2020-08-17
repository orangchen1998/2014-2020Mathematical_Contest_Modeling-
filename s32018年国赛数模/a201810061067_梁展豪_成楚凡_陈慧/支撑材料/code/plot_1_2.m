%plot_1_2
%A=xlsread('c:\users\f vadim\desktop\data.xlsx');
[m,~]=size(A);
[n,~]=size(M);
i=1:m;
plot(i,A(:,1),'k','linewidth',1.5)
hold on
i=1+1645-n+50:1645+50;
N=flipud(M(:,1));
plot(i,N,'k','linewidth',1.5)
xlabel('时间/s');ylabel('温度/摄氏度');