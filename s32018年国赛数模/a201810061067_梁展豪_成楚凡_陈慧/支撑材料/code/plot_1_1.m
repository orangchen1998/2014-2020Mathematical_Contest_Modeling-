%plot_1_1���������¶���ʱ��仯����
%A=xlsread('c:\users\f vadim\desktop\data.xlsx');
[m,n]=size(A);
i=1:m;
figure(1);
plot(i,A(:,1),'k','linewidth',1.5)
xlabel('ʱ��/s');ylabel('�¶�/���϶�');
figure(2);
plot(i,A(:,500),'k','linewidth',1.5)
xlabel('ʱ��/s');ylabel('�¶�/���϶�');
figure(3);
plot(i,A(:,500+360),'k','linewidth',1.5)
xlabel('ʱ��/s');ylabel('�¶�/���϶�');
figure(4);
plot(i,A(:,500+360+600),'k','linewidth',1.5)
xlabel('ʱ��/s');ylabel('�¶�/���϶�');