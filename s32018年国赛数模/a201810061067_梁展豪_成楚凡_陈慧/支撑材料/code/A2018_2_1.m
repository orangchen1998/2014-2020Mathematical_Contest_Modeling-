% A2018_2_1.m ���������⼰�����ȷ���
clc,clear,close all
global AA;
[sol1,xx1,zz1]=optim_1([0.59 0.34 0.07]);
%min(AA(:,6))
min(AA)
max(AA)
%max(AA(:,8))
[sol2,xx2,zz2]=optim_1([0.59+(0.59*0.02),0.34,0.07]);
[sol3,xx3,zz3]=optim_1([0.59,0.34+(0.34*0.02),0.2,0.07]);
[sol4,xx4,zz4]=optim_1([0.59,0.34,0.07+(0.2*0.02)]);
figure(1)
plot(xx1,zz1,'k',xx1,zz1,'kO','linewidth',2)
xlabel('����Լ�������ĸ���'),ylabel('��Ӧ��'),legend('��Ӧ������'),box off
xx1=xx1(1:40);zz1=zz1(1:40);xx2=xx2(1:40);zz2=zz2(1:40); % ��ȡǰ40������
xx3=xx3(1:40);zz3=zz3(1:40);xx4=xx4(1:40);zz4=zz4(1:40);xx5=xx5(1:40);zz5=zz5(1:40);
figure(2),hold on
plot(xx2,zz2-zz1,'k','linewidth',2),plot(xx2,zeros(1,40),'k-.','linewidth',2)
xlabel('����Լ�������ĸ���'),ylabel('��Ӧ��'),legend('ϵ���䶯2%ʱ��Ӧ�Ȳ�ֵ'),box off
figure(3),hold on
plot(xx3,zz3-zz1,'k','linewidth',2),plot(xx3,zeros(1,40),'k-.','linewidth',2)
xlabel('����Լ�������ĸ���'),ylabel('��Ӧ��'),legend('ϵ���䶯2%ʱ��Ӧ�Ȳ�ֵ'),box off
figure(4),hold on
plot(xx4,zz4-zz1,'k','linewidth',2),plot(xx4,zeros(1,40),'k-.','linewidth',2)
xlabel('����Լ�������ĸ���'),ylabel('��Ӧ��'),legend('ϵ���䶯2%ʱ��Ӧ�Ȳ�ֵ'),box off
st1=std(zz2-zz1);st2=std(zz3-zz1);st3=std(zz4-zz1);
figure(5),hold on
disp(['60��������¶�ϵ���䶯2%ʱ��Ӧ�Ȳ�ֵ��׼��Ϊ:',num2str(st1)])
disp(['55��������¶�ϵ���䶯2%ʱ��Ӧ�Ȳ�ֵ��׼��Ϊ:',num2str(st2)])
disp(['֯�������ϵ���䶯2%ʱ��Ӧ�Ȳ�ֵ��׼��Ϊ:',num2str(st3)])
