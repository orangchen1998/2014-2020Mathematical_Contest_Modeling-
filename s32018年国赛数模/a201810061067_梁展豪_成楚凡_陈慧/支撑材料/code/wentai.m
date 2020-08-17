%计算稳态时的温度和函数
clear;clc;
t4=75;
t=[t4];
t0=48.08;
r1=0.0006;
r2=0.006;
r3=0.0036;
r4=0.005;
r=[r1,r2,r3,r4];
k1=0.082;
k2=0.37;
k3=0.045;
k4=0.028;
k=[k1,k2,k3,k4];
q_ave=(t4-t0)/(r1/k1+r2/k2+r3/k3+r4/k4);
ttt=[];
for i=1:4
t(end+1)=t(end)-q_ave*r(1,i)/k(1,i);
end
x=0:0.000001:r1;
tt=t(1,1)-q_ave*x.^2/(2*k1)-((t(1,1)-t(1,2))/r1-q_ave*r1/2*k1)*x;

ttt=[ttt tt];
plot(x,tt);
hold on;
x=0:0.000001:r2;
tt=t(1,2)-q_ave*x.^2/(2*k2)-((t(1,2)-t(1,3))/r2-q_ave*r2/2*k2)*x;

ttt=[ttt tt];
plot(x+r1,tt);
x=0:0.000001:r3;
tt=t(1,3)-q_ave*x.^2/(2*k3)-((t(1,3)-t(1,4))/r3-q_ave*r3/2*k3)*x;

ttt=[ttt tt];
plot(x+r1+r2,tt);
x=0:0.000001:r4;
tt=t(1,4)-q_ave*x.^2/(2*k4)-((t(1,4)-t(1,5))/r4-q_ave*r4/2*k4)*x;

ttt=[ttt tt];
plot(x+r1+r2+r3,tt);
%倒置ttt
ttt=fliplr(ttt);
%na=
%syms q;
%x0=double(solve(t1-q*x^2/2*na