%¼ÆËã³ömiu
function [f,na]=solve_u()
syms x;
%x0 = double(solve(cot(x)-x/8.14,x));
%X=fsolve('cot(x)-x/8.14',1);
delta_x=0.000001;
L=[0.005 0.0036 0.006 0.0006];
na=[0.028 0.045 0.37 0.082];
%na=[0.007898449 0.012693935 0.104372355 0.023131171];
c=[1005 1726 2100 1377];
%c=[283.4978843 486.8829337 592.3836389 388.4344147];
h=15;
rou=[1.18 74.2 862 300];
Bi=zeros(1,4);
miu=[];
a=[];
f=[];
for i=1:4
%Bi(1,i)=(h*delta_x)/na(1,i);
Bi(1,i)=(h*delta_x)/na(1,i);
miu(end+1)=fsolve(@(x)(cot(x)-x/Bi(1,i)),1);
end
%miu(end+1)=fsolve(@(x)(cot(x)-x/Bi(1,1)),1);
%miu(end+1)=fsolve(@(x)(cot(x)-x/Bi(1,2)),1);
%miu(end+1)=fsolve(@(x)(cot(x)-x/Bi(1,3)),1);
%miu(end+1)=fsolve(@(x)(cot(x)-x/Bi(1,4)),1);
for i=1:4
    a(1,end+1)=(h*miu(1,i))/(c(1,i)*rou(1,i));
    f(1,end+1)=a(1,end)*1/0.39^2;%1-0.37£¬2-0.56
end
end