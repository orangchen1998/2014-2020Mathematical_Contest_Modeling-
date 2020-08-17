%limitst_2
%事先运行solve_u.m
function [st,T30]=limitst_2(ttt,r2,r4,f,na)
%计算得出M时空矩阵
T_huanj=80;
r1=0.0006;
r3=0.0036;
h=15;

[m,n]=size(ttt);
M=ttt;
for i=1:4000
   %i
    %M(i,1)
 if M(i,1)-37<=0.0001
     break;
 end
 %没有到达体温继续迭代
M(end+1,:)=zeros(1,n);
for j=2:n-1
    %迭代
    if j<=r4/0.000001
        ff=f(1,1);
    elseif j>r4/0.000001&&j<(r4+r3)/0.000001
        ff=f(1,2);
    elseif j>(r4+r3)/0.000001&&j<(r4+r3+r2)/0.000001
        ff=f(1,3);
    else
        ff=f(1,4);
    end
    M(i+1,j)=-M(i,j-1)*ff+(1+2*ff)*M(i,j)-M(i,j+1)*ff;
end
 %近肤点
M(i+1,1)=M(i+1,2);

%近热点
delta_x=0.000001;
tmp=h*delta_x/na(1,4);
M(i+1,n)=(M(i+1,n-1)+tmp*T_huanj)/(tmp+1);
end

[m,n]=size(M);
T30=M(300,1)
m
%if m>=1800&&M(300)<=44;
if m>=1100&&M(300)>44;
    st=1;
else
    st=0;
end
end
