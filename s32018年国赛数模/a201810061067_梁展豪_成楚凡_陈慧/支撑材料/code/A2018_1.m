%向前差分迭代计算
T_huanj=75;
r1=0.0006;
r2=0.006;
r3=0.0036;
r4=0.005;
h=15;
%f(1,1)=f(1,1)/1000;
%f(1,2)=f(1,2)/10;
%f=f/1000;
%事先运行稳态.m
[f,na]=solve_u();
[m,n]=size(ttt);
M=ttt;
for i=1:6000
    i
    M(i,1)
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