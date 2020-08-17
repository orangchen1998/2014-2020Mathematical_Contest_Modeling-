% targetfun1.m
% 变量说明:theta1-theta5 x(1)-x(5) h x(6) d x(7) m x(8)
function f=targetfun1(p,T30,r2)
f=p(1)*47+p(2)*T30+r2*p(3);