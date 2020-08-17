dX=diff(X);
p=2;%AR阶数
q=3;%MA阶数
coeff=coeffX{p+1,q+1};
m=2;%预测n个数
[sigmaForecast,w_Forecast] = garchpred(coeff,dX,m); %计算 n 步预报值