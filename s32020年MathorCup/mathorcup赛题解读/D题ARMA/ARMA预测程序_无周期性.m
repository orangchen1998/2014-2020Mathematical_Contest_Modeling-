dX=diff(X);
p=2;%AR����
q=3;%MA����
coeff=coeffX{p+1,q+1};
m=2;%Ԥ��n����
[sigmaForecast,w_Forecast] = garchpred(coeff,dX,m); %���� n ��Ԥ��ֵ