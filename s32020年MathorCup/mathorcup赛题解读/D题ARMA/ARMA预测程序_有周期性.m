load B.txt
spec2= garchset('R',1,'M',13,'Display','off'); %ָ��ģ�͵Ľṹ,����ȷ���״�ARΪ1��MAΪ13
[coeffX,errorsX,LLFX] = garchfit(spec2,w); %��ϲ���
[sigmaForecast,w_Forecast] = garchpred(coeffX,w,n) %�� w ��Ԥ��ֵ
yhat=y(m2)+cumsum(w_Forecast) %�� y ��Ԥ��ֵ
for j=1:n
x(m1+j)=yhat(j)+x(m1+j-s);
end
x_hat=x(m1+1:end) %��ԭ��ԭʼ���ݵ�Ԥ��ֵ