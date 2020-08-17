load B.txt
spec2= garchset('R',1,'M',13,'Display','off'); %指定模型的结构,这里确定阶次AR为1，MA为13
[coeffX,errorsX,LLFX] = garchfit(spec2,w); %拟合参数
[sigmaForecast,w_Forecast] = garchpred(coeffX,w,n) %求 w 的预报值
yhat=y(m2)+cumsum(w_Forecast) %求 y 的预报值
for j=1:n
x(m1+j)=yhat(j)+x(m1+j-s);
end
x_hat=x(m1+1:end) %复原到原始数据的预报值