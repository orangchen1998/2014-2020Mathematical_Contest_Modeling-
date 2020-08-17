
clc,clear
load A.txt
X=A(:,2)';
n=size(X,2);%数据矩阵维一行n列
x(1)=0; %赋初始值
for j=2:n
x(j)=0.8*x(j-1)+X(j)-0.4*X(j-1); %产生样本点
end
for i=0:3
for j=0:3
spec= garchset('R',i,'M',j,'Display','off'); %指定模型的结构
[coeffX{i+1,j+1},errorsX{i+1,j+1},LLFX{i+1,j+1},Innovations{i+1,j+1}]=garchfit(spec,x); 
%输出参数 Coeff是模型的参数估计值，
%errorsX 是模型参数的标准差，
%LLFX 是最大似然估计法中的对数目标函数值
%Innovations 是残差向量
S(i+1,j+1)=std(Innovations{i+1,j+1});%标准差
num=garchcount(coeffX{i+1,j+1}); %计算拟合参数的个数
%计算Akaike和Bayesian信息标准
[aic,bic]=aicbic(LLFX{i+1,j+1},num,n);
fprintf('R=%d,M=%d,AIC=%f,BIC=%f\n',i,j,aic,bic); %显示计算结果
end
end