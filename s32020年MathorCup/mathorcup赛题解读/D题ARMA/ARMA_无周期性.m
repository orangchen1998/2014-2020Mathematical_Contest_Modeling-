
clc,clear
load A.txt
X=A(:,2)';
n=size(X,2);%���ݾ���άһ��n��
x(1)=0; %����ʼֵ
for j=2:n
x(j)=0.8*x(j-1)+X(j)-0.4*X(j-1); %����������
end
for i=0:3
for j=0:3
spec= garchset('R',i,'M',j,'Display','off'); %ָ��ģ�͵Ľṹ
[coeffX{i+1,j+1},errorsX{i+1,j+1},LLFX{i+1,j+1},Innovations{i+1,j+1}]=garchfit(spec,x); 
%������� Coeff��ģ�͵Ĳ�������ֵ��
%errorsX ��ģ�Ͳ����ı�׼�
%LLFX �������Ȼ���Ʒ��еĶ���Ŀ�꺯��ֵ
%Innovations �ǲв�����
S(i+1,j+1)=std(Innovations{i+1,j+1});%��׼��
num=garchcount(coeffX{i+1,j+1}); %������ϲ����ĸ���
%����Akaike��Bayesian��Ϣ��׼
[aic,bic]=aicbic(LLFX{i+1,j+1},num,n);
fprintf('R=%d,M=%d,AIC=%f,BIC=%f\n',i,j,aic,bic); %��ʾ������
end
end