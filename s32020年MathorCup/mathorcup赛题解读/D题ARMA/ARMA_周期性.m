clc,clear
load B.txt; %��ԭʼ���ݰ��ձ��еĸ�ʽ����ڴ��ı��ļ� water.txt
x=B;
s=12; %���� s=12
n=12; %Ԥ�����ݵĸ���
m1=length(x); %ԭʼ���ݵĸ���
for i=s+1:m1
y(i-s)=x(i)-x(i-s);
end
m2=length(y); %���ڲ�ֺ����ݵĸ���
w=diff(y); %���������ԵĲ������
m3=length(w); %�������ղ�ֺ����ݵĸ���
for i=0:3
for j=0:s+1
spec= garchset('R',i,'M',j,'Display','off'); %ָ��ģ�͵Ľṹ
[coeffX,errorsX,LLFX] = garchfit(spec,w); %��ϲ���
num=garchcount(coeffX); %������ϲ����ĸ���
%compute Akaike and Bayesian Information Criteria
[aic,bic]=aicbic(LLFX,num,m3);
fprintf('R=%d,M=%d,AIC=%f,BIC=%f\n',i,j,aic,bic); %��ʾ������
end
end