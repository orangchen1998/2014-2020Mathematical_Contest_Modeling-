clc, clear
a0=xlsread('训练数据集.xls');
a1=xlsread('预测数据集.xlsx');
b0=a0(:,[1:5]); dd0=a1(:,[1:5]); %提取已分类和待分类的数据
% dd0=[dd0,80*ones(835,1)];
[b,ps]=mapstd(b0); %已分类数据的标准化
dd=mapstd('apply',dd0,ps); %待分类数据的标准化
group=a0(:,[6]); %已知样本点的类别标号
s=svmtrain(b,group) %训练支持向量机分离器
sv_index=s.SupportVectorIndices %返回支持向量的标号
beta=s.Alpha %返回分类函数的权系数
bb=s.Bias %返回分类函数的常数项
mean_and_std_trans=s.ScaleData %第 1 行返回的是已知样本点均值向量的相反数，第 2行返回的是标准差向量的倒数
check=svmclassify(s,b) %验证已知样本点
err_rate=1-sum(group==check)/length(group) %计算已知样本点的错判率
solution=svmclassify(s,dd) %对待判样本点进行分类
solution(isnan(solution))=0.5;%部分信息缺失导致的无法分类