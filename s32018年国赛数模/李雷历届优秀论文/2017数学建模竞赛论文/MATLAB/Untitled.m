clc, clear
a0=xlsread('ѵ�����ݼ�.xls');
a1=xlsread('Ԥ�����ݼ�.xlsx');
b0=a0(:,[1:5]); dd0=a1(:,[1:5]); %��ȡ�ѷ���ʹ����������
% dd0=[dd0,80*ones(835,1)];
[b,ps]=mapstd(b0); %�ѷ������ݵı�׼��
dd=mapstd('apply',dd0,ps); %���������ݵı�׼��
group=a0(:,[6]); %��֪������������
s=svmtrain(b,group) %ѵ��֧��������������
sv_index=s.SupportVectorIndices %����֧�������ı��
beta=s.Alpha %���ط��ຯ����Ȩϵ��
bb=s.Bias %���ط��ຯ���ĳ�����
mean_and_std_trans=s.ScaleData %�� 1 �з��ص�����֪�������ֵ�������෴������ 2�з��ص��Ǳ�׼�������ĵ���
check=svmclassify(s,b) %��֤��֪������
err_rate=1-sum(group==check)/length(group) %������֪������Ĵ�����
solution=svmclassify(s,dd) %�Դ�����������з���
solution(isnan(solution))=0.5;%������Ϣȱʧ���µ��޷�����