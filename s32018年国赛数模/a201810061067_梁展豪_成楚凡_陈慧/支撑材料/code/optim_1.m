%% optim_1.m
%��������solve_u.m

function [sol,xx,zz]=optim_1(p)
[f,na]=solve_u();
syms t4;
global AA;
t0=75;
r1=0.0006;
r3=0.0036;
r4=0.0055;
k1=0.082;
k2=0.37;
k3=0.045;
k4=0.028;
k=[k1,k2,k3,k4];
%������СֵΪinf,�����ռ����ֵ
pmax=20;z0=inf;sol=zeros(1,1);exitflag=1;z=[];
m=unifrnd(0.0006,0.025,pmax,1);
syms y d
options = optimoptions('fsolve','Display','none');
disp('�������������У����Ժ�...')
count=1;
for i=1:pmax
    disp(['���ڼ����',num2str(i),'�����ݣ���',num2str(pmax),'��'])
    r2=m(i);
    %����tt
    ttt=root2_1(r2);
    %����н�
    if exitflag==1
        %�������������ж��Ƿ����
        [st,T30]=limitst_1(ttt,r2,f,na);
        %�������������鿴�Ƿ�Ϊ���Ž�
        if st==1
           z(i)=targetfun1(p,T30,r2);
           if p(1)==0.59&&p(2)==0.34&&p(3)==0.07
                disp([num2str(r2),' ',num2str(T30),' ',num2str(z(i))])
                AA(count,:)=r2;
                count=count+1;
           end
            if z(i)<z0
                z0=z(i);
                sol=r2;
            end
        end
     end
end
disp(['���ű���:',num2str(r2)])
disp(['���Ž�:',num2str(z0)])
j=length(z(z>0));
zz(1:j)=z(z>0);
xx=1:j;
end