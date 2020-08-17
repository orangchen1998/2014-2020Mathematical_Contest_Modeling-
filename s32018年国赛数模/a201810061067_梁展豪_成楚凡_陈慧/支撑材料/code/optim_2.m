%% optim_2.m
%��������solve_u.m
function [sol,xx,zz]=optim_2(p)
[f,na]=solve_u();
global m1;
global m2;
c1=0.5;c2=0.5;w=0.6;%����Ⱥ����ٳ�������������
sizepop=20;   %��Ⱥ��ģ
Vmax=1;
Vmin=-1;
popmax=5;
popmin=-5;
pmax=20;z0=inf;sol=zeros(1,1);exitflag=1;z=[];
m1=unifrnd(0.0006,0.025,pmax,1);
m2=unifrnd(0.0006,0.0064,pmax,1);
options = optimoptions('fsolve','Display','none');
disp('�������������У����Ժ�...')
for i=1:sizepop
    pop(i,:)=5*rands(1,2);
    V(i,:)=rands(1,2); 
    fitness(i)=targetfun3(pop(i,:)); 
end
[bestfitness bestindex]=min(fitness);
zbest=pop(bestindex,:); 
gbest=pop;
fitnessgbest=fitness;
fitnesszbest=bestfitness;
%��������
for i=1:pmax
    disp(['���ڼ����',num2str(i),'�����ݣ���',num2str(pmax),'��'])
    r2=m1(i);
    r4=m2(i);
    %����tt
    ttt=root3_1(r2,r4);
    if exitflag==1
        [st,T30]=limitst_2(ttt,r2,r4,f,na);
        if st==1
            z(i)=targetfun2(p,T30,r2,r4);
            disp([num2str(r2),' ',num2str(r4),' ',num2str(T30),' ',num2str(z(i))])
            if z(i)<z0
                z0=z(i);
                sol=[r2,r4];
            end
        end
    end
    for j=1:sizepop
        V(j,:) = w*V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        pop(j,:)=pop(j,:)+0.5*V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        fitness(j)=targetfun3(pop(j,:));
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
    end
    yy(i)=fitnesszbest;
end
disp(['���ű���:',num2str(r2),num2str(r4)])
disp(['���Ž�:',num2str(z0)])
j=length(z(z>0));
zz(1:j)=z(z>0);
xx=1:j;
end