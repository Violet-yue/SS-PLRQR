clear all 
clc
addpath qtfm 
addpath spl2016
addpath functions
load IN.mat %Indian Pines with 10% training labeled samples
q2double4=@(H)double2q41(H,'inverse');
hsi = func_Normalized(img,1);
hsi=double(hsi);
[h,w,d]=size(hsi);
%%%
la=[];
for i=1:(h-1)*(w-1)
  if gt(i)==gt(i+1) & gt(i+1)==gt(i+h) & gt(i+h)==gt(i+h+1) & gt(i)>0
    la=[la,i];
  end
end
%%chose random patch for band grouping
RN=randperm(size(la,2),10);
im_2d=reshape(hsi,h*w,d);
%oc1=[];
%for ic=1:10  
lsx=RN(1);                                %%1~10; OC1=[]; ic;1
sx=[lsx,lsx+1,lsx+h,lsx+h+1];
hsi2=reshape(hsi,h*w,d);
%
%oc34_25_60=[];
%for is=25:5:60                        %%ip=1,2,3,4; 2,2,2
%% number of superbands
ki=40;                                    %%40:5:60; OC3=[]; is; 40,45,45
img1=im2uint8(mat2gray(hsi2(sx,:)));
[labels] = superp(img1',ki);  
%
%oc2=[];
%for ip=1:4 %number of bands in each super-band
ip=2;

tic
tz=[];tk1=[];tk2=[];X_2=[];L_2=[];S_2=[];
for j=1:ki
  tm=find(labels==j);
  tz=mod(tm,d);
  tz(find(tz==0))=d;
  tk1{j}=unique(tz);
  tk2{j}=hsi(:,:,tk1{j});
  tc=length(tk1{j});
    if tc>4
    [P]=Eigenface_f(im_2d(:,tk1{j})',ip); %%1,2,3,4 oc2=[]; ip;2
    PC1=im_2d(:,tk1{j})*P;
    PC2=reshape(PC1,h,w,ip);
    tk2{j}=PC2;
    end
  X0=tk2{j};
  X1=double2q41(X0);
  tc1=size(tk2{j},3);
  lambda = 1/(sqrt(max(h,w))*tc1);
  [L,S,ITER] = inexact_alm_qrpca(X1,lambda,1e-7,1000);
  X2=q2double4(L);
  X_2=cat(3,X_2,X2);
  L_2=cat(3,L_2,L);
  S_2=cat(3,S_2,S);
end

time_o=toc;
PC=reshape(X_2,h*w,ki*4);
fimg=PC';

[no_lines, no_rows, no_bands]= size(hsi);
OA=[];AA=[];kappa=[];ctime=[];CA=[];R=[];
for i=1
indexes=XX(:,i);
%% SVM classification
%%%
train_SL = GroundT(:,indexes);
train_samples = fimg(:,train_SL(1,:))';
train_labels= train_SL(2,:)';
%%%
test_SL = GroundT;
test_SL(:,indexes) = [];
test_samples = fimg(:,test_SL(1,:))';
test_labels = test_SL(2,:)';
tic
% Normalizing Training and original img
[train_samples,M,m] = scale_func(train_samples);
[fimg1] = scale_func(fimg',M,m);
% Selecting the paramter for SVM
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);
% Training using a Gaussian RBF kernel
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(train_labels,train_samples,parameter);
% Testing
Result = svmpredict(ones(no_lines*no_rows,1),fimg1,model); %%%
% Evaluation
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);
[OA_i,AA_i,kappa_i,CA_i]=confusion(GroudTest,ResultTest)%
R=[R,Result];
OA=[OA,OA_i];AA=[AA,AA_i];kappa=[kappa,kappa_i];CA=[CA,CA_i];
ctime_te_i=toc;
ctime=[ctime,ctime_te_i];
end
ctime_mean=mean(ctime);
OA1=mean(OA);OA_std=std(OA);
AA1=mean(AA);AA_std=std(AA);
kappa1=mean(kappa);K_std=std(kappa);
CA_mean=mean(CA,2)*100;  CA_std=std(CA')';
oour=[OA1,OA_std;AA1,AA_std;kappa1,K_std];
%
