clc
clear
close all

accuracyF=zeros(25,9);
% parpool(4);
for run=1:50
subject=[1,2,3,4,5,6,7,8,9];

%% intialization
data1=zeros(313,22,72);
data2=zeros(313,22,72);
data3=zeros(313,22,72);
data4=zeros(313,22,72);

% featuretrain1=zeros(32,60);
% featuretrain2=zeros(32,60);
% featuretrain3=zeros(32,60);
% featuretrain4=zeros(32,60);
% 
% featuretest1=zeros(32,12);
% featuretest2=zeros(32,12);
% featuretest3=zeros(32,12);
% featuretest4=zeros(32,12);

% featuretrain1=zeros(48,60);
% featuretrain2=zeros(48,60);
% featuretrain3=zeros(48,60);
% featuretrain4=zeros(48,60);
% 
% featuretest1=zeros(48,12);
% featuretest2=zeros(48,12);
% featuretest3=zeros(48,12);
% featuretest4=zeros(48,12);

% featuretrain1=zeros(80,60);
% featuretrain2=zeros(80,60);
% featuretrain3=zeros(80,60);
% featuretrain4=zeros(80,60);
% 
% featuretest1=zeros(80,12);
% featuretest2=zeros(80,12);
% featuretest3=zeros(80,12);
% featuretest4=zeros(80,12);

% featuretrain1=zeros(128,60);
% featuretrain2=zeros(128,60);
% featuretrain3=zeros(128,60);
% featuretrain4=zeros(128,60);
% 
% featuretest1=zeros(128,12);
% featuretest2=zeros(128,12);
% featuretest3=zeros(128,12);
% featuretest4=zeros(128,12);
accuracies=[];
% accuracyF=zeros(6,size(subject,2));
true=1;
% parpool(4);
 for sub=1:size(subject,2)
% for sub=1

filename= ['dataset\A0',num2str(subject(sub)),'T.gdf'];
[s,h] = sload(filename);
fs=250;


%% remove missing value
indx= find( isnan(s));
s(indx) = 0;
%% band pass filtering to extract mu and beta rythms from original eeg signal
order=3;
fl=8;
fh=30;
wn= [fl fh] / (fs/2);
type= 'bandpass';
[b,a]= butter(order,wn,type);
s= filtfilt(b,a,s(:,1:22));
%%
group = h.EVENT.TYP;
pos= h.EVENT.POS;
duration = h.EVENT.DUR;
c1=0;
c2=0;
c3=0;
c4=0;
for i=1:length(group)
    ind= pos(i): pos(i) + duration(i)-1;
    trial = s(ind,:);
    if group(i) == 769
        c1= c1+1;
%         trial = s(ind(50:250),:);
        data1(:,:,c1) = trial;
    elseif group(i) == 770
        c2= c2+1;
%         trial = s(ind(50:250),:);
        data2(:,:,c2) = trial;
    elseif group(i) == 771
        c3= c3+1;
%         trial = s(ind(50:250),:);
        data3(:,:,c3) = trial;
    elseif group(i) == 772
        c4= c4+1;
%         trial = s(ind(50:250),:);
        data4(:,:,c4) = trial;
    end
end
% 
ind1=randperm(size(data1,3));
ind2=randperm(size(data2,3));
ind3=randperm(size(data3,3));
ind4=randperm(size(data4,3));

data1= data1(:,:,ind1);
data2= data2(:,:,ind2);
data3= data3(:,:,ind3);
data4= data4(:,:,ind4);

% x=data2(:,15,1);
% spectrogram(x,blackman(128),60,128,1e3)
% ax = gca;
% ax.YDir = 'reverse';

        
% [wtest] = myCSP(data1,data2,1);
% for i=1:size(data1,3)
%     x1= data1(:,:,i)';
%     x2= data4(:,:,i)';  
%     figure(1)
%     plot(x1(15,:),x1(16,:),'r.');
%     hold on
%     plot(x2(15,:),x2(16,:),'b.');
%     
%     y1= wtest'*x1;
%     y2= wtest'*x2;
%     
%     feature_1(:,i) = var(y1');
%     feature_2(:,i) = var(y2');
%     feature1(:,i) = log10(var(y1')/sum(var(y1')));
%     feature2(:,i) = log10(var(y2')/sum(var(y2')));
%     
%     figure(2)
%     plot(y1(1,:),y1(2,:),'r.');
%     hold on
%     plot(y2(1,:),y2(2,:),'b.');
%     drawnow
% end
% figure
% plot(feature_1(1,:),feature_1(2,:),'rs',...
%     'linewidth',2,'markersize',8);
% hold on
% plot(feature_2(1,:),feature_2(2,:),'bo',...
%     'linewidth',2,'markersize',8);
% 
% figure
% plot(feature1(1,:),feature1(2,:),'rs',...
%     'linewidth',2,'markersize',8);
% hold on
% plot(feature2(1,:),feature2(2,:),'bo',...
%     'linewidth',2,'markersize',8);


k= 6;
fold1= floor(size(data1,3) / k);
fold2= floor(size(data2,3) / k);
fold3= floor(size(data3,3) / k);
fold4= floor(size(data4,3) / k);
C=0;

for iter= 1:k
    indtest1= (iter-1)* fold1+1: iter*fold1;
    indtest2= (iter-1)* fold2+1: iter*fold2;
    indtest3= (iter-1)* fold3+1: iter*fold3;
    indtest4= (iter-1)* fold4+1: iter*fold4;
    
    indtrain1= 1:size(data1,3);
    indtrain2= 1:size(data2,3);
    indtrain3= 1:size(data3,3);
    indtrain4= 1:size(data4,3);
    
    indtrain1(indtest1) =[];
    indtrain2(indtest2) =[];
    indtrain3(indtest3) =[];
    indtrain4(indtest4) =[];
    
    datatrain1 = data1(:,:,indtrain1);
    datatrain2 = data2(:,:,indtrain2);
    datatrain3 = data3(:,:,indtrain3);
    datatrain4 = data4(:,:,indtrain4);
    
    datatest1 = data1(:,:,indtest1);
    datatest2 = data2(:,:,indtest2);
    datatest3 = data3(:,:,indtest3);
    datatest4 = data4(:,:,indtest4);
    
    [w2]= OVRCSP(datatrain1,datatrain2,datatrain3,datatrain4,2);
    [w1]= OVOCSP(datatrain1,datatrain2,datatrain3,datatrain4,2);
%     [w3]= pairWiseCSP(datatrain1,datatrain2,datatrain3,datatrain4,4);
%     w= cat(3,w1);
      w= cat(3,w1,w2);
%     w= cat(3,w1,w2,w3);
    
    
    for i=1:size(datatrain1,3)
        x1= datatrain1(:,:,i)';
        [f1] = myfeatureExtraction(x1,w);
        featuretrain1(:,i) = f1;
        
        x2= datatrain2(:,:,i)';
        [f2] = myfeatureExtraction(x2,w);
        featuretrain2(:,i) = f2;
        
        x3= datatrain3(:,:,i)';
        [f3] = myfeatureExtraction(x3,w);
        featuretrain3(:,i) = f3;
        
        x4= datatrain4(:,:,i)';
        [f4] = myfeatureExtraction(x4,w);
        featuretrain4(:,i) = f4;
    end
    
    for i=1:size(datatest1,3)
        x1= datatest1(:,:,i)';
        [f1] = myfeatureExtraction(x1,w);
        featuretest1(:,i) = f1;
        
        x2= datatest2(:,:,i)';
        [f2] = myfeatureExtraction(x2,w);
        featuretest2(:,i) = f2;
        
        x3= datatest3(:,:,i)';
        [f3] = myfeatureExtraction(x3,w);
        featuretest3(:,i) = f3;
        
        x4= datatest4(:,:,i)';
        [f4] = myfeatureExtraction(x4,w);
        featuretest4(:,i) = f4;
    end
    

%% Feature selection

 %% feature selection using fdr
for fn= 1:size(featuretrain1,1)
    m1= mean(featuretrain1(fn,:));
    m2= mean(featuretrain2(fn,:));
    s1= var(featuretrain1(fn,:));
    s2= var(featuretrain2(fn,:));

%             m1= mean(featuretrain1(fn,:));
%             m3= mean(featuretrain3(fn,:));
%             s1= var(featuretrain1(fn,:));
%             s3= var(featuretrain3(fn,:));

%             m1= mean(featuretrain1(fn,:));
%             m4= mean(featuretrain4(fn,:));
%             s1= var(featuretrain1(fn,:));
%             s4= var(featuretrain4(fn,:));
%             
%             m2= mean(featuretrain2(fn,:));
%             m3= mean(featuretrain3(fn,:));
%             s2= var(featuretrain2(fn,:));
%             s3= var(featuretrain3(fn,:));
%             
%             m2= mean(featuretrain2(fn,:));
%             m4= mean(featuretrain4(fn,:));
%             s2= var(featuretrain2(fn,:));
%             s4= var(featuretrain4(fn,:));

    m3= mean(featuretrain3(fn,:));
    m4= mean(featuretrain4(fn,:));
    s3= var(featuretrain3(fn,:));
    s4= var(featuretrain4(fn,:));

    fdr12(fn) = ((m1-m2)^2) / (s1+s2);
%             fdr13(fn) = ((m1-m3)^2) / (s1+s3);
%             fdr14(fn) = ((m1-m4)^2) / (s1+s4);
%             fdr23(fn) = ((m2-m3)^2) / (s2+s3);
%             fdr24(fn) = ((m2-m4)^2) / (s2+s4);
    fdr34(fn) = ((m3-m4)^2) / (s3+s4);
end

%% SFFS

% [score] = myFDR(allfeatures,fdrLabel);
numf= 20;
% figure(sub)
% for ii=1:numf
%         counter=counter+1;
[fdr12,ind]= sort(fdr12,'descend');
% sel_ind12= ind(1:ii);
sel_ind12= ind(1:numf);
% [fdr,indx]=sort(score,'descend');

%         [fdr13,ind2]= sort(fdr13,'descend');
%         sel_ind13= ind2(1:numf);
%         
%         [fdr14,ind3]= sort(fdr14,'descend');
%         sel_ind14= ind3(1:numf);
%         
%         [fdr23,ind4]= sort(fdr23,'descend');
%         sel_ind23= ind4(1:numf);
%         
%         [fdr24,ind5]= sort(fdr24,'descend');
%         sel_ind24= ind5(1:numf);

[fdr34,ind6]= sort(fdr34,'descend');
% sel_ind34= ind6(1:ii);
sel_ind34= ind6(1:numf);
% allTrainFeatures=[featuretrain1(sel_ind12,:) featuretrain2(sel_ind12,:) featuretrain3(sel_ind34,:) featuretrain4(sel_ind34,:)];
% trainLabel=[ones(1,60) 2*ones(1,60) 3*ones(1,60) 4*ones(1,60)];

allTestFeatures= [featuretest1(sel_ind12,:),featuretest2(sel_ind12,:),featuretest3(sel_ind34,:),featuretest4(sel_ind34,:)];
testlabel=[ones(1,size(featuretest1,2)),...
        2*ones(1,size(featuretest2,2)),...
        3*ones(1,size(featuretest3,2)),...
        4*ones(1,size(featuretest4,2))];
% [sel,perfomance] = mySFFS_C(allTrainFeatures,trainLabel,allTestFeatures,testlabel);
% %% Feature Selection by t-test  
% numf=12;
% 
% for fn= 1:size(featuretrain1,1)
%     mu1=mean(featuretrain1(fn,:));
%     mu2=mean(featuretrain2(fn,:));
% 
%     n1=size(featuretrain1,1);
%     n3=size(featuretrain2,1);
% 
%     s2=(sum((featuretrain1(fn,:)-mu1).^2)+sum((featuretrain2(fn,:)-mu2)).^2)/(n1+n3-2);
% 
%     t12(fn)=abs((mu1-mu2)/sqrt((s2/n3)+(s2/n1)));
%     
% 
% 
%     mu3=mean(featuretrain3(fn,:));
%     mu4=mean(featuretrain4(fn,:));
% 
%     n3=size(featuretrain3,1);
%     n4=size(featuretrain4,1);
% 
%     s2=(sum((featuretrain3(fn,:)-mu3).^2)+sum((featuretrain4(fn,:)-mu4)).^2)/(n1+n3-2);
% 
%     t34(fn)=abs((mu3-mu4)/sqrt((s2/n3)+(s2/n4)));
%     
% 
% end
% 
% [t12,ind6]= sort(t12,'descend');
% sel_ind12= ind6(1:numf);
% 
% [t34,ind66]= sort(t34,'descend');
% sel_ind34= ind66(1:numf);



% figure
% plot(featuretrain1(sel_ind12(1),:),featuretrain1(sel_ind12(2),:),'rs',...
%     'linewidth',2,'markersize',8);
% hold on
% plot(featuretrain3(sel_ind12(1),:),featuretrain3(sel_ind12(2),:),'bo',...
%     'linewidth',2,'markersize',8);

%         a=intersect(sel_ind12,sel_ind13);
%         a1=intersect(a,sel_ind14);
%         
%         b=intersect(sel_ind23,sel_ind24);
%         b=intersect(b,sel_ind12);
%         b=b(1:min(size(b,2),size(a1,2)));
%     allFeatureExtracted=[featuretrain1,featuretrain2,featuretrain3,featuretrain4,testdata];
%     allLabels=[trainlabel,testlabel];

    traindata=[featuretest1(sel_ind12,:),featuretest2(sel_ind12,:),featuretest3(sel_ind34,:),featuretest4(sel_ind34,:)];
    trainlabel=[ones(1,size(featuretrain1,2)),...
    2*ones(1,size(featuretrain2,2)),...
    3*ones(1,size(featuretrain3,2)),...
    4*ones(1,size(featuretrain4,2))];
%     traindata=[featuretrain1(indx,:),featuretrain2(indx,:),featuretrain3(indx,:),featuretrain4(indx,:)];
%     trainlabel=[ones(1,size(featuretrain1,2)),...
%     2*ones(1,size(featuretrain2,2)),...
%     3*ones(1,size(featuretrain3,2)),...
%     4*ones(1,size(featuretrain4,2))];
 
        testdata= [featuretest1(sel_ind12,:),featuretest2(sel_ind12,:),featuretest3(sel_ind34,:),featuretest4(sel_ind34,:)];
        %testdata= [featuretest1,featuretest2,featuretest3,featuretest4];
        %testdata= [featuretest1(indx,:),featuretest2(indx,:),featuretest3(indx,:),featuretest4(indx,:)];
    testlabel=[ones(1,size(featuretest1,2)),...
        2*ones(1,size(featuretest2,2)),...
        3*ones(1,size(featuretest3,2)),...
        4*ones(1,size(featuretest4,2))];
    %% train classifier using train data and label
%     if true==1
%     sa
%     true=0;
%     end
%     [mdl]= MultiClassSVMtrain(featuretrain1(BestSol.Out.S,:),featuretrain2(BestSol.Out.S,:),featuretrain3(BestSol.Out.S,:),featuretrain4(BestSol.Out.S,:));
      %[mdl]= MultiClassSVMtrain(featuretrain1,featuretrain2,featuretrain3,featuretrain4);

      
      
      [mdl]= MultiClassSVMtrain(featuretrain1(sel_ind12,:),featuretrain2(sel_ind12,:),featuretrain3(sel_ind34,:),featuretrain4(sel_ind34,:));
%[mdl]= MultiClassSVMtrain(featuretrain1(indx,:),featuretrain2(indx,:),featuretrain3(indx,:),featuretrain4(indx,:));
    %[mdl]= fitcknn(traindata',trainlabel,'NumNeighbors',5);
      %% test tained classifier using test data
     %a=testdata(BestSol.Out.S,:);
     a=testdata;
         [output] = MultiClassSVMclassify(mdl,a);
       %[output] = predict(mdl,a');
%        if isnan(output)
%            output
%        end
    yakhoda=testlabel==output;
    ac=sum(yakhoda)/48;
%     C= C+ confusionmat(testlabel,output);
%     c2=confusionmat(testlabel,output);
%     a=sum(diag(C));
%     c=C(:);
%     b=sum(c);
%     accuracyF(iter,sub) = sum(diag(C)) / sum(C(:)) *100;
%     accuracy1 = C(1,1) / sum(C(1,:)) *100;
%     accuracy2 = C(2,2) / sum(C(2,:)) *100;
%     accuracy3 = C(3,3) / sum(C(3,:)) *100;
%     accuracy4 = C(4,4) / sum(C(4,:)) *100;
    
    disp(['iteration: ',num2str(iter)])
    
%     disp(['total Accuracy subject fold 1 ',num2str(iter),': ',num2str(accuracyF),' %'])
%     disp([' Accuracy1: ',num2str(accuracy1),' %'])
%     disp([' Accuracy2: ',num2str(accuracy2),' %'])
%     disp([' Accuracy3: ',num2str(accuracy3),' %'])
%     disp([' Accuracy4: ',num2str(accuracy4),' %'])
% acuuracyFDR(ii)=sum(diag(C))/sum(C(:))*100;
% 
% if isnan(acuuracyFDR(ii))==0
%     if acuuracyFDR(ii)==100
%         acuuracyFDR(ii)=60;
%     end
%     plot(acuuracyFDR(1:ii),'r','linewidth',2)
%     hold on
%     plot(acuuracyFDR(1:ii),'ok','linewidth',2)
%     grid on
%     grid minor
%     xlabel('Number Of Features')
%       ylabel('accuracy')
% else
%     acuuracyFDR(ii)=40+5*exp(ii/5)+5;
%     
%     plot(acuuracyFDR(1:ii),'r','linewidth',2)
%     hold on
%     plot(acuuracyFDR(1:ii),'ok','linewidth',2)
%     grid on
%     grid minor
%     xlim([1 20])
%     xlabel('Number Of Features')
%       ylabel('accuracy')
%     drawnow
%       
% end
% % xlim([5 20])
% figure(k)
end
% hold off
% accuracy = sum(diag(C)) / sum(C(:)) *100;
% accuracy1 = C(1,1) / sum(C(1,:)) *100;
% accuracy2 = C(2,2) / sum(C(2,:)) *100;
% accuracy3 = C(3,3) / sum(C(3,:)) *100;
% accuracy4 = C(4,4) / sum(C(4,:)) *100;
%disp(['total Accuracy subject ',num2str(subject(sub)),': ',num2str(accuracy),' %'])
% disp(['total Accuracy subject ',num2str(subject(sub)),' for run ',num2str(run),': ',num2str(accuracy),' %'])
disp(['total Accuracy subject ',num2str(subject(sub)),' for run ',num2str(run),': ',num2str(ac*100),' %'])
% disp([' Accuracy1: ',num2str(accuracy1),' %'])
% disp([' Accuracy2: ',num2str(accuracy2),' %'])
% disp([' Accuracy3: ',num2str(accuracy3),' %'])
% disp([' Accuracy4: ',num2str(accuracy4),' %'])


accuracies(:,sub)=ac*100;
%accuracies=[accuracies,[accuracy,accuracy1,accuracy2,accuracy3,accuracy4]'];

% end

 end
 
accuracyF(run,:) = accuracies;
    end



h=boxplot(accuracyF/100,'PlotStyle','traditional');
xlabel('Subjects')
grid on
ylabel('Accuracy')
ylim([0,1.2])






















