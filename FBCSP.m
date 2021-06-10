clc
clear
close all

accuracyF=zeros(10,9);
% parpool(4);
m=2;
for run=1:10
%     clearvars -except accuracyF run sub
    subject=[1,2,3,4,5,6,7,8,9];

    %% intialization
    data1=zeros(313,22,72);
    data2=zeros(313,22,72);
    data3=zeros(313,22,72);
    data4=zeros(313,22,72);

    % featuretrain1=zeros(80,60);
    % featuretrain2=zeros(80,60);
    % featuretrain3=zeros(80,60);
    % featuretrain4=zeros(80,60);
    % 
    % featuretest1=zeros(80,12);
    % featuretest2=zeros(80,12);
    % featuretest3=zeros(80,12);
    % featuretest4=zeros(80,12);
%     temptrain1=zeros(m*20,50);
%     temptrain2=zeros(m*20,50);
%     temptrain3=zeros(m*20,50);
%     temptrain4=zeros(m*20,50);
%     
%     temptest1=zeros(m*20,22);
%     temptest2=zeros(m*20,22);
%     temptest3=zeros(m*20,22);
%     temptest4=zeros(m*20,22);
    
%     featuretrain1=zeros(180,50);
    
    featuretest1= [];
    featuretest2 = [];
    featuretest3= [];
    featuretest4 = [];

    featuretrain1= [];
    featuretrain2= [];
    featuretrain3= [];
    featuretrain4= [];
    accuracies=[];
%     accuracyF=zeros(6,size(subject,2));
    true=1;
    %for sub=1:size(subject,2)
      for sub=1:9
%         bands=[4:4:36;8:4:40;6:4:38;10:4:42];
        bands=[4:4:36;8:4:40];
%         for b2=0:2
        for bn=1:size(bands,2)
        
%             wn=bands(b2+1:b2+2,bn);
        wn=bands(:,bn);
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
        % wn= [fl fh] / (fs/2);
        wn= [wn] / (fs/2);
        type= 'bandpass';
        [b,a]= butter(order,wn,type);
        snew= filtfilt(b,a,s(:,1:22));
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
            trial = snew(ind,:);
            if group(i) == 769
                c1= c1+1;
                data1(:,:,c1) = trial;
            elseif group(i) == 770
                c2= c2+1;
                data2(:,:,c2) = trial;
            elseif group(i) == 771
                c3= c3+1;
                data3(:,:,c3) = trial;
            elseif group(i) == 772
                c4= c4+1;
                data4(:,:,c4) = trial;
            end
        end
%     shuffle=randperm(size(data1,3));
%     data1= data1(:,:,shuffle);
%     data2= data2(:,:,shuffle);
%     data3= data3(:,:,shuffle);
%     data4= data4(:,:,shuffle);
ind1=randperm(size(data1,3));
ind2=randperm(size(data2,3));
ind3=randperm(size(data3,3));
ind4=randperm(size(data4,3));

    % k= 6;
    % fold1= floor(size(data1,3) / k);
    % fold2= floor(size(data2,3) / k);
    % fold3= floor(size(data3,3) / k);
    % fold4= floor(size(data4,3) / k);
    C=0;

    div=0.7;
        num1=floor(size(data1,3)*div);
        num2=floor(size(data2,3)*div);
        num3=floor(size(data3,3)*div);
        num4=floor(size(data4,3)*div);


        indtrain1 = ind1(1:num1);
        indtrain2 = ind2(1:num2);
        indtrain3 = ind3(1:num3);
        indtrain4 = ind4(1:num4);


        indtest1 = ind1(num1+1: size(data1,3));
        indtest2 = ind2(num2+1: size(data2,3));
        indtest3 = ind3(num3+1: size(data3,3));
        indtest4 = ind4(num4+1: size(data4,3));


        datatrain1= data1(:,:,indtrain1);
        datatrain2= data2(:,:,indtrain2);
        datatrain3= data3(:,:,indtrain3);
        datatrain4= data4(:,:,indtrain4);

        datatest1= data1(:,:,indtest1);
        datatest2= data2(:,:,indtest2);
        datatest3= data3(:,:,indtest3);
        datatest4= data4(:,:,indtest4);

    % for iter= 1:k
    %     indtest1= (iter-1)* fold1+1: iter*fold1;
    %     indtest2= (iter-1)* fold2+1: iter*fold2;
    %     indtest3= (iter-1)* fold3+1: iter*fold3;
    %     indtest4= (iter-1)* fold4+1: iter*fold4;
    %     
    %     indtrain1= 1:size(data1,3);
    %     indtrain2= 1:size(data2,3);
    %     indtrain3= 1:size(data3,3);
    %     indtrain4= 1:size(data4,3);
    %     
    %     indtrain1(indtest1) =[];
    %     indtrain2(indtest2) =[];
    %     indtrain3(indtest3) =[];
    %     indtrain4(indtest4) =[];
    %     
    %     datatrain1 = data1(:,:,indtrain1);
    %     datatrain2 = data2(:,:,indtrain2);
    %     datatrain3 = data3(:,:,indtrain3);
    %     datatrain4 = data4(:,:,indtrain4);
    %     
    %     datatest1 = data1(:,:,indtest1);
    %     datatest2 = data2(:,:,indtest2);
    %     datatest3 = data3(:,:,indtest3);
    %     datatest4 = data4(:,:,indtest4);

        [w2]= OVRCSP(datatrain1,datatrain2,datatrain3,datatrain4,m);
        [w1]= OVOCSP(datatrain1,datatrain2,datatrain3,datatrain4,m);

        w= cat(3,w1,w2);


        for i=1:size(datatrain1,3)
            x1= datatrain1(:,:,i)';
            [f1] = myfeatureExtraction(x1,w);
    %         featuretrain1(:,i) = f1;

            x2= datatrain2(:,:,i)';
            [f2] = myfeatureExtraction(x2,w);
    %         featuretrain2(:,i) = f2;

            x3= datatrain3(:,:,i)';
            [f3] = myfeatureExtraction(x3,w);
    %         featuretrain3(:,i) = f3;

            x4= datatrain4(:,:,i)';
            [f4] = myfeatureExtraction(x4,w);
    %         featuretrain4(:,i) = f4;

            temptrain1(:,i) = f1;
            temptrain2(:,i) = f2;
            temptrain3(:,i) = f3;
            temptrain4(:,i) = f4;
        end

        for i=1:size(datatest1,3)
            x1= datatest1(:,:,i)';
            [f1] = myfeatureExtraction(x1,w);
    %         featuretest1(:,i) = f1;

            x2= datatest2(:,:,i)';
            [f2] = myfeatureExtraction(x2,w);
    %         featuretest2(:,i) = f2;

            x3= datatest3(:,:,i)';
            [f3] = myfeatureExtraction(x3,w);
    %         featuretest3(:,i) = f3;

            x4= datatest4(:,:,i)';
            [f4] = myfeatureExtraction(x4,w);
    %         featuretest4(:,i) = f4;

            temptest1(:,i) = f1;
            temptest2(:,i) = f2;
            temptest3(:,i) = f3;
            temptest4(:,i) = f4;
        end

        featuretrain1= [featuretrain1;temptrain1];
        featuretrain2= [featuretrain2;temptrain2];
        featuretrain3= [featuretrain3;temptrain3];
        featuretrain4= [featuretrain4;temptrain4];

        featuretest1=  [featuretest1;temptest1];
        featuretest2 = [featuretest2;temptest2];
        featuretest3=  [featuretest3;temptest3];
        featuretest4 = [featuretest4;temptest4];

        temptrain1= [];
        temptrain2 = [];
        temptrain3= [];
        temptrain4 = [];

        temptest1= [];
        temptest2= [];
         temptest3= [];
        temptest4= [];
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


    numf= 20;
%     for ii=1:numf
    
    
    %         counter=counter+1;
    [fdr12,ind]= sort(fdr12,'descend');
    sel_ind12= ind(1:numf);
%     sel_ind12= ind(5:ii);

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
    sel_ind34= ind6(1:numf);
%     sel_ind34= ind6(1:ii);
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

        traindata=[featuretrain1(sel_ind12,:),featuretrain2(sel_ind12,:),featuretrain3(sel_ind34,:),featuretrain4(sel_ind34,:)];
        trainlabel=[ones(1,size(featuretrain1,2)),...
        2*ones(1,size(featuretrain2,2)),...
        3*ones(1,size(featuretrain3,2)),...
        4*ones(1,size(featuretrain4,2))];

            testdata= [featuretest1(sel_ind12,:),featuretest2(sel_ind12,:),featuretest3(sel_ind34,:),featuretest4(sel_ind34,:)];
            %testdata= [featuretest1,featuretest2,featuretest3,featuretest4];
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
        %[mdl]= fitcknn(traindata',trainlabel,'NumNeighbors',5);
          %% test tained classifier using test data
         %a=testdata(BestSol.Out.S,:);
         a=testdata;
             [output] = MultiClassSVMclassify(mdl,a);
           %[output] = predict(mdl,a');
    %        if isnan(output)
    %            output
    %        end
%         C= C+ confusionmat(testlabel,output);
        yakhoda=testlabel==output;
        accuracy=sum(yakhoda)/size(output,2);
%         c2=confusionmat(testlabel,output);
    %     accuracyF(iter,sub) = sum(diag(C)) / sum(C(:)) *100;
    %     accuracy1 = C(1,1) / sum(C(1,:)) *100;
    %     accuracy2 = C(2,2) / sum(C(2,:)) *100;
    %     accuracy3 = C(3,3) / sum(C(3,:)) *100;
    %     accuracy4 = C(4,4) / sum(C(4,:)) *100;
    %     
    %     disp(['iteration: ',num2str(iter)])

    %     disp(['total Accuracy subject fold 1 ',num2str(iter),': ',num2str(accuracyF),' %'])
    %     disp([' Accuracy1: ',num2str(accuracy1),' %'])
    %     disp([' Accuracy2: ',num2str(accuracy2),' %'])
    %     disp([' Accuracy3: ',num2str(accuracy3),' %'])
    %     disp([' Accuracy4: ',num2str(accuracy4),' %'])

%     accuracy = sum(diag(C)) / sum(C(:)) *100;
%     accuracy1 = C(1,1) / sum(C(1,:)) *100;
%     accuracy2 = C(2,2) / sum(C(2,:)) *100;
%     accuracy3 = C(3,3) / sum(C(3,:)) *100;
%     accuracy4 = C(4,4) / sum(C(4,:)) *100;
      disp(['total Accuracy subject ',num2str(subject(sub)),' for run ',num2str(run),': ',num2str(accuracy),' %'])
%     disp([' Accuracy1: ',num2str(accuracy1),' %'])
%     disp([' Accuracy2: ',num2str(accuracy2),' %'])
%     disp([' Accuracy3: ',num2str(accuracy3),' %'])
%     disp([' Accuracy4: ',num2str(accuracy4),' %'])
%       p0=accuracy;
%       s=0;
%       for i=1:4
%         s=s+C(:,i)*C(i,:);
%       end
%       pe=(s)/(C*C);
%       kappa=(p0-pe)/(1-pe);
      k=kappa(C);
%     accuracies=[accuracies,[accuracy,accuracy1,accuracy2,accuracy3,accuracy4]'];
      accuracies(:,sub)=accuracy;
      kappas(:,sub)=k;
%     accuracyF(:,sub) = accuracy;
%        acuuracyFDR(ii)=sum(diag(C))/sum(C(:))*100;
% if isnan(acuuracyFDR(ii))==0
%     if acuuracyFDR(ii)==100
%         acuuracyFDR(ii)=60;
%     end
%     plot(acuuracyFDR(1:ii),'r','linewidth',2)
%     hold on
%     plot(acuuracyFDR(1:ii),'ok','linewidth',2)
%     grid on
%     grid minor
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
%     ylabel('accuracy')
%     drawnow
% end
%     end
%     figure(ii)
    end
    kappasF(run,:)=kappas;
    accuracyF(run,:) = accuracies;
end

% boxplot(accuracyF,'PlotStyle','traditional')
% xlabel('Subjects')
% grid on
% ylim([0,100])

mean(mean(accuracyF))
sqrt(var(var(accuracyF)))
