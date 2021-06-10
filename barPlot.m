% label=zeros(1,10);
% label={};
% for i=1:9
%     label(1,i)={'sub'+num2str(i)};
%     if i==10
%         label(i)='Ave';
%     end
% end
% label

accuraies=[73.5651,77.1228,77.2572  ; 77.4958,77.9364,77.9768 ;  85.5685,88.3243,88.2272 ; 70.4296, 72.8140,74.3043 ; 55.0560, 58.4818,57.5350 ;  75.4529,76.7973,82.2116 ;  72.4163,72.8495,78.0782 ;  85.0701,86.3499,85.8367 ;91.4801,93.7598,94.9994;76.2816, 78.27, 79.6029];


X = categorical({'Sub 1','Sub 2','Sub 3','Sub 4', 'Sub 5','Sub 6','Sub 7','Sub 8','Sub 9',' Ave'});
X = reordercats(X,{'Sub 1','Sub 2','Sub 3','Sub 4', 'Sub 5','Sub 6','Sub 7','Sub 8','Sub 9',' Ave'});

bar(X,accuraies)
grid on
grid minor
ylabel('Accuracy')
legend('OVR-CSP','OVO-CSP','OVR & OVO CSP')