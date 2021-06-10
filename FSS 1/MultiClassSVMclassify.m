function [output] = MultiClasspredict(mdl,datatest)
for i=1:size(datatest,2)
    x= datatest(:,i);
    y= predict(mdl.svm1,x');
    if y==1
        output(i) =1;
    elseif y==2
        y= predict(mdl.svm2,x');
        if y==1
            output(i) =2;
        elseif y==2
            y= predict(mdl.svm3,x');
            if y==1
                output(i) =3;
            elseif y==2
                y= predict(mdl.svm4,x');
                if y==1
                    output(i) =4;
                elseif y==2
                    output(i) =nan;
                end
            end
        end
    end
end
end