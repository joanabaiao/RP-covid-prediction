function metrics = performance(y_test, y_predicted, num_classes, show_plot)

% ------------ CONFUSION MATRIX ------------
cm = confusionmat(y_predicted, y_test); 
TP = zeros(1,num_classes); 
TN = zeros(1,num_classes);
FP = zeros(1,num_classes); 
FN = zeros(1,num_classes); 

if show_plot == true  
    figure;
    confusionchart(cm)
    title(strcat("Confusion matrix"))
end
    
for i = 1:num_classes
    TP(i) = cm(i,i);
    FP(i) = sum(cm(:,i)) - cm(i,i);
    FN(i) = sum(cm(i,:)) - cm(i,i);
    TN(i) = sum(cm(:)) - TP(i) - FP(i) - FN(i);
end

% -------------- METRICS --------------
accuracy = zeros(1,num_classes);
sensitivity = zeros(1,num_classes);
specificity = zeros(1,num_classes);
f_score = zeros(1,num_classes);
mcc = zeros(1,num_classes);

beta = 1;
for i=1:num_classes
    accuracy(i) = (TP(i)+TN(i))/(TP(i)+TN(i)+FP(i)+FN(i))*100;
    sensitivity(i) = (TP(i)/(TP(i)+FN(i))*100);
    specificity(i) = (TN(i)/(TN(i)+FP(i))*100);
    mcc(i) = (TP(i)*TN(i)-FP(i)*FN(i))/sqrt((TP(i)+FP(i))*(TP(i)+FN(i))*(TN(i)+FP(i))*(TN(i)+FN(i)))*100;  
    
    R = TP(i)/(TP(i)+FN(i));
    P = TP(i)/(TP(i)+FP(i));
    f_score(i) = ((beta^2+1)*P*R)/(beta^2*P+R)*100;
end

metrics.accuracy = mean(accuracy);
metrics.sensitivity = mean(sensitivity);
metrics.specificity = mean(specificity);
metrics.f_score = mean(f_score);
metrics.mcc = mean(mcc);

end
