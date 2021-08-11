function final_metrics = classification(data, n_run, test_option, p_train, k_cv, classifier_method, find_parameters, parameters, show_plot)

accuracy = zeros(1,n_run);
sensitivity = zeros(1,n_run);
specificity = zeros(1,n_run);
f_score = zeros(1,n_run);
mcc = zeros(1,n_run);

if find_parameters == 1 && (strcmp(classifier_method,'KNN') || strcmp(classifier_method,'SVM Linear') || strcmp(classifier_method,'SVM RBF'))
    parameters = get_parameters(data, classifier_method, show_plot);
end

disp(test_option);

if strcmp(test_option,'Holdout')
    
    for i = 1:n_run
        
        c = cvpartition(data.y, 'Holdout', 1-p_train, 'Stratify', true);
        idx_train = c.training;
        idx_test = c.test;

        % Training dataset
        train_data.X = data.X(:, idx_train);
        train_data.y = data.y(idx_train);
        train_data.dim = size(train_data.X,1);
        train_data.num_data = size(train_data.X,2);
        train_data.num_classes = data.num_classes;
        train_data.name = 'Dataset (training)';

        % Test dataset 
        test_data.X = data.X(:, idx_test);
        test_data.y = data.y(idx_test);
        test_data.dim = size(test_data.X,1);
        test_data.num_data = size(test_data.X,2);
        test_data.num_classes = data.num_classes;
        test_data.name = 'Dataset (test)';

        y_predicted = classifiers(data, train_data, test_data, classifier_method, parameters, show_plot);
        metrics = performance(test_data.y, y_predicted, data.num_classes, show_plot);

        accuracy(i) = metrics.accuracy;
        sensitivity(i) = metrics.sensitivity;
        specificity(i) = metrics.specificity;
        f_score(i) = metrics.f_score;
        mcc(i) = metrics.mcc;

    end
    
elseif strcmp(test_option,'Cross-Validation')
    
     c = cvpartition(data.y, 'KFold', k_cv, 'Stratify', true);
     
     for i = 1:c.NumTestSets
         
        idx_train = c.training(i);
        idx_test = c.test(i);

        % Training dataset
        train_data.X = data.X(:, idx_train);
        train_data.y = data.y(idx_train);
        train_data.dim = size(train_data.X,1);
        train_data.num_data = size(train_data.X,2);
        train_data.num_classes = data.num_classes;
        train_data.name = 'Dataset (training)';

        % Test dataset 
        test_data.X = data.X(:, idx_test);
        test_data.y = data.y(idx_test);
        test_data.dim = size(test_data.X,1);
        test_data.num_data = size(test_data.X,2);
        test_data.num_classes = data.num_classes;
        test_data.name = 'Dataset (test)';

        y_predicted = classifiers(data, train_data, test_data, classifier_method, parameters, show_plot);
        metrics = performance(test_data.y, y_predicted, data.num_classes, show_plot);

        accuracy(i) = metrics.accuracy;
        sensitivity(i) = metrics.sensitivity;
        specificity(i) = metrics.specificity;
        f_score(i) = metrics.f_score;
        mcc(i) = metrics.mcc;

     end     
   
end

final_metrics.accuracy = mean(accuracy);
final_metrics.accuracy_std = std(accuracy);

final_metrics.sensitivity = mean(sensitivity);
final_metrics.sensitivity_std = std(sensitivity);

final_metrics.specificity = mean(specificity);
final_metrics.specificity_std = std(specificity);

final_metrics.f_score = mean(f_score);
final_metrics.f_score_std = std(f_score);

final_metrics.mcc = mean(mcc);
final_metrics.mcc_std = std(mcc);

end
