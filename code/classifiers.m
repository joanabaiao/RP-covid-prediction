function y_predicted = classifiers(data, train_data, test_data, classifier_method, parameters, show_plot)

% -------- MINIMUM DISTANCE CLASSIFIER - EUCLIDEAN DISTANCE --------
if strcmp(classifier_method, "MDC - Euclidean")
    
    mu = zeros(train_data.dim, train_data.num_classes);
    g = zeros(1, test_data.num_classes);
    y_predicted = zeros(1, test_data.num_data);
    
    for i = 1:train_data.num_classes
        idx = find(train_data.y == i);
        mu(:,i) = mean(train_data.X(:,idx), 2);
    end
    
    for i = 1:test_data.num_data
        for m = 1:size(mu, 2)
            g(m) = mu(:,m)' * test_data.X(:,i) - 0.5 * (mu(:,m))' * mu(:,m);
        end
        y_predicted(i) = find(g == max(g));  
    end

% -------- MINIMUM DISTANCE CLASSIFIER - MAHALANOBIS DISTANCE --------
elseif strcmp(classifier_method, "MDC - Mahalanobis")
    
    mu = zeros(train_data.dim, train_data.num_classes);
    c = zeros(train_data.dim);
    g = zeros(1, test_data.num_classes);
    y_predicted = zeros(1, test_data.num_data);
    
    for i = 1:train_data.num_classes
        idx = find(train_data.y == i);
        mu(:,i) = mean(train_data.X(:,idx), 2);  
        c = c + cov(train_data.X(:,idx)');
    end
    c = c./train_data.num_classes;
    c_inverse = c^-1;
    
    for i = 1:test_data.num_data
        for m = 1:size(mu, 2)
            g(m) = mu(:,m)' * c_inverse * test_data.X(:,i) - 0.5 * (mu(:,m))' * c_inverse * mu(:,m);
        end
        y_predicted(i) = find(g == max(g));  
    end

% ------------------------ FISHER CLASSIFIER ------------------------    
elseif strcmp(classifier_method, "Fisher LDA")
    
    if train_data.num_classes == 2  
        fisher_model = fld(train_data);
        y_predicted = linclass(test_data.X, fisher_model);  

    else
        t = templateDiscriminant('DiscrimType','linear');
        fisher_model = fitcecoc(train_data.X', train_data.y','Coding','onevsall','Learners',t);
        y_predicted = predict(fisher_model, test_data.X');
    end

% ------------------------ BAYES CLASSIFIER ------------------------
elseif strcmp(classifier_method, "Bayes")
    
    l = zeros(1, train_data.num_classes);
    for i = 1:train_data.num_classes
         idx = find(train_data.y == i);      
         trn.X = train_data.X(:,idx);
         trn.y = train_data.y(idx);
         model.Pclass{i} = mlcgmm(trn);  
         l(i) = length(idx); 
    end

    model.Prior = l/sum(l);
    model.fun = 'bayescls';

    y_predicted = bayescls(test_data.X, model);

% ---------------- K-NEAREST NEIGHBORS ALGORITHM ----------------
elseif strcmp(classifier_method, "KNN")
    
    k_neighbors = parameters.k_neighbors;
     
    model = knnrule(train_data, k_neighbors);
    y_predicted = knnclass(test_data.X, model);
 
% ------------------------ SVM LINEAR ---------------------------
elseif strcmp(classifier_method, "SVM Linear")
    
    C = parameters.C;
    
    if data.num_classes == 2
        svm_model = fitcsvm(train_data.X', train_data.y','KernelFunction','linear','BoxConstraint',C,'Solver','SMO');
    else 
        svm = templateSVM('KernelFunction','linear','BoxConstraint',C,'Solver','SMO');
        svm_model = fitcecoc(train_data.X', train_data.y','Learners',svm,'Coding','onevsone'); 
    end
    [y_predicted, dfce] = predict(svm_model, test_data.X');

% -------------------- SVM NON-LINEAR (RBF) --------------------
elseif strcmp(classifier_method, "SVM RBF")
    
    C = parameters.C;
    G = parameters.G;
    
    if data.num_classes==2
        svm_model = fitcsvm(train_data.X', train_data.y','KernelFunction','rbf','BoxConstraint', C,'KernelScale',sqrt(1/2*G),'Solver','SMO');   
    else
        svm = templateSVM('KernelFunction','rbf','BoxConstraint', C,'KernelScale',sqrt(1/(2*G)),'Solver','SMO');
        svm_model = fitcecoc(train_data.X', train_data.y','Learners',svm,'Coding','onevsone');
    end
    [y_predicted, dfce] = predict(svm_model, test_data.X');   
end


%% PLOT 

if show_plot   
    
    if data.num_classes == 2 && (strcmp(classifier_method, "MDC - Euclidean") || strcmp(classifier_method, "MDC - Mahalanobis") || strcmp(classifier_method, "Fisher LDA"))

        idx1 = find(train_data.y == 1);
        idx2 = find(train_data.y == 2);
        mu1 = mean(train_data.X(:,idx1),2);
        mu2 = mean(train_data.X(:,idx2),2);
        avg_mu = (mu1 + mu2)/2;
        
        if strcmp(classifier_method, "MDC - Euclidean")
            w = mu1' - mu2';
            b = -0.5 * (mu1'*mu1 - mu2'*mu2);

        elseif strcmp(classifier_method, "MDC - Mahalanobis")           
            w = (mu1' - mu2') * c_inverse;
            b = -0.5 * (mu1' * c_inverse * mu1 - mu2' * c_inverse * mu2);

        elseif strcmp(classifier_method, "Fisher LDA")
            w = fisher_model.W;
            b = fisher_model.b;
        end

         % PLOT 1 DIMENSION
        if train_data.dim == 1

            figure;
            ppatterns(train_data);
            hold on
            plot(mu1, 0, 'gx', 'MarkerSize', 10, 'LineWidth', 2)
            plot(mu2, 0, 'go', 'MarkerSize', 10, 'LineWidth', 2)
            plot(avg_mu, 0, 'gs', 'Markersize', 10, 'LineWidth', 2)
            legend('class 1', 'class 2', 'mean class 1', 'mean class 2', 'Average Mean')
            title(strcat(classifier_method, " - 1D"))

        % PLOT 2 DIMENSION
        elseif train_data.dim == 2

            figure;
            ppatterns(train_data);
            hold on   
            plot(mu1(1), mu1(2), 'gx', 'MarkerSize', 10, 'LineWidth', 2)
            plot(mu2(1), mu2(2), 'go', 'MarkerSize', 10, 'LineWidth', 2)
            plot([mu1(1) mu2(1)], [mu1(2) mu2(2)], 'black--', 'MarkerSize', 10, 'LineWidth', 2)
            plot(avg_mu(1), avg_mu(2), 'gs', 'Markersize', 10, 'LineWidth', 2)
            pline(w, b, 'g-') 
            legend('class 1', 'class 2', 'mean class 1', 'mean class 2', 'Line between means', 'Average Mean', 'Hyperplane')
            title(strcat(classifier_method, " - 2D "))

        % PLOT 3 DIMENSION
        elseif train_data.dim == 3

            figure;
            ppatterns(train_data);
            hold on  
            plane3(w,b)
            legend('class 1', 'class 2','Hyperplane')
            title(strcat(classifier_method, " - 3D"))
        end
        
        
    elseif (strcmp(classifier_method, "KNN") || strcmp(classifier_method, "Bayes")) && train_data.dim == 2
        
        figure;
        ppatterns(train_data);
        pboundary(model);
        title(classifier_method);
        
    % SVM Classifiers - ROC curve
    elseif strcmp(classifier_method, "SVM Linear") || strcmp(classifier_method, "SVM RBF")
        
        [FP, FN] = roc(dfce, test_data.y);
        TP = 1-FN;

        figure;
        stairs(FP, TP, 'Linewidth', 1.5);
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title('ROC Curve');
        text(0.5, 0.5, 'AUC', 'FontWeight', 'bold');
        text(0.48, 0.45, num2str(abs(trapz(FP,TP))));
    end
    
end



end