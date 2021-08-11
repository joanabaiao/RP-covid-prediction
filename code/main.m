% RP
% clc; clear all; close all;
function metrics = main(scenario, selection_method, show_heatmap, n_features, corr_threshold, ...
                    reduction_method, dim, kaiser_criterion, show_pareto, show_eigenvalues, show_reduced_data,...
                    classifier_method, k_neighbors, C, G, find_parameters, ...
                    test_option, k_cv, p_train, show_plot)
%% Import data

data = import_data(scenario);
disp(scenario);

data = scalestd(data);

%% Feature Selection Methods

% selection_method = "Kruskal-Wallis";
% selection_method = "Correlation";
% selection_method = "Kruskal-Wallis + Correlation";

% show_heatmap = 1;
% corr_threshold = 0.90;  
% n_features = 5;

selected_data = feature_selection(selection_method, data, corr_threshold, n_features, show_heatmap);

%% Feature Reduction Methods

%reduction_method = "PCA";
%reduction_method = "LDA";

% dim = 2;
% kaiser_criterion = 0;
% show_pareto = 1;
% show_eigenvalues = 1;
% show_reduced_data = 1;

reduced_data = feature_reduction(reduction_method, selected_data, dim, kaiser_criterion, show_pareto, show_eigenvalues, show_reduced_data);

%% Classification

% test_option = "Holdout";
% test_option = "CV";

% classifier_method = "MDC - Euclidean";
% classifier_method = "MDC - Mahalanobis";
% classifier_method = "Fisher LDA";
% classifier_method = "Bayes";
% classifier_method = "KNN";
% classifier_method = "SVM Linear";
% classifier_method = "SVM RBF";

n_runs = 5;
% p_train = 0.7;
% k_cv = 5;
% show_plot = true;

% k_neighbors = 7;
% C = 10;
% G = 3;
% find_parameters = 0;

parameters.k_neighbors = k_neighbors;
parameters.C = C;
parameters.G = G;

metrics = classification(reduced_data, n_runs, test_option, p_train, k_cv, classifier_method, find_parameters, parameters, show_plot);

fprintf("\nMETRICS:");
fprintf("\n* Accuracy: (%.2f ± %.2f)%%", metrics.accuracy, metrics.accuracy_std);
fprintf("\n* Sensitivity: (%.2f ± %.2f)%%", metrics.sensitivity, metrics.sensitivity_std);
fprintf("\n* Specificity: (%.2f ± %.2f)%%", metrics.specificity, metrics.specificity_std);
fprintf("\n* F-score: (%.2f ± %.2f)%%", metrics.f_score, metrics.f_score_std);
fprintf("\n* MCC: (%.2f ± %.2f)%%\n\n", metrics.mcc, metrics.mcc_std);

end