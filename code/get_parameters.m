function parameters = get_parameters(data, classifier_method, show_plot)

n_runs = 1;
p_trn = 0.5;
disp("Finding appropriate parameters...");

% KNN - Find appropriate number of neighbours K
if strcmp(classifier_method,'KNN')
    
    n_k = 40;
    error = zeros(n_runs, n_k);

    for j = 1:n_runs

        idx = randperm(data.num_data);
        n_trn_samp = ceil(data.num_data * p_trn); % Number of training patterns

        % Define training dataset with the most discriminative features
        trn.X = data.X(:, idx(1:n_trn_samp));
        trn.y = data.y(idx(1:n_trn_samp));
        trn.dim = size(trn.X, 1);
        trn.num_data = size(trn.X, 2);

        % Define test dataset with the most discriminative features
        tst.X = data.X(:, idx(n_trn_samp+1:end));
        tst.y = data.y(idx(n_trn_samp+1:end));
        tst.dim = size(tst.X, 1);
        tst.num_data = size(tst.X, 2);

        for i = 1:n_k
            clear model
            model = knnrule(trn, i);
            ypred = knnclass(tst.X, model);
            error(j,i) = cerror(ypred, tst.y)*100;
        end
    end
    
    if n_runs > 1
        mu_error = mean(error);
        std_error = std(error);
    else
        mu_error = error;
        std_error = zeros(1,numel(error));
    end
     
    idx = find(mu_error == min(mu_error));
    k = idx(1);
    fprintf("K: %d", k);
    parameters.k_neighbors = k;
    
    if show_plot
        figure;
        errorbar(1:n_k, mu_error, std_error);
        hold on
        plot(k, min(mu_error), 'ro');
        xlabel('k');
        ylabel('Testing error (%)')
        grid on
    end
 
    
% SVM Linear - Find appropriate cost C
elseif strcmp(classifier_method,'SVM Linear') 
    
    c_pot = -10:10; % exponent values for the cost
    C = 2.^c_pot; % define different costs as powers of 2
    error = zeros(n_runs, numel(C)); % testing error matrix

    for n = 1:n_runs

        idx = randperm(data.num_data);
        n_trn = ceil(data.num_data * p_trn); % number of training patterns

        % Training dataset
        trn.X = data.X(:, idx(1:n_trn));
        trn.y = data.y(idx(1:n_trn));
        trn.dim = 2;
        trn.num_data = n_trn;

        % Test dataset
        tst.X = data.X(:, idx(n_trn+1:end));
        tst.y = data.y(idx(n_trn+1:end));
        tst.dim = 2;
        tst.num_data = data.num_data - n_trn;

        for co=1:numel(C)
            if data.num_classes == 2
                model = fitcsvm(trn.X',trn.y','KernelFunction','linear','BoxConstraint',C(co),'Solver','SMO');
            else 
                svm = templateSVM('KernelFunction','linear','BoxConstraint',C(co),'Solver','SMO');
                model = fitcecoc(trn.X',trn.y','Learners',svm,'Coding','onevsone');
            end
            ypred = predict(model,tst.X');
            error(n,co)= cerror(ypred', tst.y) * 100;
        end
    end

    if n_runs > 1
        mu_error = mean(error);
        std_error = std(error);
    else
        mu_error = error;
        std_error = zeros(1,numel(error));
    end
    
    if show_plot
        % Plot
        figure;
        plot(c_pot, mu_error,'o')
        ylabel('Testing error (%)')
        set(gca,'xtick', c_pot)
        set(gca,'xticklabel',strcat('2^', cellfun(@num2str, num2cell(c_pot),'UniformOutput',0)))
        hold on
        errorbar(c_pot, mu_error, std_error)
        xlim([c_pot(1) c_pot(end)]);
        grid on
    end

    % Best value
    idx_C = find(mu_error == min(mu_error));
    idx_C = idx_C(1);
    fprintf('\nBest C value = 2^%d\n', c_pot(idx_C));
    C = 2.^(c_pot(idx_C));
    parameters.C = C;


% SVM RBF - Find appropriate C and G
elseif strcmp(classifier_method,'SVM RBF') 
     
    c_pot = -15:20;
    g_pot = -20:15;
    C = 2.^c_pot;
    G = 2.^g_pot;
    error = zeros(n_runs, numel(C), numel(G)); 

    for n = 1:n_runs

        idx = randperm(data.num_data);
        n_trn = ceil(data.num_data * p_trn); % number of training patterns

        % Training dataset
        trn.X = data.X(:, idx(1:n_trn));
        trn.y = data.y(idx(1:n_trn));
        trn.dim = 2;
        trn.num_data = n_trn;

        % Test dataset
        tst.X = data.X(:, idx(n_trn+1:end));
        tst.y = data.y(idx(n_trn+1:end));
        tst.dim = 2;
        tst.num_data = data.num_data - n_trn;

        for co = 1:numel(C)
            for go = 1:numel(G)
                if data.num_classes==2
                    model = fitcsvm(trn.X',trn.y','KernelFunction','rbf','BoxConstraint',C(co),'KernelScale',sqrt(1/2*G(go)),'Solver','SMO');   
                else
                    svm = templateSVM('KernelFunction','rbf','BoxConstraint',C(co),'KernelScale',sqrt(1/(2*G(go))),'Solver','SMO');
                    model = fitcecoc(trn.X',trn.y','Learners',svm,'Coding','onevsone');
                end

                ypred = predict(model,tst.X');
                error(n,co,go)= cerror(ypred', tst.y) * 100;
            end
        end

    end
    
    if n_runs > 1
        mu_error = squeeze(mean(error));
    else
        mu_error = squeeze(error);
    end
    
    [idx_C, idx_G] = find(mu_error == min(min(mu_error)));
    idx_C = idx_C(1);
    idx_G = idx_G(1);
    C = 2.^(c_pot(idx_C));
    G = 2.^(g_pot(idx_G));
    parameters.C = C;
    parameters.G = G;
    fprintf('\nBest C value = 2^%d\n', c_pot(idx_C));
    fprintf('\nBest G value = 2^%d\n', g_pot(idx_G));
    
    if show_plot
        % PLOT
        figure;
        contourf(g_pot, c_pot, mu_error);
        xlabel('Gamma')
        ylabel('Cost')
        set(gca,'xtick',g_pot([1:5:numel(g_pot)]))
        set(gca,'xticklabel',strcat('2^',cellfun(@num2str,num2cell(g_pot([1:5:numel(g_pot)])),'UniformOutput',0)))
        set(gca,'ytick',c_pot([1:5:numel(c_pot)]))
        set(gca,'yticklabel',strcat('2^',cellfun(@num2str,num2cell(c_pot([1:5:numel(c_pot)])),'UniformOutput',0)))
        colorbar
        hold on
        plot(g_pot(idx_G),c_pot(idx_C),'rx','markersize',8,'linewidth',2)
    end
    
end

end