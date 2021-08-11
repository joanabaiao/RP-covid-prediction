function reduced_data = feature_reduction(reduction_method, data, dim, kaiser_criterion, show_pareto, show_eigenvalues, show_reduced_data)

if strcmp(reduction_method,"PCA")
    
    model = pca(data.X);
    
    % Select number of dimensions 
    if kaiser_criterion == 1 || dim < 0 || dim > data.dim   
        new_dim = length(find(model.eigval(:)>1)); % Kaiser Criterion
    else
        new_dim = dim;
    end
    
    % Plot: Eig. Values
    if show_eigenvalues == 1
        
        figure; 
        plot(model.eigval, 'o-', 'MarkerFaceColor','#0072BD', 'LineWidth', 1.5); 
        hold on
        plot(ones(data.dim,1),'r', 'LineWidth', 1.5)
        hold on
        plot(new_dim,1,'k*','MarkerSize',10, 'LineWidth',2)
        title('Principal Component vs. Eig Value'); 
        xlabel('Principal Component'); ylabel('Eig Value'); grid on;        
    end
    
    % Plot: Variance
    if show_pareto == 1
   
        variance = zeros(1, data.dim);
        for i=1:data.dim
            variance(i) = cumsum(model.eigval(i))./sum(model.eigval)*100;
        end

        figure; 
        bar(categorical(1:data.dim), variance);
        hold on
        plot(cumsum(model.eigval(1:data.dim))./sum(model.eigval)*100, 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor','r');
        title('Pareto diagram'); 
        xlabel('Principal Component'); ylabel('Explained variance (%)'); grid on;
        ylim([0 105]);
        
    end
    
    model = pca(data.X, new_dim);
          
elseif strcmp(reduction_method, "LDA")
     
    % Select number of dimensions
    new_dim = data.num_classes - 1;
    model = lda(data, new_dim);

end 

fprintf('\n%s: %d features\n', reduction_method, new_dim);

reduced_data = linproj(data, model);
reduced_data.dim = size(reduced_data.X,1);
reduced_data.num_data = size(reduced_data.X,2);
reduced_data.name = data.name;
reduced_data.num_classes = data.num_classes;

% Plot: Reduced data
if show_reduced_data && new_dim >= 1 && new_dim <= 3
    
    figure;
    ppatterns(reduced_data)
    title(strcat("Feature reduction - ", string(new_dim), " components"));
    legend();
    
end
    

end




