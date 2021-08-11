function selected_data = feature_selection(selection_method, data, corr_threshold, n_features, show_heatmap)

if strcmp(selection_method, "Correlation")
      
    R = corrcoef(data.X'); % Correlation matrix  
    fprintf('\nCorrelation threshold: %.2f\n', corr_threshold);
    
    if show_heatmap == 1
        figure;
        heatmap(data.names,data.names,abs(R));
    end

    removed_idx = [];
    for i = 1:size(R,1)
        for j = i:size(R,2)
            if abs(R(i,j)) >= corr_threshold && i ~= j && ~ismember(i,removed_idx)
                removed_idx = [removed_idx j];
            end
        end
    end
    
    selected_features = data.X(setdiff(1:end, removed_idx), :);
    selected_names = data.names(setdiff(1:end, removed_idx));

    
elseif strcmp(selection_method, "Kruskal-Wallis")
    
    fprintf('\nNumber of features: %d\n', n_features);   
    
    rank = cell(data.dim,2); 
    for i = 1:data.dim
        [~,tbl,~] = kruskalwallis(data.X(i,:), data.y, 'off');
        rank{i,1} = data.names{i};
        rank{i,2} = tbl{2,5};   
    end   
    
    [~, idx] = sort([rank{:,2}], 2, 'descend'); % +imp -> -imp
            
    selected_features = data.X(idx(1:n_features), :);
    selected_names = data.names(idx(1:n_features));

    feature_ranking = [sprintf('\nK-W Feature ranking:\n')];
    for i = 1:data.dim
       feature_ranking = [feature_ranking sprintf('%s ---> %.4f\n', rank{idx(i),1}, rank{idx(i),2})];
    end
    disp(feature_ranking);
    
elseif strcmp(selection_method, "Kruskal-Wallis + Correlation")
    
    fprintf('\nNumber of features: %d\n', n_features);  
    fprintf('Correlation threshold: %.2f\n', corr_threshold);
    
    % Kruskal-Wallis
    rank = cell(data.dim,2); 
    for i = 1:data.dim
        [~,tbl,~] = kruskalwallis(data.X(i,:), data.y, 'off');
        rank{i,1} = data.names{i};
        rank{i,2} = tbl{2,5};   
    end
    
    [~, idx] = sort([rank{:,2}], 'descend'); % +imp -> -imp
            
    selected_features = data.X(idx(1:n_features), :);
    selected_names = data.names(idx(1:n_features));
    
    feature_ranking = [sprintf('\nK-W Feature ranking:\n')];
    for i = 1:data.dim
       feature_ranking = [feature_ranking sprintf('%s ---> %.2f\n', rank{idx(i),1}, rank{idx(i),2})];
    end
    disp(feature_ranking);
    
    % Correlation
    R = corrcoef(selected_features'); 
    
    if show_heatmap == 1
        figure;
        heatmap(data.names,data.names,abs(R));
    end
    
    removed_idx = [];
    for i = 1:size(R,1)
        for j = i:size(R,2)
            if abs(R(i,j)) >= corr_threshold && i ~= j && ~ismember(i,removed_idx)
                removed_idx = [removed_idx j];
            end
        end
    end
    
    selected_features = selected_features(setdiff(1:end, removed_idx), :);
    selected_names = selected_names(setdiff(1:end, removed_idx));
    
end

selected_data = data;
selected_data.X = selected_features;
selected_data.y = data.y;
selected_data.names = selected_names;
selected_data.dim = size(selected_data.X, 1);

end


