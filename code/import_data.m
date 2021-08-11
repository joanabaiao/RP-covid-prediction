function data = import_data(scenario)

if strcmp(scenario,'A')
    
    path = fullfile(pwd, 'Preprocessing','datasetA.csv');
    
    dataA = readtable(path);
    column_names = dataA.Properties.VariableNames; 
    
    data.X = table2array(dataA(:,1:end-1))'; 
    data.dim = size(data.X,1);
    data.num_data = size(data.X,2);
    data.names = column_names(:,1:end-1);
    data.name = 'Dataset A';
    data.num_classes = 2;
    data.y = table2array(dataA(:,end))'; % 1 - released; 2 - others
    
elseif strcmp(scenario,'B')
    
    path = fullfile(pwd, 'Preprocessing','datasetBC.csv');

    dataB = readtable(path);
    column_names = dataB.Properties.VariableNames; 
    
    data.X = table2array(dataB(:,1:end-1))'; 
    data.dim = size(data.X,1);
    data.num_data = size(data.X,2);
    data.names = column_names(:,1:end-1);
    data.name = 'Dataset B';
    data.num_classes = 2;
    
    state = table2array(dataB(:,end))';
    idx1 = find(state == 2); % Deceased
    idx2 = find(state ~= 2); % Others
    data.y(idx1) = 1; % 1 - Deceased; 2 - others
    data.y(idx2) = 2;
    
elseif strcmp(scenario,'C')
    
    path = fullfile(pwd, 'Preprocessing','datasetBC.csv');

    dataC = readtable(path);
    column_names = dataC.Properties.VariableNames; 
    data.X = table2array(dataC(:,1:end-1))'; 
    data.dim = size(data.X,1);
    data.num_data = size(data.X,2);
    data.names = column_names(:,1:end-1);
    data.name = 'Dataset C';
    data.num_classes = 3;

    state = table2array(dataC(:,end))';
    idx1 = find(state == 0); % Released
    idx2 = find(state == 1); % Isolated
    idx3 = find(state == 2); % Deceased
    data.y(idx1) = 1; 
    data.y(idx2) = 2;
    data.y(idx3) = 3;
    
end



end