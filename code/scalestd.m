function data_scaled = scalestd(data)

mu = mean(data.X,2);
s = std(data.X,[],2);

data_scaled = data;
data_scaled.X = (data.X - repmat(mu,1,data.num_data)) ./ repmat(s,1,data.num_data);

end