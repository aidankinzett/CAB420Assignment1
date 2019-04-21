function mini_batches = create_mini_batches(obj, X,y, batch_size )

%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

data_values = [X,y];

data_values = data_values(randperm(size(data_values, 1)), :); % shuffle your data
n_mini_batches = ceil(size(data_values, 1)/batch_size)  - 1; %  based on your data and the batch size compute the number of batches
mini_batches = zeros(batch_size,3,n_mini_batches);

for i = 1:n_mini_batches
    disp(i)
   %TODO extract the minibatch values
   mini_batches(:,:,i) = data_values(i+((i-1)*batch_size):batch_size+((i-1)*(batch_size+1)), :);
end

end