close all
clear


iris = load('data/iris.txt'); 
pi = randperm( size (iris ,1)); 
Y = iris(pi ,5); X = iris(pi ,1:2);

%% a)
classes = unique(Y);
figure;
gscatter(X(:,1),X(:,2),Y)
%% b,c 
for k=[1, 3, 10, 30]
    learner = knnClassify(k, X, Y);
    class2DPlot(learner, X, Y);
    title(sprintf('knnClassify Decision Regions with k=%.2f',k));
end

%%%
% The lower k values are again more complex, with the higher k values
% ommiting some of the k data points from the correct regions.

%% d
testp = randperm(size(iris,1), ceil(size(iris,1)/5));
trainp = setdiff(1:size(iris,1), testp);

training = iris(testp, :);
testing = iris(trainp, :);

kvalues = [1, 2, 5, 10, 50, 100, 200];
err = zeros(length(kvalues), 1);

for i = 1:length(kvalues)
    learner = knnClassify(kvalues(i), training(:, 1:4), training(:, 5));
    Yhat = predict(learner, testing(:, 1:4));
    
    err(i) = sum(Yhat(:) ~= testing(:,5));
end

plot(kvalues, err, 'b*')
title('Perfomance of different k values');

%%%
% k=1 overfits and k=200 overfit, k=4 appears to give the best results

