%% Set up Data
clear
mTrain=load('data/mTrainData.txt');
mTest = load('data/mTestData.txt');

Xte=mTest(:,1); Yte=mTest (:,2);

%% b)

for k=[1 2 3 5 10 50]
    figure
    Xtr=mTrain(:,1); Ytr=mTrain (:,2);
    plot(Xtr(1:20),Ytr(1:20),'bo'); 
    hold on
    title(sprintf('knn Regression where k=%.2f', k));

    learner = knnRegress(k,Xtr,Ytr);
    xline = [0:.01:1]';
    yline = predict( learner , xline);
    plot(xline, yline, 'r-');
    legend('Training Data','knn Regression');
end

%%%
% Lower k values produce more complex functions

%% c)
