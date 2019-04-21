%% set up data
clear
close all
iris=load('data/iris.txt'); % load the text file 
X = iris(:,1:2); Y=iris(:,end); % get first two features 
[X, Y] = shuffleData(X,Y); % reorder randomly 
X = rescale(X); % works much better for rescaled data 
XA = X(Y<2,:); YA=Y(Y<2); % 
XB = X(Y>0,:); YB=Y(Y>0); % 

%% a)
figure;
hold on;
X0 = XA(YA==0, 1:end);
scatter(X0(:, 1), X0(:, 2));
X1 = XA(YA==1, 1:end);
scatter(X1(:, 1), X1(:, 2));
title('Class 0 and 1');
legend('Class 0', 'Class 1');
hold off;

figure;
hold on;
X1 = XB(YB==1, 1:end);
scatter(X1(:, 1), X1(:, 2));
X2 = XB(YB==2, 1:end);
scatter(X2(:, 1), X2(:, 2));
title('Class 1 and 2');
legend('Class 1', 'Class 2');
hold off;


%% b)
close all
learner = logisticClassify2(); % create "blank" learner 
learner = setClasses(learner, unique(YA)); % define class labels using YA or YB 
wts = [0.5 1 -0.25]; 
learner = setWeights(learner, wts); % set the learner's parameters
plot2DLinear(learner, XA, YA);
title('Class 0 and 1');

figure;
learner2 = logisticClassify2(); % create "blank" learner 
learner2 = setClasses(learner2, unique(YB)); % define class labels using YA or YB 
wts = [0.5 1 -0.25]; 
learner2 = setWeights(learner2, wts); % set the learner's parameters
plot2DLinear(learner2, XB, YB);
title('Class 1 and 2');


%% c)
YApred = predict(learner, XA);
YBpred = predict(learner2, XB);

figure;
plotClassify2D(learner, XA, YA);
title('Error rate on data set A')

figure;
plotClassify2D(learner2, XB, YB);
title('Error rate on data set B')

errA = err(learner, XA, YA)
errB = err(learner2, XB, YB)

%% e) Complete train.m

%% f)
% Train Class A
train(learner, XA, YA, 'stopIter', 200, 'stepsize', 0.5);
legend('Error Rate', 'Surrogate Loss');
% Plot final converged classifier decision boundaries.
figure();
plotClassify2D(learner, XA, YA);

%% Train Class B
train(learner, XB, YB, 'stopIter', 200, 'stepsize', 0.5);
legend('Error Rate', 'Surrogate Loss');
% Plot final converged classifier decision boundaries.
figure();
plotClassify2D(learner, XB, YB);

%% g)
clc
% Train Set A
train_in_batches(learner, XA, YA,11, 'stopIter', 100, 'stepsize', 0.1);
legend('Error Rate', 'Surrogate Loss');
% Plot final converged classifier decision boundaries.
figure();
plotClassify2D(learner, XA, YA);

%% Train Set B
figure
train_in_batches(learner, XB, YB,11, 'stopIter', 100, 'stepsize', 0.1);
legend('Error Rate', 'Surrogate Loss');
% Plot final converged classifier decision boundaries.
figure();
plotClassify2D(learner, XB, YB);



