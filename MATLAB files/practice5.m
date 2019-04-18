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
plot(XA(YA==0), 'ro');
hold on;
plot(XA(YA==1), 'b^');
hold off;

figure;
plot(XB(YB==1), 'ro');
hold on;
plot(XB(YB==2), 'b^');
hold off;


%% b)
close all
learner = logisticClassify2(); % create "blank" learner 
learner = setClasses(learner, unique(YA)); % define class labels using YA or YB 
wts = [0.5 1 -0.25]; % TODO: fill in values 
learner = setWeights(learner, wts); % set the learner's parameters
plot2DLinear(learner, XA, YA);

learner2 = logisticClassify2(); % create "blank" learner 
learner2 = setClasses(learner2, unique(YB)); % define class labels using YA or YB 
wts = [0.5 1 -0.25]; % TODO: fill in values 
learner2 = setWeights(learner2, wts); % set the learner's parameters
plot2DLinear(learner2, XB, YB);

%% c)
YApred = predict(learner, XA);
YBpred = predict(learner2, XB);

figure;
plotClassify2D(learner, XA, YA);

figure;
plotClassify2D(learner2, XB, YB);

errA = err(learner, XA, YA);
errB = err(learner2, XB, YB);

%% e)
e = train(learner, XA, YA);

