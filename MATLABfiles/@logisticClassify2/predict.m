function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

% (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
% (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );

% 1)

wts = obj.wts;

pred = (sign(wts(1) + wts(2).*(Xte(:,1)) + wts(3).*(Xte(:,2))))+1;

pred(pred==0)=1;

Yte = obj.classes(pred);
