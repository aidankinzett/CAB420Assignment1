function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
[n,d] = size(X);
if (d~=2) 
  error('Sorry -- plot2DLogistic only works on 2D data...'); 
end

classes = obj.classes;

% plot data
gscatter(X(:,1),X(:,2),Y)
hold on
axis manual

% find boundary line

x = linspace(-2,2);
y = linspace(-2,2);

f = @(x) -(obj.wts(1)+x*obj.wts(2))/obj.wts(3);
y = f(x);

% plot boundary line
plot(x,y,'g--','LineWidth',2,'DisplayName','Boundary')
hold off