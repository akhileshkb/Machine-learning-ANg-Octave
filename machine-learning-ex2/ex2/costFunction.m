function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
one = zeros(size(y));
one = one.+1;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
g= sigmoid(X*theta);
%disp(g)
%g2 = sigmoid(one - X*theta);
%disp(g2)
J = ((-y.*log(g))-((one-y).*log(one - g)));
%disp(J);
J = sum(J)/m;

%grad(1) = sum((g - y).*X(:,1))/m;

for iter = 1:length(theta)
	grad(iter) = sum((g - y).*X(:,iter))/m;




% =============================================================

end
