function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
one = zeros(size(y)).+1;
theta1 = theta(2:length(theta));
theta_reg = [0;theta(2:end, :);];

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

g = sigmoid(X*theta);
J = ((-y.*log(g))-((one-y).*log(one - g)));
J = sum(J)/m;
J = J + lambda*sum(theta1.*theta1)/m;

for iter = 1:length(theta)
  if iter == 1
    grad(iter) = sum((g - y).*X(:,iter))/m;
  else
    grad(iter) = (sum((g - y).*X(:,iter))/m) + lambda*theta(iter)/m;
  endif
endfor



% =============================================================

end
