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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
t1 = theta(1);
t2 = theta(2:end);

h  = sigmoid(X * theta);
J =  (((y' * log(h)) + (1-y)' * log(1 - h))/-m) + lambda/(2*m) * sum(t2.^2);

g1 = t1 - sum(X'(1,:) * (h-y))/m;
g2 = t2 * (1 - lambda/m) - X'(2:end,:) * (h-y)/m;
grad = [g1; g2];






% =============================================================

end
