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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% ---------- Compute the cost function (vectorized) ------------
H = sigmoid(X * theta);

temp = (-y .* log(H)) - ((1 - y) .* log(1 - H));
J = (1 / m) * sum(temp);


% ------------ Compute the gradient (vectorized) --------------
H = H - y;   % Vector of h(x) - y;
grad = (1 / m) * (X' * H);

% =============================================================

grad = grad(:);


%{

% ------------------------- Compute J -------------------------
for i = 1:m,
	hypothesis = sigmoid(X(i, :) * theta);
	J = J + (-y(i)*log(hypothesis) - (1 - y(i))*log(1 - hypothesis));
end

J = (1 / (m)) * J;


% ---------------------- Compute Gradient -----------------------

for i = 1:m,
	% Grab one training example
	x = X(i, :);
	grad = grad + ((sigmoid(x * theta) - y(i)) * x');
end

grad = (1 / m) * grad;
%}

end
