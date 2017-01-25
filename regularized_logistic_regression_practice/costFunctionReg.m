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


% ------- Compute the (regularized) cost function (vectorized) ---------
H = sigmoid(X * theta);

temp = (-y .* log(H)) - ((1 - y) .* log(1 - H));
J = (1 / m) * sum(temp);
J = J + (lambda / (2 * m)) * sum(theta .^ 2);

% ----------- Compute the (regularized) gradient (vectorized) ----------
H = H - y;   % Vector of h(x) - y;
grad = (1 / m) * (X' * H);

temp = theta;
temp(1) = 0;

grad = grad + (lambda / m) * temp;

%{}

% ---------------------- Calculate Cost ----------------------

for i = 1:m,
	hypothesis = sigmoid(X(i, :) * theta);
	J = J + (1/m) * (-y(i)*log(hypothesis) - (1 - y(i))*log(1 - hypothesis));
end

n = length(theta);
for j = 2:n,
	J = J + (lambda / (2 * m)) * (theta(j))^2;
end


% --------------------- Calculate Gradient ----------------------
for k = 1:m,
	% Grab one training example
	x = X(k, :);

	% Update gradient vector
	grad = grad + (sigmoid(x * theta) - y(k)) * x';
end

grad = (1 / m) * grad;


% Adjust gradient vector for regulariation
for j = 2:n
	grad(j) = grad(j) + (lambda / m)*(theta(j));
end
%}

% =============================================================

end
