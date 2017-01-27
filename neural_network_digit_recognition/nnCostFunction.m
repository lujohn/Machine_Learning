function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ---------------- Feedforward the Neural Network (Regularized) ------------------

% J is calculated over a double summation. First sum, sums over different
% classifiers. Note: Yk will be a number from 1 to K. Must convert to a K-dim 
% vector of 0's or 1's. Second sum sums over traning examples.

% Append a column of ones
X = [ones(m, 1) X];
A1 = X;

% Compute layer 2 activation
Z2 = Theta1 * A1';
A2 = sigmoid(Z2);

% Compute layer 3 activation (also is H(theta));
A2 = [ones(1, m) ; A2];   % Append row of ones to top of A2
Z3 = Theta2 * A2;
% NOTE: output i (corresponding to example i) is in i'th column
H = sigmoid(Z3);

% Loop through all classifiers
for i = 1:m,
	for k = 1:num_labels,
		% Convert y digit to a vector representation.
		y_vec = zeros(num_labels, 1);
		y_vec(y(i)) = 1;  % For ex, if y is 3, then y_vec is [0 0 0 1 0 0 0 ...]

		% Increment cost function
		J = J + [-y_vec(k) * log(H(k, i)) - (1 - y_vec(k)) * log(1 - (H(k, i)))];
	end
end

J = (1 / m) * J;

% Code to regularize parameter
Theta1_reg = Theta1(:, 2:end);
Theta1_reg = Theta1_reg .^ 2;

Theta2_reg = Theta2(:, 2:end);
Theta2_reg = Theta2_reg .^ 2;

J = J + (lambda / (2 * m)) * (sum(sum(Theta1_reg)) + sum(sum(Theta2_reg)));

% ------------- Backpropagation Algorithm (to compute gradients) ------------

Delta_1 = zeros(hidden_layer_size, input_layer_size + 1);
Delta_2 = zeros(num_labels, hidden_layer_size + 1);

for i = 1:m,
	% Grab one output vector
	a_3 = H(:, i);
	a_2 = A2(:, i);

	% Vectorize y
	y_vec = zeros(num_labels, 1);
	y_vec(y(i)) = 1;

	% Outer Layer
	delta_3 = a_3 - y_vec;

	% Hidden Layer (Layer 2)
	delta_2 = (Theta2' * delta_3) .* [ones(1, 1) ; sigmoidGradient(Z2(:, i))];
	delta_2 = delta_2(2:end);   % remove bias term

	% Accumluate Gradient
	Delta_2 = Delta_2 + (delta_3 * a_2');
	Delta_1 = Delta_1 + (delta_2 * X(i, :));
end

Theta2_grad = (1 / m) * Delta_2;
Theta1_grad = (1 / m) * Delta_1;

% Add Regularization to Gradient
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end); 



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
