function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% Caluculate the new positions for centroids (vectorized implementation)
% The new position of centroid i will be the mean of the positions of 
% examples assigned to it.
for i = 1:K,
	% find all training examples assigned to centroid i
	% u(j) = 1 if ex j is assigned to centroid i.
	% 0 otherwise.
	u = (idx == i);

	% find the mean position
	Y = (u .* X);
	Y = Y(any(Y, 2), :);   % remove all zero rows
	count = size(Y, 1);
	centroids(i, : ) = (1 / count) * sum(Y);  % Update centroid i position
end

end

