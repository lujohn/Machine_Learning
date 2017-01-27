function idx = findClosestCentroids(X, centroids)

%   This function implments the cluster asignment phase of the K-means algorithm.
% 	It computes the centroid memberships for every example (rows of X). The
% 	return value idx, is a vector of the centroid assignments.

% 	For example, idx(i) = 5 indicates that example i is assigned to centroid 5.
%	Each i is within {1, 2, . . . , K}.

% 	Note: The rows of the centroids matrix represent the position of each 
%	centroid.

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% get number of training examples
m = size(X, 1);

for i = 1:m,
	% find distance of example X(i) to 
	x = X(i, :);   % grab example i

	% calculate the squared distance to each centroid
	% I will do operations with column vectors instead of row vectors.
	temp = x' - centroids';
	squared_distances = sum(temp .^ 2);
	[min_dis min_dis_idx] = min(squared_distances);

	% assign example i's centroid number to idx(i)
	idx(i) = min_dis_idx;
end





% =============================================================

end

