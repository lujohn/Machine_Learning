function centroids = kMeansInitCentroids(X, K)

% This function randomly initializes the centroids in the K-means algorithm.

centroids = zeros(K, size(X, 2));
rand_idx = randperm(size(X, 1));
centroids = X(rand_idx(1:K), :);

end

