# This script first generates synthetic data from two different multivariate Gaussian distributions.
# Then, it fits a Gaussian Mixture Model with two components to the data, predicts the cluster assignments
#  of the data points, and plots the results.

# You should see a scatter plot where the data points are colored according to their cluster assignments. 
# The GMM has successfully identified the two clusters in the synthetic data.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_spd_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data from a mixture of Gaussians
n_samples = 300

# Mean and covariance of the first Gaussian
mean1 = [0, 0]
cov1 = make_spd_matrix(2)

# Mean and covariance of the second Gaussian
mean2 = [3, 3]
cov2 = make_spd_matrix(2)

# Generate samples
X = np.vstack([
    np.random.multivariate_normal(mean1, cov1, n_samples),
    np.random.multivariate_normal(mean2, cov2, n_samples)
])

# Fit a Gaussian Mixture Model with two components (clusters)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict the cluster assignments of the data points
clusters = gmm.predict(X)

# Plot the generated data and cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='.')
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
