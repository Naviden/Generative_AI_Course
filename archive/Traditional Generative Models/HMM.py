# This script first defines a Gaussian HMM with two hidden states, 
# sets its parameters, and generates a sequence of 100 observations. 
# It then fits a new HMM to these generated observations and prints out 
# the estimated parameters of the fitted model.

# Please note that the estimated parameters might not be exactly the same 
# as the original parameters due to the stochastic nature of the HMM and 
# the relatively small amount of data.

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Define a Gaussian HMM
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)

# Define the parameters of the HMM
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([[0.7, 0.3],
                             [0.3, 0.7]])
model.means_ = np.array([[0.0], [3.0]])
model.covars_ = np.tile(np.identity(1), (2, 1, 1))

# Generate a sequence of observations
obs, states = model.sample(100)

# Plot the generated observations and their corresponding hidden states
plt.plot(obs, label="Observations")
plt.plot(states, label="Hidden States", linestyle="--")
plt.legend()
plt.show()

# Fit an HMM to the observations
model_fit = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
model_fit.fit(obs)

# Print the parameters of the fitted HMM
print("Fitted Start Probabilities:")
print(model_fit.startprob_)
print("\nFitted Transition Matrix:")
print(model_fit.transmat_)
print("\nFitted Means:")
print(model_fit.means_)
print("\nFitted Covariances:")
print(model_fit.covars_)
