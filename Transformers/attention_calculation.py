import numpy as np

# Define the embeddings and transformation matrices
e_he = np.array([2, 4])
e_loves = np.array([3, 2])
e_cats = np.array([1, 3])

W_Q = np.array([[1, 2], [0, 1]])
W_K = np.array([[1, 0], [2, 1]])
W_V = np.array([[2, 1], [1, 0]])

# Calculate Q, K, V for each word
Q_loves = np.dot(e_loves, W_Q)
K_loves = np.dot(e_loves, W_K)
V_loves = np.dot(e_loves, W_V)

# Calculate K for other words
K_he = np.dot(e_he, W_K)
K_cats = np.dot(e_cats, W_K)

# Calculate V for other words
V_he = np.dot(e_he, W_V)
V_cats = np.dot(e_cats, W_V)

# Calculate similarity scores
S_loves_he = np.dot(Q_loves, K_he)
S_loves_loves = np.dot(Q_loves, K_loves)
S_loves_cats = np.dot(Q_loves, K_cats)

# Apply softmax function to the scores
def softmax(scores):
    e_scores = np.exp(scores)
    return e_scores / np.sum(e_scores)

scores = np.array([S_loves_he, S_loves_loves, S_loves_cats])
attention_weights = softmax(scores)

# Calculate the final attention output for "loves"
Attention_loves = attention_weights[0] * V_he + attention_weights[1] * V_loves + attention_weights[2] * V_cats


