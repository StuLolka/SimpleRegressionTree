import numpy as np

def generate_data(n_samples=300, n_features=1, noise=0.1, random_state=42):
    np.random.seed(random_state)
    X = np.random.uniform(-5, 5, size=(n_samples, n_features))
    y = np.sum([np.sin(X[:, i]) + np.cos(X[:, i]**2) for i in range(n_features)], axis=0)
    y += np.random.randn(n_samples) * noise
    return X, y
