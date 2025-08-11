import numpy as np
import sys
import os
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SimpleRegressionTree.RegressionTree import RegressionTree

class RandomForest:
    def __init__(self, n_estimators=100, max_features=None, max_depth=None):
        self.__models = []
        self.__n_estimators = n_estimators
        self.__max_features = max_features
        self.__max_depth = max_depth
        self.__features_per_model = []

    def fit(self, X, Y):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        if self.__max_features is None:
            self.__max_features = int(np.sqrt(n_features))

        for t in range(self.__n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_X = X[indices]
            sample_Y = Y[indices]

            feature_indices = np.random.choice(n_features, size=self.__max_features, replace=False)
            self.__features_per_model.append(feature_indices)

            sample_X = sample_X[:, feature_indices]

            model = RegressionTree(max_depth=self.__max_depth)
            model.fit(sample_X, sample_Y)
            self.__models.append(model)

    def get_max_depth(self):
        max_depth = 0
        for model in self.__models:
            depth = model.get_depth()
            if max_depth < depth:
                max_depth = depth
        return max_depth

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        predictions = []
        for model, feature_indices in zip(self.__models, self.__features_per_model):
            pred = model.predict(X[:, feature_indices])
            predictions.append(pred)
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)
