import numpy as np

class RegressionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.__tree = {}
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.__tree = self.__build_tree(X, Y, depth=0)

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.array([self.__predict_one(row, self.__tree) for row in X])

    def get_depth(self):
        return self.__calc_depth(self.__tree)

    def __calc_depth(self, tree):
        if 'value' in tree:
            return 0
        return 1 + max(self.__calc_depth(tree['left']), self.__calc_depth(tree['right']))

    def __predict_one(self, x, tree):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature']] <= tree['threshold']:
            return self.__predict_one(x, tree['left'])
        else:
            return self.__predict_one(x, tree['right'])

    def __build_tree(self, X, Y, depth):
        n_samples, n_features = X.shape
        if (self.__max_depth is not None and depth >= self.__max_depth) or \
                n_samples < self.__min_samples_split or \
                np.all(Y == Y[0]):
            return {'value': np.mean(Y)}

        SS_parent = self.__squared_error(Y)

        best_feature = None
        best_threshold = None
        best_score = -np.inf

        for feature_idx in range(n_features):
            sorted_idx = np.argsort(X[:, feature_idx])
            X_feat = X[sorted_idx, feature_idx]
            Y_sorted = Y[sorted_idx]

            # thresholds = midpoints between consecutive unique values
            mids = (X_feat[:-1] + X_feat[1:]) / 2
            mask = X_feat[:-1] != X_feat[1:]
            thresholds = mids[mask]

            for threshold in thresholds:
                left_mask = X_feat <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                SS_left = self.__squared_error(Y_sorted[left_mask])
                SS_right = self.__squared_error(Y_sorted[right_mask])
                score = SS_parent - SS_left - SS_right

                if score > best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_feature is None:
            return {'value': np.mean(Y)}

        left_mask_global = X[:, best_feature] <= best_threshold
        right_mask_global = ~left_mask_global

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self.__build_tree(X[left_mask_global], Y[left_mask_global], depth + 1),
            'right': self.__build_tree(X[right_mask_global], Y[right_mask_global], depth + 1)
        }

    def __squared_error(self, y):
        # Sum of squared deviations (SSE)
        return np.var(y) * len(y)
