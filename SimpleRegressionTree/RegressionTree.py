import numpy as np

class RegressionTree:

    def __init__(self, max_depth=None, min_samples_split=2):
        self.__tree = {}
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split

    def fit(self, X, Y):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.__tree = {}
        self.__build_tree(X, Y, self.__tree, self.__max_depth)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        preds = []
        for x in X:
            preds.append(self.__search(self.__tree, x))
        return preds

    def get_depth(self):
        return self.__calc_depth(self.__tree)

    def get_tree(self):
        return self.__tree

    def __calc_depth(self, tree):
        if 'value' in tree:
            return 0
        return 1 + max(self.__calc_depth(tree['left']), self.__calc_depth(tree['right']))


    def __search(self, tree, x):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature']] <= tree['t']:
            return self.__search(tree['left'], x)
        else:
            return self.__search(tree['right'], x)

    def __build_tree(self, X, Y, tree, max_depth):
        if len(X) < self.__min_samples_split or max_depth == 0 or np.all(Y == Y[0]):
            tree['value'] = np.mean(Y)
            return

        best_impurity = -1
        best_t = None
        best_j = None

        for j in range(X.shape[1]):
            unique_values = np.unique(X[:, j])
            if len(unique_values) > 100:
                unique_values = np.random.choice(unique_values, size=100, replace=False)
            for t in unique_values:
                left_mask = X[:, j] <= t
                right_mask = X[:, j] > t

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = Y[left_mask]
                y_right = Y[right_mask]
                impurity = self.__square_impurity(Y, y_left, y_right)

                if impurity > best_impurity:
                    best_impurity = impurity
                    best_t = t
                    best_j = j

        if best_j is None:
            tree['value'] = np.mean(Y)
            return

        tree['t'] = best_t
        tree['feature'] = best_j
        tree['left'] = {}
        tree['right'] = {}

        if max_depth is not None:
            max_depth -= 1

        left_mask = X[:, best_j] <= best_t
        right_mask = X[:, best_j] > best_t

        self.__build_tree(X[left_mask], Y[left_mask], tree['left'], max_depth)
        self.__build_tree(X[right_mask], Y[right_mask], tree['right'], max_depth)

    def __squared_impurity(self, y):
        # Same as: np.sum((y - np.mean(y)) ** 2)
        return np.var(y) * len(y)

    def __square_impurity(self, y_parent, y_left, y_right):
        N = len(y_parent)
        impurity_l = self.__squared_impurity(y_left)
        impurity_r = self.__squared_impurity(y_right)
        impurity_p = self.__squared_impurity(y_parent)
        return impurity_p - len(y_left) / N * impurity_l - len(y_right) / N * impurity_r