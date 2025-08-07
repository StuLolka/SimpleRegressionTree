import numpy as np
import matplotlib.pyplot as plt


class RegressionTree:

    def __init__(self):
        self.__tree = {}

    def fit(self, X, Y, max_depth=None):
        if X.shape == (len(X),):
            X = X.reshape(len(X), 1)
        self.__tree = {}
        self.__build_tree(X, Y, self.__tree, max_depth)

    def predict(self, X):
        if X.shape == (len(X),):
            X = X.reshape(len(X), 1)
        preds = []
        for x in X:
            preds.append(self.__search(self.__tree, x))
        return preds

    def __search(self, tree, x):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature']] <= tree['t']:
            return self.__search(tree['left'], x)
        else:
            return self.__search(tree['right'], x)

    def __build_tree(self, X, Y, tree, max_depth):
        if len(X) < 2 or max_depth == 0 or np.all(Y == Y[0]):
            tree['value'] = np.mean(Y)
            return

        best_impurity = -1
        best_t = None
        best_j = None

        for j in range(X.shape[1]):
            for t in X[:, j]:
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


X_data = np.arange(-5, 5, 0.1)
Y_data = X_data ** 2

fig, ax = plt.subplots(2, 2)
fig.canvas.manager.set_window_title('Quadratic function')
count_1 = 0
count_2 = 0
for depth in range(2, 9, 2):
    model = RegressionTree()
    model.fit(X_data, Y_data, max_depth=depth)
    predict = model.predict(X_data)

    ax[count_2][count_1].set_title(f'max depth = {depth}')
    ax[count_2][count_1].plot(X_data, Y_data, color='g')
    ax[count_2][count_1].plot(X_data, predict, color='r')
    count_1 += 1
    if count_1 == 2:
        count_1 = 0
        count_2 += 1

plt.tight_layout()

# Cosine function
X_data = np.arange(-5, 5, 0.1)
Y_data = np.cos(X_data)

fig2, ax2 = plt.subplots(2, 2)
fig2.canvas.manager.set_window_title('Cosine function')
count_1 = 0
count_2 = 0
for depth in range(2, 9, 2):
    model = RegressionTree()
    model.fit(X_data, Y_data, max_depth=depth)
    predict = model.predict(X_data)

    ax2[count_2][count_1].set_title(f'max depth = {depth}')
    ax2[count_2][count_1].plot(X_data, Y_data, color='g')
    ax2[count_2][count_1].plot(X_data, predict, color='r')
    count_1 += 1
    if count_1 == 2:
        count_1 = 0
        count_2 += 1


plt.tight_layout()
plt.show()