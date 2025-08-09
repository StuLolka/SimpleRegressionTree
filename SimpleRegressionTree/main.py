import numpy as np
import matplotlib.pyplot as plt
from RegressionTree import RegressionTree
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_functions import *

def plot_data(X, Y, title):
    fig, ax = plt.subplots(2, 2)
    fig.canvas.manager.set_window_title(title)
    count_1 = 0
    count_2 = 0

    random_train_indices = np.random.choice(len(X), size=int(len(X) * 0.7), replace=False)
    random_test_indices = np.array([i for i in range(len(X)) if i not in random_train_indices])

    X_train, y_train, X_test, y_test = X[random_train_indices], Y[random_train_indices], X[
        random_test_indices], Y[random_test_indices]

    for depth in range(2, 9, 2):
        model = RegressionTree(max_depth=depth)
        model.fit(X_train, y_train)
        predict = model.predict(X_test)


        ax[count_2][count_1].set_title(f'max depth = {depth}')
        ax[count_2][count_1].scatter(X_test, y_test, color='g')
        ax[count_2][count_1].scatter(X_test, predict, color='r')
        count_1 += 1
        if count_1 == 2:
            count_1 = 0
            count_2 += 1
    plt.tight_layout()

X_data = np.arange(-15, 15, 0.1)
# Quadratic function
Y_data = X_data ** 2
plot_data(X_data, Y_data, 'Quadratic function')

# Cosine function
Y_data = np.cos(X_data)
plot_data(X_data, Y_data, 'Cosine function')

# Random function
X_data, Y_data = generate_data()
plot_data(X_data, Y_data, 'Random function')

plt.show()