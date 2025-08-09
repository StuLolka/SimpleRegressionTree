from RandomForest import RandomForest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SimpleRegressionTree.RegressionTree import RegressionTree
from test_functions import *

X_data, Y_data = generate_data()
# Train and test split
indices = np.arange(len(X_data))
np.random.shuffle(indices)
train_size = int(0.7 * len(X_data))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, y_train = X_data[train_idx], Y_data[train_idx]
X_test, y_test = X_data[test_idx], Y_data[test_idx]

# Simple Regression Tree
srt = RegressionTree()
srt.fit(X_train, y_train)
preds_srt = srt.predict(X_test)
mse_srt = np.mean((y_test - preds_srt) ** 2)

# Random forest
rf = RandomForest()
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
mse_rf = np.mean((y_test - preds_rf) ** 2)

print(f"Test MSE - Simple Regression Tree: {mse_srt:.4f}, Random Forest: {mse_rf:.4f}")

sorted_idx = np.argsort(X_data[:, 0])
X_sorted = X_data[sorted_idx]
Y_sorted = Y_data[sorted_idx]

sorted_test_idx = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sorted_test_idx]
preds_srt = np.array(preds_srt)
preds_rf = np.array(preds_rf)

preds_srt_sorted = preds_srt[sorted_test_idx]
preds_rf_sorted = preds_rf[sorted_test_idx]

fig, ax = plt.subplots(3, 1)
ax[0].plot(X_sorted, Y_sorted, color='g')
ax[0].set_title(f'Original')
ax[1].plot(X_sorted, Y_sorted, color='g')
ax[1].plot(X_test_sorted, preds_srt_sorted, color='r')
ax[1].set_title(f'RegressionTree, mse = {mse_srt:.4f}')
ax[2].plot(X_sorted, Y_sorted, color='g')
ax[2].plot(X_test_sorted, preds_rf_sorted, color='b')
ax[2].set_title(f'RandomForest, mse = {mse_rf:.4f}')


plt.tight_layout()
plt.show()








