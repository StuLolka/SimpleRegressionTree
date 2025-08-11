import numpy as np
from RegressionTree import RegressionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def test_simple_fit_predict():
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([1, 2, 3, 4, 5])
    tree = RegressionTree()
    tree.fit(X, Y)
    preds = tree.predict(np.array([[1], [2], [3], [4], [5]]))
    assert np.allclose(preds, Y, atol=1e-5)

def test_constant_Y():
    X = np.array([1, 2, 3])
    Y = np.array([10, 10, 10])
    tree = RegressionTree()
    tree.fit(X, Y)
    preds = tree.predict(np.array([[1], [2], [3]]))
    assert np.allclose(preds, [10, 10, 10], atol=1e-5)

def test_depth_limit():
    X = np.array([1, 2, 3])
    Y = np.array([1, 2, 3, 4])
    tree = RegressionTree(max_depth=0)
    tree.fit(X, Y)
    preds = tree.predict(np.array([[1], [2], [3], [4]]))
    # if max_depth=0 the tree should not be split, just average by Y
    assert np.allclose(preds, [2.5, 2.5, 2.5, 2.5], atol=1e-5)

def test_multifeature():
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    Y = np.array([1, 2, 3, 4])
    tree = RegressionTree()
    tree.fit(X, Y)
    preds = tree.predict(X)
    assert len(preds) == len(Y)


def test_min_samples_split():
    X = np.array([1, 2, 3, 4])
    Y = np.array([1, 2, 3, 4])
    tree = RegressionTree(min_samples_split=5)  # Больше, чем len(X)
    tree.fit(X, Y)
    preds = tree.predict(np.array([[1], [2], [3], [4]]))
    # The tree should not be split, prediction is mean Y
    assert np.allclose(preds, [2.5, 2.5, 2.5, 2.5], atol=1e-5)

def test_noisy_data():
    X = np.arange(100).reshape(-1, 1)
    Y = X.ravel() + np.random.randn(100) * 10
    tree = RegressionTree(max_depth=3)
    tree.fit(X, Y)
    preds = tree.predict(X)
    # Check that the MSE is at least better than the average prediction
    mse_tree = np.mean((Y - preds) ** 2)
    mse_mean = np.mean((Y - np.mean(Y)) ** 2)
    assert mse_tree < mse_mean
    sklearn_tree = DecisionTreeRegressor(max_depth=3)
    sklearn_tree.fit(X, Y)
    sklearn_preds = sklearn_tree.predict(X)
    sklearn_mse = mean_squared_error(Y, sklearn_preds)
    print(f'mse_tree = {mse_tree}, sklearn_mse = {sklearn_mse}')
    assert np.isclose(mse_tree, sklearn_mse, rtol=0.01)

def test_1d_vs_2d_input():
    X_1d = np.array([1, 2, 3])
    X_2d = np.array([[1, 1], [2, 2], [3, 3]])
    Y = np.array([1, 2, 3])
    tree_1d = RegressionTree()
    tree_2d = RegressionTree()
    tree_1d.fit(X_1d, Y)
    tree_2d.fit(X_2d, Y)
    preds_1d = tree_1d.predict(np.array([[1], [2], [3]]))
    preds_2d = tree_2d.predict(np.array([[1], [2], [3]]))
    assert np.allclose(preds_1d, preds_2d, atol=1e-5)
    sklearn_tree_2d = DecisionTreeRegressor()
    sklearn_tree_2d.fit(X_2d, Y)
    sklearn_preds_2d = sklearn_tree_2d.predict(X_2d)
    assert np.allclose(preds_1d, sklearn_preds_2d, atol=1e-5)
    assert np.allclose(preds_2d, sklearn_preds_2d, atol=1e-5)

def test_max_depth_respected():
    X = np.arange(10).reshape(-1, 1)
    Y = X.ravel() ** 2
    tree = RegressionTree(max_depth=2)
    tree.fit(X, Y)
    assert tree.get_depth() <= 2

def test_identical_X():
    X = np.array([[1], [1], [1], [1]])
    Y = np.array([1, 2, 3, 4])
    tree = RegressionTree()
    tree.fit(X, Y)
    preds = tree.predict(X)
    # All X are the same, tree should not be split
    sklearn_tree = DecisionTreeRegressor()
    sklearn_tree.fit(X, Y)
    sklearn_preds = sklearn_tree.predict(X)
    assert np.allclose(preds, [2.5, 2.5, 2.5, 2.5], atol=1e-5)
    assert np.allclose(preds, sklearn_preds, atol=1e-5)

def test_compare_to_sklearn():
    X = np.random.rand(100, 3)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1

    tree = RegressionTree(max_depth=3)
    tree.fit(X, y)
    preds = tree.predict(X)
    mse = mean_squared_error(y, preds)

    # Sklearn реализация
    sklearn_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    sklearn_tree.fit(X, y)
    sklearn_preds = sklearn_tree.predict(X)
    sklearn_mse = mean_squared_error(y, sklearn_preds)

    assert np.isclose(mse, sklearn_mse, rtol=0.01)


