import numpy as np
from main import RegressionTree

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
    tree = RegressionTree()
    tree.fit(X, Y, max_depth=0)
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
