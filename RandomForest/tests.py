import numpy as np
from RandomForest import RandomForest

def test_basic_functionality():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    Y = np.array([1, 2, 3, 4])
    rf = RandomForest(n_estimators=10, max_depth=2)
    rf.fit(X, Y)
    preds = rf.predict(X)
    assert len(preds) == len(Y)
    assert not np.isnan(preds).any()


def test_consistency():
    X = np.random.rand(100, 3)
    Y = X[:, 0] + X[:, 1] * 2
    rf = RandomForest(n_estimators=50, max_depth=3)
    rf.fit(X, Y)
    preds1 = rf.predict(X[:5])
    preds2 = rf.predict(X[:5])
    assert np.allclose(preds1, preds2)

def test_input_dimensions():
    # 1D input
    X1 = np.array([1, 2, 3, 4])
    Y1 = np.array([1, 2, 3, 4])
    rf = RandomForest(n_estimators=5)
    rf.fit(X1, Y1)
    preds1 = rf.predict(np.array([1, 2]))
    assert len(preds1) == 2

    # 2D input
    X2 = np.array([[1, 2], [3, 4], [5, 6]])
    Y2 = np.array([1, 2, 3])
    rf.fit(X2, Y2)
    preds2 = rf.predict(np.array([[1, 2], [3, 4]]))
    assert len(preds2) == 2

def test_n_estimators_impact():
    X = np.random.rand(100, 2)

    Y = X[:, 0] * 2 + X[:, 1] * 3
    rf1 = RandomForest(n_estimators=10)
    rf2 = RandomForest(n_estimators=100)
    rf1.fit(X, Y)
    rf2.fit(X, Y)
    preds1 = rf1.predict(X[:5])
    preds2 = rf2.predict(X[:5])
    # Проверяем, что предсказания не сильно различаются
    assert np.mean(np.abs(preds1 - preds2)) < 0.1

def test_max_features_impact():
    X = np.random.rand(100, 5)
    Y = X[:, 0] + X[:, 1] * 2
    rf1 = RandomForest(max_features=1)
    rf2 = RandomForest(max_features=3)
    rf1.fit(X, Y)
    rf2.fit(X, Y)
    preds1 = rf1.predict(X[:5])
    preds2 = rf2.predict(X[:5])
    assert not np.allclose(preds1, preds2)

def test_noise_robustness():
    X = np.random.rand(250, 3)
    Y = X[:, 0] * 2 + np.random.randn(250) * 0.5
    rf = RandomForest(n_estimators=50)
    rf.fit(X, Y)
    preds = rf.predict(X)
    mse = np.mean((Y - preds) ** 2)
    assert mse < 1.0

def test_max_depth_respected():
    X = np.random.rand(100, 3)
    Y = X[:, 0] * 2
    rf = RandomForest(max_depth=2)
    rf.fit(X, Y)
    assert rf.get_max_depth() <= 2