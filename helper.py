import numpy as np

try:
    from sklearn.model_selection import cross_val_score, LeaveOneOut
    from sklearn.neighbors import KNeighborsClassifier

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def evaluate_subset(data, feature_indices):
    """Return leave-one-out 1-NN accuracy (%) for selected feature indices."""
    if not feature_indices:
        raise ValueError("feature_indices cannot be empty")

    y = data[:, 0]
    X = data[:, feature_indices]

    if SKLEARN_AVAILABLE:
        model = KNeighborsClassifier(n_neighbors=1, p=2)
        scores = cross_val_score(model, X, y, cv=LeaveOneOut())
        return float(scores.mean() * 100.0)

    # NumPy fallback when scikit-learn is not installed.
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest_idx = np.argmin(dist, axis=1)
    predictions = y[nearest_idx]
    return float(np.mean(predictions == y) * 100.0)
