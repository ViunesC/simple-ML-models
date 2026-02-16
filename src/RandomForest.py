import numpy as np
from DecisionTree import DecisionTree
from collections import Counter


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_features)
            X_samples, y_samples = self._bootstrap_samples(X, y)
            tree.fit(X_samples, y_samples)
            self.trees.append(tree)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        tree_preds = np.swapaxes(predictions, 0, 1)
        voted_predictions = np.array(
            [self._most_common_label(pred) for pred in tree_preds]
        )
        return voted_predictions

    def _bootstrap_samples(self, X: np.ndarray, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        """Find the most common label (majority vote)."""
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
