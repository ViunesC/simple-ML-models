import numpy as np


def unit_step_func(x):
    return np.where(x > 0, 1, 0)


class Perceptron:
    def __init__(self, lr=1e-2, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        _, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)  # random init instead of 0 could be better
        self.bias = 0

        # normalize y
        y_ = np.where(y > 0, 1, 0)

        # setup training
        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                linear_output = np.dot(x, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)

                # apply perceptron update rule
                delta_w = self.lr * (y_[idx] - y_pred) * x
                delta_b = self.lr * (y_[idx] - y_pred)

                self.weights += delta_w
                self.bias += delta_b

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = self.activation_func(linear_output)

        return predictions
