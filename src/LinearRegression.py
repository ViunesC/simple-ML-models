import numpy as np


class LinearRegression:
    def __init__(self, lr=1e-3, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X: np.ndarray, y):
        # initialize the training by setting weights and bias to be 0
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # give prediction with LR function:
            # y = wX + b
            y_pred = np.dot(X, self.weights) + self.bias

            # calculate the gradient of loss function with respect to w_i and b
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update the weights and bias with gradient descent
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias

        return y_pred
