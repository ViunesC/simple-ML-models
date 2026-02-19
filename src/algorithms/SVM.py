import numpy as np

class SVM:

    def __init__(self, lr=1e-2, lambda_param=1e-2, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X:np.ndarray, y):
        _, n_features = X.shape

        # regularize y
        y_ = np.where(y <= 0, -1, 1)

        # init weights & bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                condition = y_[idx] * (np.dot(x, self.weights) - self.bias) >= 1

                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        prediction = np.dot(X, self.weights) - self.bias
        return np.sign(prediction)
