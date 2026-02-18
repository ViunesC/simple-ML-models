import numpy as np


class NaiveBayes:
    def fit(self, X: np.ndarray, y: np.ndarray):
        # extract info
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean and variance for P(x_i | y) for each i (features) for each class y
        # and prior (frequency) of each class
        self._means = np.zeros((n_classes, n_features), dtype=np.float64)
        self._vars = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]  # extract all X belongs to class c
            self._means[idx, :] = X_c.mean(axis=0)
            self._vars[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        predicts = [self._predict(x) for x in X]
        return np.array(predicts)

    def _predict(self, x):
        # for each class:
        # calculate the prior and posterior and sum them
        # then return the class with highest posterior sum
        # formula: y = argmax_y[log(P(x_1|y)) + log(P(x_2|y)) + log(P(x_3|y)) + ... + log(P(x_n|y)) + log(P(y))]
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._means[class_idx]
        var = self._vars[class_idx]

        denom = np.sqrt(2 * np.pi * var)
        num_factor = np.pow(x - mean, 2) / (2 * var)
        num = np.exp(-num_factor)

        return num / denom
