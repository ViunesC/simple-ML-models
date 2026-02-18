import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X: np.ndarray):
        # get X bar (mean) of X, then do a mean-centering 
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # get covariance matrix
        cov = np.cov(X.T)

        # get eigenvector and eigenvalue of covariance matrix
        eig_vec, eig_val = np.linalg.eig(cov)

        # since eigenvector we got is a column vector, we shall transpose it to row vector for easy calculation
        eig_vec = eig_vec.T

        # sort the eigenvector and eigenvalues in decreasing order
        idxs = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[idxs]
        eig_val = eig_val[idxs]

        # pick top n eigenvectors and keep it as our principle components
        self.components = eig_vec[:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)