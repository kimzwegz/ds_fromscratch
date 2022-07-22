import numpy as np
import pandas as pd


class My_PCA():
    def __init__(self, X):
        self.X = X
        self._n_sample = X.shape[0]
        self._U, self._s , self._V = np.linalg.svd(X-X.mean(0))

    def transform(self):
        return self._U * self._s

    def singular(self):
        return {'matrix': self._s , 'values': np.diag(self._s)}

    def components(self):
        return self._U

    def explained_var(self):
        return (self._s**2/(self._n_sample-1))

    def explained_var_ratio(self):
        return np.array([i**2 / sum(self._s**2) for i in self._s])