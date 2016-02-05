from abc import ABCMeta, abstractmethod
import numpy as np
try:
    import cPickle as cp  # Python 2
except ImportError:
    import pickle as cp  # Python 3


class ModelABC:
    """
    Abstract class representing a model.

    """
    __metaclass__ = ABCMeta


class Model(ModelABC):
    """
    Mother class for a model with utility methods.
    """

    def __init__(self, name, X, y):
        self.name = name
        if np.ndim(X) != 2:
            raise(ModelError("Expecting ndarray for X"))
        else:
            self.X = X
        if np.ndim(y) != 2:
            raise ModelError("Expecting ndarray for y")
        else:
            self.y = y

    def _k_fold_generator(self, k_fold):
        subset_size = int(np.floor(self.X.shape[0] / float(k_fold)))
        for k in range(k_fold):
            X_train = np.append(self.X[:k * subset_size, :], self.X[(k + 1) * subset_size:, :], axis=0)
            X_valid = self.X[k * subset_size:, :][:subset_size, :]
            y_train = np.append(self.y[:k * subset_size, :], self.y[(k + 1) * subset_size:, :], axis=0)
            y_valid = self.y[k * subset_size:, :][:subset_size, :]
            yield X_train, y_train, X_valid, y_valid


class ModelError(Exception):
    def __init__(self, message):
        self.message = message

