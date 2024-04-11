from algorithm import Algorithm
from sklearn.linear_model import Lasso
import my_utils
import numpy as np


class AlgorithmLasso(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)
        self.indices = None

    def get_selected_indices(self):
        lasso = Lasso(alpha=0.001)
        lasso.fit(self.X_train, self.y_train)
        self.indices = np.argsort(np.abs(lasso.coef_))[::-1][:self.target_feature_size]
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]