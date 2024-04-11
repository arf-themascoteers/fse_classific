from algorithm import Algorithm
from sklearn.feature_selection import mutual_info_regression
import numpy as np


class AlgorithmMI(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)
        self.indices = None

    def get_selected_indices(self):
        mi_scores = mutual_info_regression(self.X_train, self.y_train)
        self.indices = np.argsort(mi_scores)[::-1][:self.target_feature_size]
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]