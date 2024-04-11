from algorithm import Algorithm
import my_utils
from sklearn.feature_selection import SelectFromModel
import numpy as np


class AlgorithmFM(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        model = my_utils.get_internal_model()
        sfm = SelectFromModel(model, threshold=-np.inf, max_features=5)
        sfm.fit(self.X_train, self.y_train)
        selected_feature_indices = np.where(sfm.get_support())[0]
        return sfm, selected_feature_indices
