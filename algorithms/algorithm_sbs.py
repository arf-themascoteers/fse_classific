from algorithm import Algorithm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import my_utils


class AlgorithmSBS(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        sfs = SFS(my_utils.get_internal_model(),
                  k_features=self.target_feature_size,
                  forward=False, floating=False, scoring='r2', cv=5)
        sfs.fit(self.X_train, self.y_train)
        return sfs, sfs.k_feature_idx_