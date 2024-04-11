from algorithm import Algorithm
from mlxtend.feature_selection import ExhaustiveFeatureSelector
import my_utils


class AlgorithmEx(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        selector = ExhaustiveFeatureSelector(my_utils.get_internal_model(),
                                             min_features=1,
                                             max_features=self.target_feature_size,
                                             scoring='neg_mean_squared_error', print_progress=True)

        selector.fit(self.X_train, self.y_train)
        return selector, selector.best_idx_