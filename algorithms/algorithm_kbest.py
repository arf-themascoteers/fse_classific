from algorithm import Algorithm
from sklearn.feature_selection import SelectKBest, f_regression


class AlgorithmKBest(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        selector = SelectKBest(score_func=f_regression, k=self.target_feature_size)
        selector.fit(self.X_train, self.y_train)
        return selector, selector.get_support(indices=True)
