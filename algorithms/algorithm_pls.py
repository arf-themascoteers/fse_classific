from algorithm import Algorithm
from sklearn.cross_decomposition import PLSRegression


class AlgorithmPLS(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        pls = PLSRegression(n_components=self.target_feature_size)
        pls.fit(self.X_train, self.y_train)
        return pls,[]