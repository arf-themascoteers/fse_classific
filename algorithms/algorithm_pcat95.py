from algorithm import Algorithm
from sklearn.decomposition import PCA


class AlgorithmPCAT95(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        pcat95 = PCA(n_components=0.95)
        pcat95.fit(self.X_train)
        print(f"AlgorithmPCAT95: {pcat95.n_components_}")
        return pcat95, []