from algorithm import Algorithm
from sklearn.manifold import LocallyLinearEmbedding


class AlgorithmLLE(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        lle = LocallyLinearEmbedding(n_neighbors=3, n_components=self.target_feature_size)
        lle.fit(self.X_train)
        return lle,[]