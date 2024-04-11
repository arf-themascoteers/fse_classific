from algorithm import Algorithm
from sklearn.ensemble import RandomForestRegressor


class AlgorithmTBFI(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        selected_features_indices =  \
            model.feature_importances_.argsort()[-self.target_feature_size:][::-1]
        return model, selected_features_indices
