from algorithm import Algorithm
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression


class AlgorithmPCA(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        rfecv = RFECV(estimator=LinearRegression(), step=1, cv=5, min_features_to_select= self.target_feature_size)
        rfecv.fit(self.X_train, self.y_train)
        indices = [i for i, selected in enumerate(rfecv.support_) if selected]
        return rfecv, indices