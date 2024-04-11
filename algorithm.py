from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class Algorithm(ABC):
    def __init__(self, X_train, y_train, target_feature_size):
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=40)
        self.target_feature_size = target_feature_size
        self.selected_indices = []
        self.model = None

    def fit(self):
        self.model, self.selected_indices = self.get_selected_indices()
        return self.selected_indices

    def transform(self, X):
        if len(self.selected_indices) != 0:
            return X[:,self.selected_indices]
        return self.model.transform(X)

    @abstractmethod
    def get_selected_indices(self):
        pass
