from algorithm import Algorithm
from algorithms.fscr.fscr import FSCR


class AlgorithmFSCRNS(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        fscr = FSCR(self.X_train.shape[0], self.target_feature_size, False)
        fscr.fit(self.X_train, self.y_train, self.X_validation, self.y_validation)
        return fscr, fscr.get_indices()