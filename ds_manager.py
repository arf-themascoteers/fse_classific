import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DSManager:
    def __init__(self,dataset):
        self.dataset = dataset
        np.random.seed(0)
        dataset_path = f"data/{dataset}.csv"
        df = pd.read_csv(dataset_path)
        self.X_columns = DSManager.get_spectral_columns(df, self.dataset)
        self.y_column = DSManager.get_y_column(self.dataset)
        df = df[self.X_columns+[self.y_column]]
        df = df.sample(frac=1)
        df['crop'], class_labels = pd.factorize(df['crop'])
        self.full_data = df.to_numpy()
        self.full_data[:,0:-1] = DSManager._normalize(self.full_data[:,0:-1])

    def __repr__(self):
        return self.get_name()

    def get_name(self):
        return self.dataset

    def count_rows(self):
        return self.full_data.shape[0]

    def count_features(self):
        return len(self.X_columns)

    @staticmethod
    def get_spectral_columns(df, dataset):
        return list(df.columns)[1:]

    @staticmethod
    def get_y_column(dataset):
        return "crop"

    @staticmethod
    def _normalize(data):
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        return data

    def get_datasets(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.1, random_state=2)
        train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]
        validation_x = validation_data[:, :-1]
        validation_y = validation_data[:, -1]

        return train_x, train_y, test_x, test_y, validation_x, validation_y

    def get_X_y(self):
        return self.get_X_y_from_data(self.full_data)

    @staticmethod
    def get_X_y_from_data(data):
        x = data[:, :-1]
        y = data[:, -1]
        return x, y

    def get_train_test_validation(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.1, random_state=2)
        train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
        return train_data, test_data, validation_data

    def get_train_test_validation_X_y(self):
        train_data, test_data, validation_data = self.get_train_test_validation()
        return *DSManager.get_X_y_from_data(train_data), \
            *DSManager.get_X_y_from_data(test_data),\
            *DSManager.get_X_y_from_data(validation_data)

    def get_train_test(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.3, random_state=2)
        return train_data, test_data

    def get_train_test_X_y(self):
        train_data, test_data = self.get_train_test()
        return *DSManager.get_X_y_from_data(train_data), \
            *DSManager.get_X_y_from_data(test_data)

