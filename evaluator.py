from ds_manager import DSManager
from datetime import datetime
import os
from algorithm_creator import AlgorithmCreator
from sklearn.metrics import r2_score, mean_squared_error
import math
import pandas as pd
import my_utils


class Evaluator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.filename = os.path.join("results","results.csv")
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as file:
                file.write("algorithm,rows,columns,time,target_size,final_size,"
                           "accuracy_train,accuracy_test,"
                           "rmse_train,rmse_test,"
                           "selected_features\n")

    def evaluate(self):
        for task in self.tasks:
            print(task)
            dataset = task["dataset"]
            target_feature_size = task["target_feature_size"]
            algorithm_name = task["algorithm"]
            dataset = DSManager(dataset=dataset)
            if self.is_done(algorithm_name, dataset, target_feature_size):
                print("Done already. Skipping.")
                continue
            elapsed_time, accuracy_original, rmse_original, \
                accuracy_reduced_train, rmse_reduced_train, \
                accuracy_reduced_test, rmse_reduced_test, \
                final_indices, selected_features = \
                self.do_algorithm(algorithm_name, dataset, target_feature_size)


            with open(self.filename, 'a') as file:
                file.write(
                    f"{algorithm_name},{dataset.count_rows()},"
                    f"{dataset.count_features()},{round(elapsed_time,2)},{target_feature_size},{final_indices},"
                    f"{accuracy_reduced_train},{accuracy_reduced_test},"
                    f"{rmse_reduced_train},{rmse_reduced_test},"
                    f"{';'.join(str(i) for i in selected_features)}\n")

    def is_done(self,algorithm_name,dataset,target_feature_size):
        df = pd.read_csv(self.filename)
        if len(df) == 0:
            return False
        rows = df.loc[
            (df['algorithm'] == algorithm_name) &
            (df['rows'] == dataset.count_rows()) &
            (df['columns'] == dataset.count_features()) &
            (df['target_size'] == target_feature_size)
        ]
        return len(rows) != 0

    def do_algorithm(self, algorithm_name, dataset, target_feature_size):
        X_train, y_train, X_test, y_test = dataset.get_train_test_X_y()
        print(f"X_train,X_test: {X_train.shape} {X_test.shape}")
        #_, _, accuracy_original, rmse_original = Evaluator.get_metrics(algorithm_name, X_train, y_train, X_test, y_test)
        algorithm = AlgorithmCreator.create(algorithm_name, X_train, y_train, target_feature_size)
        start_time = datetime.now()
        selected_features = algorithm.fit()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        X_train_reduced = algorithm.transform(X_train)
        X_test_reduced = algorithm.transform(X_test)
        accuracy_reduced_train, rmse_reduced_train, accuracy_reduced_test, rmse_reduced_test = \
            Evaluator.get_metrics(algorithm_name, X_train_reduced, y_train, X_test_reduced, y_test)
        return elapsed_time, 0, 0, \
            accuracy_reduced_train, rmse_reduced_train, \
            accuracy_reduced_test, rmse_reduced_test, X_test_reduced.shape[1], selected_features

    @staticmethod
    def get_metrics(algorithm_name, X_train, y_train, X_test, y_test):
        metric_evaluator = my_utils.get_metric_evaluator(algorithm_name, X_train)
        metric_evaluator.fit(X_train, y_train)

        train_accuracy, train_rmse = metric_evaluator.prediction_accuracy(X_train, y_train)
        test_accuracy, test_rmse = metric_evaluator.prediction_accuracy(X_test, y_test)

        print(f"train_accuracy {train_accuracy}; test_accuracy {test_accuracy}")
        print(f"train_rmse {train_rmse}; test_rmse {test_rmse}")

        return train_accuracy, train_rmse, test_accuracy, test_rmse
