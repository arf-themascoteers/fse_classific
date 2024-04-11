from model_ann import ModelANN
from ds_manager import DSManager
from sklearn.metrics import r2_score, mean_squared_error
import math

ds = DSManager(dataset="ghsi")
X_train, y_train, X_test, y_test = ds.get_train_test_X_y()
ann = ModelANN(X_train)
ann.fit(X_train, y_train)

train_accuracy = ann.prediction_accuracy(X_train, y_train)
test_accuracy = ann.prediction_accuracy(X_test, y_test)

print(f"train_accuracy {train_accuracy}; test_accuracy {test_accuracy}")


