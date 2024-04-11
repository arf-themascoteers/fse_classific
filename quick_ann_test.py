import my_utils
from model_ann import ModelANN
from ds_manager import DSManager
from sklearn.metrics import r2_score, mean_squared_error
import math

ds = DSManager(dataset="ghsi")
X_train, y_train, X_test, y_test = ds.get_train_test_X_y()
ann = ModelANN(X_train)
ann.fit(X_train, y_train)

y_pred = ann.predict(X_train)
r2_train = round(r2_score(y_train, y_pred), 2)
rmse_train = round(math.sqrt(mean_squared_error(y_train, y_pred)), 2)

y_pred = ann.predict(X_test)
r2_test = round(r2_score(y_test, y_pred), 2)
rmse_test = round(math.sqrt(mean_squared_error(y_test, y_pred)), 2)

print(f"r2 train {r2_train}")
print(f"r2 test {r2_test}")
print(f"rmse train {rmse_train}")
print(f"rmse test {rmse_test}")
