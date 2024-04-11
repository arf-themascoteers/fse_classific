from sklearn.linear_model import LinearRegression
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


def get_hidden_for_full(feature_size):
    h1 = 50
    h2 = 10
    if 50 <= feature_size < 100:
        h1 = 40
    elif 100 <= feature_size < 200:
        h1 = 30
    elif 200 <= feature_size < 300 :
        h1 = 20
    elif 300 <= feature_size < 350:
        h1 = 18
    elif 350 <= feature_size < 400:
        h1 = 16
    elif 400 <= feature_size:
        h1 = 10
        h2 = 8

    return h1, h2


def get_hidden_for_short(feature_size):
    h1 = 15
    h2 = 10
    if 50 <= feature_size < 100:
        h1 = 11
    elif 100 <= feature_size < 200:
        h1 = 6
        h2 = 5
    elif 200 <= feature_size < 250:
        h1 = 5
        h2 = 4
    elif 250 < feature_size:
        h1 = 5
        h2 = 0
    return h1, h2


def get_hidden(rows, feature_size):
    if rows > 3000:
        return get_hidden_for_full(feature_size)
    return get_hidden_for_short(feature_size)


def get_metric_evaluator(algorithm_name, X):
    return RandomForestClassifier()


def get_internal_model():
    return LinearRegression()


def get_linear(rows, feature_size):
    if rows < feature_size:
        return nn.Sequential(
            nn.Linear(feature_size, 1),
        )
    h1, h2 = get_hidden(rows, feature_size)
    if h1 == 0:
        return nn.Sequential(nn.Linear(feature_size, 1))
    if h2 == 0:
        return nn.Sequential(
            nn.Linear(feature_size, h1),
            nn.LeakyReLU(),
            nn.Linear(h1, 1),
        )
    return nn.Sequential(
        nn.Linear(feature_size, h1),
        nn.LeakyReLU(),
        nn.Linear(h1, h2),
        nn.LeakyReLU(),
        nn.Linear(h2, 1)
    )


def get_lr(rows, features):
    return 0.001


def get_epoch(rows, features):
    return 350
