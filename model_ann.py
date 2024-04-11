import torch.nn as nn
import torch
import my_utils
from sklearn.metrics import r2_score
import numpy as np


class ModelANN(nn.Module):
    def __init__(self, X):
        super().__init__()
        torch.manual_seed(3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows = X.shape[0]
        self.target_feature_size = X.shape[1]
        self.linear = nn.Sequential(
            nn.Linear(self.target_feature_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 5)
        )
        self.epoch = 10000
        self.lr = 0.001
        self.criterion = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, X):
        return self.linear(X)

    def create_optimizer(self):
        weight_decay = self.lr/10
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

    def fit(self, X, y):
        self.train()
        optimizer = self.create_optimizer()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).type(torch.LongTensor).to(self.device)
        loss = torch.tensor(-1.1)
        for epoch in range(self.epoch):
            if epoch%50 == 0:
                acc,_ = self.prediction_accuracy(X,y, False)
                print(f"{epoch}: Loss {loss.item()} "
                      f"Prediction {round(acc,5)}")

            y_hat = self(X)
            loss = self.criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def predict(self, X, temp=False):
        self.eval()
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = self(X)
        _, predicted = torch.max(y, 1)
        if temp:
            self.train()
        return predicted, y

    def prediction_accuracy(self, X, y_true, temp=False):
        total = y_true.shape[0]
        predicted,y_hat = self.predict(X, temp)
        if not torch.is_tensor(y_true):
            y_true = torch.tensor(y_true, dtype=torch.float32).type(torch.LongTensor).to(self.device)
        correct = (predicted == y_true).sum()
        loss = self.criterion(y_hat, y_true)
        return (correct/total).item(), loss.item()

