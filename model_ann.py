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
        self.epoch = my_utils.get_epoch(rows, self.target_feature_size)
        self.lr = my_utils.get_lr(rows, self.target_feature_size)
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
        loss = torch.tensor(-1)
        for epoch in range(self.epoch):
            if epoch%50 == 0:
                print(f"{epoch}: Loss {round(loss.item(),5)} "
                      f"Prediction {round(self.prediction_accuracy(X,y, False),5)}")

            y_hat = self(X)
            loss = self.criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


    def predict(self, X, temp=False):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = self(X)
        _, predicted = torch.max(y, 1)
        if temp:
            self.train()
        return predicted.detach().cpu().numpy()

    def prediction_accuracy(self, X, y_true, temp=False):
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.detach().cpu().numpy()
        total = y_true.shape[0]
        predicted = self.predict(X, temp)
        correct = (predicted == y_true).sum()
        return correct/total

