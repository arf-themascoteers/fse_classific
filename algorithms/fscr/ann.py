import torch.nn as nn
import torch
from algorithms.fscr.band_index import BandIndex


class ANN(nn.Module):
    def __init__(self, target_feature_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_feature_size = target_feature_size
        self.linear = nn.Sequential(
            nn.Linear(self.target_feature_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 5)
        )
        init_vals = torch.linspace(0.001,0.99, target_feature_size+2)
        modules = []
        for i in range(self.target_feature_size):
            modules.append(BandIndex( ANN.inverse_sigmoid_torch(init_vals[i+1])))
        self.machines = nn.ModuleList(modules)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, spline, size):
        outputs = torch.zeros(size, self.target_feature_size, dtype=torch.float32).to(self.device)
        for i,machine in enumerate(self.machines):
            outputs[:,i] = machine(spline)
        soc_hat = self.linear(outputs)
        return soc_hat

    def get_indices(self):
        return [machine.index_value() for machine in self.machines]

