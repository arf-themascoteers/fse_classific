import torch.nn as nn
import torch
import torch.nn.functional as F
from algorithms.fscr.band_index import BandIndex
import my_utils


class ANN(nn.Module):
    def __init__(self, rows, target_feature_size, sigmoid=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_feature_size = target_feature_size
        # self.linear = nn.Sequential(
        #     nn.Linear(self.target_feature_size, 15),
        #     nn.LeakyReLU(),
        #     nn.Linear(15, 10),
        #     nn.LeakyReLU(),
        #     nn.Linear(10, 1)
        # )
        self.linear = nn.Sequential(
            nn.Linear(self.target_feature_size, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1)
        )
        init_vals = torch.linspace(0.001,0.99, target_feature_size+2)
        modules = []
        for i in range(self.target_feature_size):
            modules.append(BandIndex( ANN.inverse_sigmoid_torch(init_vals[i+1]),sigmoid))
        self.machines = nn.ModuleList(modules)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, spline, size):
        outputs = torch.zeros(size, self.target_feature_size, dtype=torch.float32).to(self.device)
        for i,machine in enumerate(self.machines):
            outputs[:,i] = machine(spline)
        soc_hat = self.linear(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def retention_loss(self):
        loss = None
        for i in range(3, len(self.machines)):
            later_band = self.machines[i].raw_index
            past_band = self.machines[i-1].raw_index
            this_loss = -1 * torch.square(past_band-later_band)
            if loss is None:
                loss = this_loss
            else:
                loss = loss + this_loss
        return loss

    def get_indices(self):
        return [machine.index_value() for machine in self.machines]

