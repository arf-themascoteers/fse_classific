import torch.nn as nn
import torch
import torch.nn.functional as F


class BandIndex(nn.Module):
    def __init__(self, val=None, sigmoid=True):
        super().__init__()
        self.sigmoid = sigmoid
        if val is None:
            val = torch.rand(1)
            if self.sigmoid:
                val = (val*10)-5
        self.raw_index = nn.Parameter(val)

    def forward(self, spline):
        outs = spline.evaluate(self.index_value())
        return outs

    def index_value(self):
        if self.sigmoid:
            return F.sigmoid(self.raw_index)
        return self.raw_index

    def range_loss(self):
        if self.sigmoid:
            return 0
        loss_l_lower = F.relu(-1 * self.raw_index)
        loss_l_upper = F.relu(self.raw_index - 1)
        return loss_l_lower + loss_l_upper
