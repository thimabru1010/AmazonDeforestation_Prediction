import torch
import torch.nn as nn

class WMSELoss(nn.Module):
    def __init__(self, weight=1, mask=None):
        super(WMSELoss, self).__init__()
        self.weight = weight
        self.mask = mask

    def forward(self, y_pred, y_true):
        # y_pred = y_pred.view(-1)
        # y_true = y_true.view(-1)
        if self.mask is not None:
            y_pred = y_pred[:, :, :, self.mask != 0]
            y_true = y_true[:, :, :, self.mask != 0]

        weights = y_true.clone()
        weights[y_true > 0] = self.weight
        # weights[y_true <= 0] = 0.5
        weights[y_true <= 0] = (1 - self.weight)
        return torch.mean(weights * (y_pred - y_true) ** 2)