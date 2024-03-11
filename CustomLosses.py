import torch
import torch.nn as nn

class WMSELoss(nn.Module):
    def __init__(self, weight=1, ignore_pixel=-1):
        super(WMSELoss, self).__init__()
        self.weight = weight
        self.ignore_pixel = ignore_pixel

    def forward(self, y_pred, y_true):
        y_pred = y_pred[y_true != self.ignore_pixel]
        y_true = y_true[y_true != self.ignore_pixel]
        
        # print(y_pred.shape, y_true.shape)
        # print(y_pred.mean(), y_true.mean())

        if self.weight == 1:
            result = torch.mean((y_pred - y_true) ** 2)
        else:
            weights = y_true.clone()
            weights[y_true > 0] = self.weight
            # weights[y_true <= 0] = 0.5
            weights[y_true <= 0] = (1 - self.weight)
            result = torch.mean(weights * (y_pred - y_true) ** 2)
        # Handle NaN
        if torch.isnan(result):
            result = torch.tensor(0, dtype=torch.float32, device=y_pred.device)
        return result
    
class WMAELoss(nn.Module):
    def __init__(self, weight=1, ignore_pixel=-1):
        super(WMAELoss, self).__init__()
        self.weight = weight
        self.ignore_pixel = ignore_pixel

    def forward(self, y_pred, y_true):      
        y_pred = y_pred[y_true != self.ignore_pixel]
        y_true = y_true[y_true != self.ignore_pixel]
        

        if self.weight == 1:
            result = torch.mean(torch.abs(y_pred - y_true))
        else:
            weights = y_true.clone()
            weights[y_true > 0] = self.weight
            # weights[y_true <= 0] = 0.5
            weights[y_true <= 0] = (1 - self.weight)
            result = torch.mean(weights * torch.abs(y_pred - y_true))
        # Handle NaN
        if torch.isnan(result):
            result = torch.tensor(0, dtype=torch.float32, device=y_pred.device)
        return result