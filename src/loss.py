import torch
import torch.nn as nn
import numpy as np

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):        
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class WeightedRMSELoss(nn.Module):
    def __init__(self, lat, device, eps=1e-06):
        super().__init__()
        self.weights = np.cos(np.deg2rad(lat)).values
        self.weights /= self.weights.mean()
        self.weights = torch.from_numpy(self.weights).to(device)
        self.eps = eps
        
    def forward(self, y_hat, y):
        error = y_hat - y
        mse = torch.mean(error**2 * self.weights[None, None, :, None])
        return torch.sqrt(mse + self.eps)
    
class WeightedMSELoss(nn.Module):
    def __init__(self, lat, device):
        super().__init__()
        self.weights = np.cos(np.deg2rad(lat)).values
        self.weights /= self.weights.mean()
        self.weights = torch.from_numpy(self.weights).to(device)
        
    def forward(self, y_hat, y):
        error = y_hat - y
        mse = torch.mean(error**2 * self.weights[None, None, :, None])
        return mse

class WeightedMAELoss(nn.Module):
    def __init__(self, lat, device):
        super().__init__()
        self.weights = np.cos(np.deg2rad(lat)).values
        self.weights /= self.weights.mean()
        self.weights = torch.from_numpy(self.weights).to(device)
        
    def forward(self, y_hat, y):
        error = y_hat - y
        mse = torch.mean(torch.abs(error) * self.weights[None, None, :, None])
        return mse