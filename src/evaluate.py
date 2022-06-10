import xarray as xr

import torch

class Evaluator:
    
    def __init__(self, model, data_loader, loss_fn, device):
        self.model = model
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.device = device
        
    def evaluate(self):
        self.model.eval()
        loader_size = len(self.data_loader)
        cumulative_loss = 0.0
        with torch.no_grad(): 
            for batch_i, (inputs, target) in enumerate(self.data_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)

                # get prediction
                output = self.model(inputs)
                loss = self.loss_fn(output, target)
                cumulative_loss += loss.item()

        return cumulative_loss/loader_size