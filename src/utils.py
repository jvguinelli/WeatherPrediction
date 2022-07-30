import os
from datetime import datetime

import xarray as xr
import numpy as np
import torch


class Utils:
    
    def __init__(self, root_folder, model_name):
        self.experiment_id = datetime.now().strftime('%d%m%Y_%H%M%S')
        model_folder_path = os.path.join(root_folder, model_name)
        
        if not os.path.isdir(model_folder_path):
            os.mkdir(model_folder_path)
            
        self.path = os.path.join(model_folder_path, self.experiment_id)
            
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
            
    def save_losses(self, epoch, train_loss, val_loss):
        with open(os.path.join(self.path, 'losses.csv'), 'a') as f:
            f.write(f'{epoch}, {train_loss}, {val_loss}\n')
            
    @staticmethod
    def load_checkpoint(experiment_id, root_folder, model_name, filename=None):
        if experiment_id:
            filename = os.path.join(root_folder, model_name, experiment_id, 'checkpoint.pth')  
        epoch, loss = 0.0, 0.0
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            name = os.path.basename(filename)
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f'=> Loaded checkpoint {name} (best epoch: {epoch}, validation rmse: {loss:.4f})')
        else:
            print(f'=> No checkpoint found at {filename}')

        return checkpoint
    
def load_test_data(path, var, years=slice('2017', '2018'), cmip=False):
    """
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window
    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    
    assert var in ['z', 't'], 'Test data only for Z500 and T850'
    ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
    try:
        ds = ds.sel(level=500 if var == 'z' else 850).drop('level')
    except ValueError:
        pass
    return ds.sel(time=years)


    
def create_predictions(model, data_loader, multi_dt=False, no_mean=False, device=None):
    """Create non-iterative predictions"""
    level_names = data_loader.dataset.data.isel(level=data_loader.dataset.output_idxs).level_names
    level = data_loader.dataset.data.isel(level=data_loader.dataset.output_idxs).level

    mean = data_loader.dataset.mean.isel(level=data_loader.dataset.output_idxs).values[None, :, None, None] if not no_mean else 0
    std = data_loader.dataset.std.isel(level=data_loader.dataset.output_idxs).values[None, :, None, None]

    # my implementation

    model.eval()
    outputs = []
    with torch.no_grad(): 
        for batch_i, (inputs, target) in enumerate(data_loader):
            inputs, target = inputs.to(device), target.to(device)
            outputs.append(model(inputs))
            
    preds = torch.cat(outputs).cpu()

    # end my implementation

    preds = xr.DataArray(
        preds[0] if multi_dt else preds,
        dims=['time', 'level', 'lat', 'lon'],
        coords={'time': data_loader.dataset.valid_time, 
                'lat': data_loader.dataset.data.lat, 'lon': data_loader.dataset.data.lon,
                'level': level,
                'level_names': level_names
                },
    )

    preds = preds * std + mean

    unique_vars = list(set([l.split('_')[0] for l in preds.level_names.values]))

    # Reverse tranforms
    if hasattr(data_loader.dataset.mean, 'tp_log') and 'tp' in unique_vars:
        tp_idx = list(preds.level_names).index('tp')
        preds.values[..., tp_idx] = log_retrans(preds.values[..., tp_idx], data_loader.dataset.mean.tp_log)

    das = []
    for v in unique_vars:
        idxs = [i for i, vv in enumerate(preds.level_names.values) if vv.split('_')[0] == v]
        da = preds.isel(level=idxs).squeeze().drop('level_names')
        if not 'level' in da.dims: da = da.drop('level')
        das.append({v: da})
    return xr.merge(das)

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse