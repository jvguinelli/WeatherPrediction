import warnings
warnings.filterwarnings('ignore')

import ast, os
import configargparse as arg
import random
import traceback

import xarray as xr
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.dataset import WeatherBenchDataset
from src.attention_models import WeatherPred
from src.loss import WeightedMAELoss, WeightedMSELoss, WeightedRMSELoss
from src.train import Trainer, EarlyStopping
from src.evaluate import Evaluator
from src.utils import Utils, load_test_data, create_predictions, compute_weighted_rmse

def get_arguments(my_config=None):
    parser = arg.ArgParser()
    parser.add_argument('-c', '--my-config', is_config_file=True, help='config file path', default=my_config)
    parser.add_argument('-e', '--epochs', type=int, default=40)
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('-p', '--patience', type=int, default=5)
    parser.add_argument('-w', '--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_check_interval', type=int, default=800)
    
    parser.add_argument('-l', '--num_layers', type=int, default=8)
    parser.add_argument('-d', '--hidden_dim', type=int, default=128)
    parser.add_argument('-k', '--kernels', type=int, nargs='+', default=None, help='Kernel size for each layer')
    parser.add_argument('--heads', type=int, default=8)
    #parser.add_argument('--filters', type=int, nargs='+', required=True, help='Filters for each layer')
    parser.add_argument('--dim_key', type=int, default=32) 
    parser.add_argument('--rel_pos_length', type=int, default=3) 
    
    parser.add_argument('--network_type', type=str, default='gsa-net', help='Type')
    
    
    parser.add_argument('--train_years', type=str, nargs='+', default=('1979', '2015'), help='Start/stop years for training')
    parser.add_argument('--valid_years', type=str, nargs='+', default=('2016', '2016'), help='Start/stop years for validation')
    parser.add_argument('--test_years', type=str, nargs='+', default=('2017', '2018'), help='Start/stop years for testing')
    parser.add_argument('--shuffle_train', action='store_true', help='Indicate if training data should be shuffled')
    
    parser.add_argument('--data_subsample', type=int, default=1, help='Subsampling for training data')
    parser.add_argument('--norm_subsample', type=int, default=1, help='Subsampling for mean/std')
    parser.add_argument('--discard_first', type=int, default=None, help='Discard first x time steps in train generator')
    
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data')
    parser.add_argument('--exp_id', type=str, help='Experiment identifier')
    parser.add_argument('--lead_time', type=int, required=True, help='Forecast lead time')
    parser.add_argument('--nt_in', type=int, default=1, help='Number of input time steps')
    parser.add_argument('--dt_in', type=int, default=1, help='Time step of intput time steps (after subsampling)')
    parser.add_argument('--in_vars', required=True, help='Variables: as an ugly dictionary...')
    parser.add_argument('--out_vars', nargs='+', help='Output variables. Format {var}_{level}', default=None)
    
    parser.add_argument('--ext_mean', type=str, default='./mean.nc', help='External normalization mean')
    parser.add_argument('--ext_std', type=str, default='./std.nc', help='External normalization std')
    parser.add_argument('--cont_time', type=int, default=0, help='Continuous time 0/1')
    parser.add_argument('--multi_dt', type=int, default=1, help='Differentiate through multiple time steps')
    parser.add_argument('--load_part_size', type=int, default=26280, help='if 0: load the entire train dataset to memory; else: load only the specified size') # default equivalent to 3 years
    
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no_stop', action='store_true', dest='no_stop')
      
    args = parser.parse_args() if my_config is None else parser.parse_args(args=[])
    args.in_vars = ast.literal_eval(args.in_vars)
    return args

def define_seed(number): 
    # define a different seed in every iteration 
    seed = (number * 10) + 1000
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

def init_seed(number):
    seed = (number * 10) + 1000
    np.random.seed(seed)    

def run(config):
    define_seed(42)
    
    if not os.path.isdir('./output'):
        os.mkdir('./output')
        
    ds = xr.merge(
        [xr.open_mfdataset(f'{config.data_dir}/{var}/*.nc', combine='by_coords') for var in config.in_vars.keys()],
        fill_value=0  # For the 'tisr' NaNs
    )
    
    train_data = ds.sel(time=slice(*config.train_years))
    val_data = ds.sel(time=slice(*config.valid_years))
    test_data = ds.sel(time=slice(*config.test_years))
    
    ds_mean = xr.open_dataarray(config.ext_mean)
    ds_std = xr.open_dataarray(config.ext_std)
    
    ds_train = WeatherBenchDataset(train_data, config.in_vars, lead_time=config.lead_time, output_vars=config.out_vars, 
                                   nt_in=config.nt_in, dt_in=config.dt_in, mean=ds_mean, std=ds_std, normalize=True, 
                                   cont_time=True, multi_dt=config.multi_dt, discard_first=config.discard_first, 
                                   data_subsample=config.data_subsample, norm_subsample=config.norm_subsample, tp_log = None, 
                                   load=True, load_part_size=config.load_part_size, verbose=config.verbose)

    ds_val = WeatherBenchDataset(val_data, config.in_vars, lead_time=config.lead_time, output_vars=config.out_vars, 
                                 nt_in=config.nt_in, dt_in=config.dt_in, mean=ds_train.mean, std=ds_train.std, normalize=True,
                                 cont_time=True, multi_dt=config.multi_dt, data_subsample=config.data_subsample, 
                                 norm_subsample=config.norm_subsample, tp_log=None, load=True, verbose=config.verbose)
    
    params = {'batch_size': config.batch, 
              'num_workers': config.workers, 
              'worker_init_fn': init_seed}
    
    train_loader = DataLoader(dataset=ds_train, shuffle=config.shuffle_train, **params)
    val_loader = DataLoader(dataset=ds_val, shuffle=False, **params)
    
    in_channels = ds_train[0][0].shape[0]
    out_channels = ds_train[0][1].shape[0]
    
    print('Train X shape:', ds_train[0][0].shape, 'Train Y shape:', ds_train[0][1].shape)
    print('Valid X shape:', ds_val[0][0].shape, 'Valid Y shape:', ds_val[0][1].shape)
    
    model = WeatherPred(in_channels=in_channels, hidden_dim=config.hidden_dim, out_channels=out_channels, 
                        num_layers=config.num_layers, heads=config.heads, dim_key=config.dim_key, rel_pos_length=config.rel_pos_length)
    model_name = model.__class__.__name__
    model = model.to(config.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), weight_decay=1e-5)
    loss_fn = WeightedMAELoss(ds.lat, config.device)
    evaluator = Evaluator(model, val_loader, loss_fn, config.device)
    
    util = Utils('./output', model_name)
  
    checkpoint_file_path = os.path.join(util.path, 'checkpoint.pth')
    early_stopping = EarlyStopping(filename=checkpoint_file_path, patience=config.patience, no_stop=config.no_stop, verbose=config.verbose)
    
    trainer = Trainer(model, train_loader, optimizer, loss_fn, early_stopping, evaluator, util, config.device, verbose=config.verbose)
    trainer.fit(epochs=config.epochs, val_check_interval=config.val_check_interval)
    
    checkpoint = Utils.load_checkpoint(experiment_id=util.experiment_id, root_folder='./output', model_name=model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    del(ds_val)
    del(train_loader)
    del(val_loader)
    
    ds_test = WeatherBenchDataset(test_data, config.in_vars, lead_time=config.lead_time, output_vars=config.out_vars, 
                                  nt_in=config.nt_in, dt_in=config.dt_in, mean=ds_train.mean, std=ds_train.std, normalize=True,
                                  cont_time=True, multi_dt=config.multi_dt, data_subsample=config.data_subsample,
                                  norm_subsample=config.norm_subsample, tp_log=None, load=True, verbose=config.verbose)
    
    test_loader = DataLoader(dataset=ds_test, shuffle=False, **params)
    
    preds = create_predictions(model, test_loader, device=config.device)
    
    z500_valid = load_test_data(f'{config.data_dir}geopotential', 'z', years=slice(*config.test_years))
    
    test_loss = compute_weighted_rmse(preds, z500_valid).load()
        
    print(f'WeightedRMSELoss ({config.lead_time / 24} days): {test_loss}')

if __name__ == '__main__':
    args = get_arguments()
    
    device = torch.device('cpu')
    device_descr = 'CPU'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_descr = 'GPU'
    
    args.device = device
        
    #print(f'RUN MODEL: {args.model.upper()}')
    print(f'RUN MODEL')
    print(f'Device: {device_descr}') 
    print(f'Settings: {args}')
    
    try:
        run(args)
    except Exception as e:
        traceback.print_exc()
        message = '=> Error: ' + str(e)
    
    print(f'END \n')