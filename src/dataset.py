import re
import datetime

import xarray as xr
import numpy as np

from torch.utils.data import Dataset, DataLoader

class WeatherBenchDataset(Dataset):
    
    def __init__(self, dataset, var_dict, lead_time, output_vars, nt_in=3, dt_in=6, mean=None, std=None, 
                 normalize=True, cont_time=True, fixed_time=True, multi_dt=1, discard_first=None,
                 data_subsample=1, norm_subsample=1, tp_log=None, load=True, load_part_size=0, verbose=1):
        '''
        - nt_in: number of input time steps
        - dt_in: distance in hours between input time steps
        - discard_first: number of initial values to discard
        - load_part_size: if 0: load the entire dataset to memory; else: load only the specified size
        '''
        
        super(WeatherBenchDataset, self).__init__()
        self.lead_time = lead_time
        self.nt_in = nt_in
        self.dt_in = dt_in
        self.cont_time = cont_time
        self.fixed_time = fixed_time
        self.multi_dt = multi_dt
        self.load_part_size = load_part_size
        
        data = []
        level_names = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for long_var, params in var_dict.items():
            if long_var == 'constants':
                for var in params:
                    data.append(dataset[var].expand_dims(
                        {'level': generic_level, 'time': dataset.time}, (1, 0)
                    ))
                    level_names.append(var)
            else:
                var, levels = params
                da = dataset[var]
                try:
                    data.append(da.sel(level=levels))
                    level_names += [f'{var}_{level}' for level in levels]
                except KeyError:
                    data.append(da.expand_dims({'level': generic_level}, 1))
                    level_names.append(var)

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        
        if discard_first is not None:
            self.data = self.data.isel(time=slice(discard_first, None))
        self.data['level_names'] = xr.DataArray(
            level_names, dims=['level'], coords={'level': self.data.level}
        )
        
        self.output_idxs = [i for i, l in enumerate(self.data.level_names.values)
                            if any([bool(re.match(o, l)) for o in output_vars])]
        self.const_idxs = [i for i, l in enumerate(self.data.level_names) if l in var_dict['constants']]
        
        self.not_const_idxs = [i for i, l in enumerate(self.data.level_names) if l not in var_dict['constants']]
        
        # Subsample
        self.data = self.data.isel(time=slice(0, None, data_subsample))
        self.dt = self.data.time.diff('time')[0].values / np.timedelta64(1, 'h')
        self.dt_in = int(self.dt_in // self.dt)
        self.nt_offset = (nt_in - 1) * self.dt_in
        
        # Normalize
        if verbose: print('DS normalize', datetime.datetime.now().time())
        if mean is not None:
            self.mean = mean
        else:
            self.mean = self.data.isel(time=slice(0, None, norm_subsample)).mean(
                ('time', 'lat', 'lon')).compute()
            if 'tp' in self.data.level_names:  # set tp mean to zero but not if ext
                tp_idx = list(self.data.level_names).index('tp')
                self.mean.values[tp_idx] = 0
                                
        if std is not None:
            self.std = std
        else:
            self.std = self.data.isel(time=slice(0, None, norm_subsample)).std(
                ('time', 'lat', 'lon')).compute()

        if tp_log is not None:
            self.mean.attrs['tp_log'] = tp_log
            self.std.attrs['tp_log'] = tp_log
        if normalize:
            self.data = (self.data - self.mean) / self.std
        
        if verbose: print('DG load', datetime.datetime.now().time())
        
        if load:
            if self.load_part_size == 0:
                self.data.load()
            else:
                self.loaded_data = self.data
                # (self.nt + self.nt_offset) - load this extra size to obtain the 'y' to last index exanple
                self.loaded_data = self.loaded_data.isel(time=slice(0, self.load_part_size + (self.nt + self.nt_offset)))
                self.loaded_init_idx = 0
                self.loaded_data.load()
            
        if verbose: print('DG done', datetime.datetime.now().time())
        
    @property
    def nt(self):
        assert (self.lead_time / self.dt).is_integer(), "lead_time and dt not compatible."
        return int(self.lead_time / self.dt)
    
    @property
    def valid_time(self):
        start = self.nt + self.nt_offset
        stop = None
        if self.multi_dt > 1:
            diff = self.nt - self.nt // self.multi_dt
            start -= diff 
            stop = -diff
        
        return self.data.isel(time=slice(start, stop)).time
        
    def __getitem__(self, index):
        
        if self.load_part_size: 
            loaded_end_idx = self.loaded_init_idx + self.load_part_size
            if index < self.loaded_init_idx or index >= loaded_end_idx:
                self.loaded_data = self.data
                
                i = index // self.load_part_size # ith part to load
                init_idx = i * self.load_part_size
                end_idx = ((i + 1) * self.load_part_size) + (self.nt + self.nt_offset)
                
                self.loaded_data = self.loaded_data.isel(time=slice(init_idx, end_idx))
                
                self.loaded_init_idx = init_idx
                self.loaded_data.load()

            index = index - self.loaded_init_idx
            
            data = self.loaded_data
        else:
            data = self.data
        
        
        index = index + ((self.nt_in -1) * self.dt_in)
        nt = self.nt
        
        if self.cont_time: # previsao continua (diferente de direta)
            if not self.fixed_time:
                # gera randomicamente o time step que deve ser previsto quando este não é fixo
                # altera o valor de nt para esse número randomicamente gerado
                nt = np.random.randint(self.min_nt, self.nt + 1)
                
            # a partir do time step que deve ser previsto, cria uma nova variável a ser passada para o modelo
            ftime = (np.array([nt]) * self.dt / 100) * np.ones((len(data.lat), len(data.lon)))

        X_data = data

        X = X_data.isel(time=index).values.astype('float32')
        
        # seria quando considera mais de um time step como entrada ou como predicao??
        if self.multi_dt > 1: consts = X[..., self.const_idxs]

        if self.nt_in > 1:
            X = np.concatenate([
                                   # remove consts to avoid repitition, they are already presente in X
                                   X_data.isel(time=index - nt_in * self.dt_in).values[..., self.not_const_idxs]
                                   for nt_in in range(self.nt_in - 1, 0, -1)
                               ] + [X], axis=-1).astype('float32')

        if self.multi_dt > 1:
            X = [X[..., self.not_const_idxs], consts]
            step = self.nt // self.multi_dt
            y = [
                data.isel(time=index + nt, level=self.output_idxs).values.astype('float32')
                for nt in np.arange(step, self.nt + step, step)
            ]
        else:
            y = data.isel(time=index + nt, level=self.output_idxs).values.astype('float32')

        if self.cont_time:  # no caso de predicao continua, adiciona a variavel referente ao time step 
                            # a ser previsto ao canal
            X = np.concatenate([X, ftime[..., None]], -1).astype('float32')
        
        X = np.transpose(X, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))
        return X, y
        
    def __len__(self):
        return self.data.shape[0] - (self.nt + self.nt_offset)