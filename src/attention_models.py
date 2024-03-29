import torch
from torch import nn
import torch.nn.functional as F

from positional_encodings import PositionalEncodingPermute2D, Summer, PositionalEncoding2D
from gsa_pytorch import GSA
from axial_attention import AxialAttention, AxialPositionalEmbedding

from .convlstm import ConvLSTM


class BasicGSA(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=1, num_layers=5, heads=8, dim_key=32, rel_pos_length=3):
        super().__init__()
        
        self.init_conv = nn.Sequential(
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=7, padding=(0, 3), padding_mode='circular'),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU()
        )
        
        self.positional_encoding = Summer(PositionalEncodingPermute2D(hidden_dim))
        
        self.attention = nn.Sequential()
        
        for i in range(num_layers):
            self.attention.add_module(
                f'GSA_{i}', nn.Sequential(GSA(dim = hidden_dim, 
                                              dim_out=hidden_dim,
                                              dim_key = dim_key,
                                              heads=heads,
                                              rel_pos_length=rel_pos_length
                                             ),
                                          nn.BatchNorm2d(hidden_dim)
                                         )
            )
        
        self.end_conv = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, padding=(0, 1), padding_mode='circular')
        )
        
        #for module in self.modules():
        #    if isinstance(module, torch.nn.modules.BatchNorm2d):
        #        module.track_running_stats = False
        #        module.running_mean = None
        #        module.running_var = None        

    def forward(self, x):
        x = self.init_conv(x)
        x = self.positional_encoding(x)
        x = self.attention(x)
        x = self.end_conv(x)
        return x

    
class ConvGSA(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=1, num_layers=5, heads=8, dim_key=32, rel_pos_length=3):
        super().__init__()
        
        self.init_conv = nn.Sequential(
            CircularPad3d([1, 1, 0, 0, 0, 0]),
            nn.ConstantPad3d([0, 0, 1, 1, 0, 0], 0),
            nn.Conv3d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(hidden_dim),
            nn.LeakyReLU()
        )
        
        self.positional_encoding = Summer(PositionalEncodingPermute2D(hidden_dim))
        
        self.attention = nn.Sequential()
        
        for i in range(num_layers):
            self.attention.add_module(
                f'GSA_{i}', nn.Sequential(GSA(dim = hidden_dim, 
                                              dim_out=hidden_dim,
                                              dim_key = dim_key,
                                              heads=heads,
                                              rel_pos_length=rel_pos_length
                                             ),
                                          nn.BatchNorm2d(hidden_dim)
                                         )
            )
        
        self.end_conv = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, padding=(0, 1), padding_mode='circular')
        )
        
        
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.momentum = 0.2

    def forward(self, x):
        x = self.init_conv(x)
        x = torch.squeeze(x, dim=2)
        x = self.positional_encoding(x)
        x = self.attention(x)
        x = self.end_conv(x)
        return x
    

class ConvLSTMGSA(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=1, num_layers=5, heads=8, dim_key=32, rel_pos_length=3):
        super().__init__()
        
        self.init_convlstm = ConvLSTM(input_dim=in_channels, 
                                      hidden_dim=128, 
                                      kernel_size=(1, 1),
                                      num_layers=1,
                                      batch_first=True
                                     )
        
        
        self.dropout = nn.Dropout(0.3)
        
        self.positional_encoding = Summer(PositionalEncodingPermute2D(hidden_dim))
        
        self.attention = nn.Sequential()
        
        for i in range(num_layers):
            self.attention.add_module(
                f'GSA_{i}', nn.Sequential(GSA(dim = hidden_dim, 
                                              dim_out=hidden_dim,
                                              dim_key = dim_key,
                                              heads=heads,
                                              rel_pos_length=rel_pos_length
                                             ),
                                          nn.BatchNorm2d(hidden_dim)
                                         )
            )
        
        self.end_conv = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, padding=(0, 1), padding_mode='circular')
        )
        
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.momentum = 0.2

    def forward(self, x):
        _, lst_states = self.init_convlstm(x)
        x = lst_states[0][0]
        x = self.dropout(x)
        x = self.positional_encoding(x)
        x = self.attention(x)
        x = self.end_conv(x)
        return x
    
    
class ConvLSTMAxial(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_channels=1, num_layers=5, heads=8, dim_key=32, rel_pos_length=3):
        super().__init__()
        
        self.init_convlstm = ConvLSTM(input_dim=in_channels, 
                                      hidden_dim=128, 
                                      kernel_size=(3, 3),
                                      num_layers=1,
                                      batch_first=True
                                     )
        
        self.positional_encoding = AxialPositionalEmbedding(dim=hidden_dim, shape=(32, 64))
        
        self.attention = nn.Sequential()
        
        for i in range(num_layers):
            self.attention.add_module(
                f'GSA_{i}', nn.Sequential(AxialAttention(dim = hidden_dim, 
                                                         dim_index=1,
                                                         dim_heads=dim_key,
                                                         heads=heads,
                                                         num_dimensions=2, 
                                                        )
                                         )
            )
        
        self.end_conv = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, padding=(0, 1), padding_mode='circular')
        )
        
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.momentum = 0.2

    def forward(self, x):
        _, lst_states = self.init_convlstm(x)
        x = lst_states[0][0]
        x = self.positional_encoding(x)
        x = self.attention(x)
        x = self.end_conv(x)
        return x
    
    
class CircularPad3d(nn.Module):
    def __init__(self, padding):
        super(CircularPad3d, self).__init__()
        self.padding = padding
    
    def forward(self, input):
        return F.pad(input, self.padding, 'circular')

    def extra_repr(self):
        return '{}'.format(self.padding)