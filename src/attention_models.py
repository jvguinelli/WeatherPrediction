import torch
from torch import nn

from positional_encodings import PositionalEncodingPermute2D, Summer, PositionalEncoding2D
from gsa_pytorch import GSA

class WeatherPred(nn.Module):
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

    def forward(self, x):
        x = self.init_conv(x)
        x = self.positional_encoding(x)
        x = self.attention(x)
        x = self.end_conv(x)
        return x