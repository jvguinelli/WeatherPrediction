import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, filters, kernel=3, bn_position=None, bias=True, dropout=0, activation=nn.LeakyReLU):
        super().__init__()
        
        self.bn_position = bn_position
        self.dropout = dropout
        padding = kernel // 2
        
        self.periodic_conv = nn.Sequential(
            nn.ZeroPad2d((0, 0, padding, padding)),
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel, 
                      padding=(0, padding), padding_mode='circular', bias=bias),
        )
        
        self.bn = nn.BatchNorm2d(filters)
        self.activation = nn.LeakyReLU()
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        if self.bn_position == 'pre': 
            x = self.bn(x)
        x = self.periodic_conv(x)
        if self.bn_position == 'mid': 
            x = self.bn(x)
        x = self.activation(x)
        if self.bn_position == 'post': 
            x = self.bn(x)
        
        if self.dropout > 0: 
            x = self.dropout_layer(x)
        
        return x
    
class ResBlock(nn.Module):
    def __init__(self, filters, kernel, bn_position=None, bias=True, dropout=0, skip=True, activation=nn.LeakyReLU):
        super().__init__()
        
        self.conv_operation = nn.Sequential()
        
        for i in range(2):
            self.conv_operation.add_module(
                f'conv_0{i + 1}',
                ConvBlock(in_channels=filters, filters=filters, kernel=kernel, bn_position=bn_position, bias=bias, 
                          dropout=dropout, activation=activation)
                )
            
        self.skip = skip
        
    def forward(self, x):
        inputs = x
        x = self.conv_operation(x)
        if self.skip: 
            x = inputs + x
            
        return x
    
    
class BasicResNet(nn.Module):
    
    def __init__(self, in_channels, filters, kernels, bn_position=None, skip=True, bias=True, dropout=0, activation=nn.LeakyReLU):
        super().__init__()
        
        self.init_conv = ConvBlock(in_channels=in_channels, filters=filters[0], kernel=kernels[0], bn_position=bn_position, 
                                   bias=bias, dropout=dropout, activation=activation)
        
        self.res_block = nn.Sequential()
        
        for f, k in zip(filters[1:-1], kernels[1:-1]):
            self.res_block.add_module('res_block',
                                      ResBlock(f, k, bn_position=bn_position, bias=bias, dropout=dropout, 
                                               skip=skip, activation=activation)
                                    )
            
        self.end_conv = ConvBlock(in_channels=filters[-2], filters=filters[-1], kernel=kernels[-1], bn_position=bn_position, 
                                  bias=bias, dropout=dropout, activation=activation)
        #output = Activation('linear', dtype='float32')(output)
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_block(x)
        x = self.end_conv(x)
        return x