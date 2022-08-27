import torch
from torch import nn
import torch.nn.functional as F

from spherenet import SphereConv2D


class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, filters, kernel=3, bn_position=None, bias=True, dropout=0, activation=nn.LeakyReLU, spheric=False):
        super().__init__()
        
        self.bn_position = bn_position
        self.dropout = dropout
        padding = kernel // 2
        self.spheric = spheric
        
        if not spheric:
            self.periodic_conv = nn.Sequential(
                nn.ZeroPad2d((0, 0, padding, padding)),
                nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel, 
                          padding=(0, padding), padding_mode='circular', bias=bias),
            )
        else:
            self.spheric_conv = SphereConv2D(in_c=in_channels, out_c=filters, bias=bias)
        
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(filters)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        if self.bn_position == 'pre': 
            x = self.bn(x)
        
        if self.spheric:
            x = self.spheric_conv(x)
        else:
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
    def __init__(self, filters, kernel, bn_position=None, bias=True, dropout=0, skip=True, activation=nn.LeakyReLU, spheric=False):
        super().__init__()
        
        self.conv_operation = nn.Sequential()
        
        for i in range(2):
            self.conv_operation.add_module(
                f'conv_0{i + 1}',
                ConvBlock(in_channels=filters, filters=filters, kernel=kernel, bn_position=bn_position, bias=bias, 
                          dropout=dropout, activation=activation, spheric=spheric)
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
        
        for i, (f, k) in enumerate(zip(filters[1:-1], kernels[1:-1])):
            self.res_block.add_module(f'res_block_{i}',
                                      ResBlock(f, k, bn_position=bn_position, bias=bias, dropout=dropout, 
                                               skip=skip, activation=activation)
                                    )
            
        self.end_conv = ConvBlock(in_channels=filters[-2], filters=filters[-1], kernel=kernels[-1], bn_position=bn_position, 
                                  bias=bias, dropout=dropout, activation=activation).periodic_conv
        #output = Activation('linear', dtype='float32')(output)
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_block(x)
        x = self.end_conv(x)
        return x
    
class SphericResNet(nn.Module):
    
    def __init__(self, in_channels, filters, kernels, bn_position=None, skip=True, bias=True, dropout=0, activation=nn.LeakyReLU):
        super().__init__()
        
        self.init_conv = ConvBlock(in_channels=in_channels, filters=filters[0], kernel=kernels[0], bn_position=bn_position, 
                                   bias=bias, dropout=dropout, activation=activation, spheric=True)
        
        self.res_block = nn.Sequential()
        
        for i, (f, k) in enumerate(zip(filters[1:-1], kernels[1:-1])):
            self.res_block.add_module(f'spheric_res_block_{i}',
                                      ResBlock(f, k, bn_position=bn_position, bias=bias, dropout=dropout, 
                                               skip=skip, activation=activation, spheric=True)
                                    )
            
        self.end_conv = ConvBlock(in_channels=filters[-2], filters=filters[-1], kernel=kernels[-1], bn_position=bn_position, 
                                  bias=bias, dropout=dropout, activation=activation, spheric=True).spheric_conv
        #output = Activation('linear', dtype='float32')(output)
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_block(x)
        x = self.end_conv(x)
        return x
    
    
    
