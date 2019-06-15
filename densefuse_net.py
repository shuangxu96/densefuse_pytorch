# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:03:03 2019

@author: win10
"""

import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv2', nn.Conv2d(64,64,3,1,1))
        self.layers.add_module('Act2' , nn.ReLU(inplace=True))
        self.layers.add_module('Conv3', nn.Conv2d(64,32,3,1,1))
        self.layers.add_module('Act3' , nn.ReLU(inplace=True))        
        self.layers.add_module('Conv4', nn.Conv2d(32,16,3,1,1))
        self.layers.add_module('Act4' , nn.ReLU(inplace=True))
        self.layers.add_module('Conv5', nn.Conv2d(16,1,3,1,1))
        
    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        
        self.Conv1 = nn.Conv2d(1,16,3,1,1)
        self.Relu  = nn.ReLU(inplace=True)
        
        self.layers = nn.ModuleDict({
            'DenseConv1': nn.Conv2d(16,16,3,1,1),
            'DenseConv2': nn.Conv2d(32,16,3,1,1),
            'DenseConv3': nn.Conv2d(48,16,3,1,1)
        })
        
    def forward(self, x):
        x = self.Relu(self.Conv1(x))
        for i in range(len(self.layers)):
            out = self.layers['DenseConv'+str(i+1)] ( x )
            x = torch.cat([x,out],1)
        return x

class DenseFuseNet(nn.Module):
    
    def __init__(self):
        super(DenseFuseNet,self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self,x):
        return self.decoder(self.encoder(x))


        
            
        
