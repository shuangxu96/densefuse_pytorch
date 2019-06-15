# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:32:38 2019

@author: win10
"""
import torch

from densefuse_net import DenseFuseNet
from utils import test

device = 'cuda'

model = DenseFuseNet().to(device)
model.load_state_dict(torch.load('./train_result/model_weight.pkl')['weight'])

test_path = './images/IV_images/'     
test(test_path, model, mode='add')