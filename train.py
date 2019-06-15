# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:59:40 2019

@author: win10
"""
import os
os.chdir(r'D:\py_code\densefuse_pytorch')

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from densefuse_net import DenseFuseNet
from dataset import AEDataset
from ssim import SSIM 
from utils import mkdir

import os
import scipy.io as scio
import numpy as np
from matplotlib import pyplot as plt

# Parameters
root = 'D:/coco/train2014_train/'
root_val = 'D:/coco/train2014_val/'
train_path = './train_result/'
epochs = 4
batch_size = 2
device = 'cuda'
lr = 1e-4
lambd = 1
loss_interval = 1000
model_interval = 1000
# Dataset
data = AEDataset(root, resize= [256,256], transform = None, gray = True)
loader = DataLoader(data, batch_size = batch_size, shuffle=True)
data_val = AEDataset(root_val, resize= [256,256], transform = None, gray = True)
loader_val = DataLoader(data_val, batch_size = 100, shuffle=True)

# Model
model = DenseFuseNet().to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr = lr)
MSE_fun = nn.MSELoss()
SSIM_fun = SSIM()


# Training
mse_train = []
ssim_train = []
loss_train = []
mse_val = []
ssim_val = []
loss_val = []
mkdir(train_path)
print('============ Training Begins ===============')
for iteration in range(epochs):
    for index, img in enumerate(loader):
        img = img.to(device)
        
        optimizer.zero_grad()
        img_recon = model(img)
        mse_loss = MSE_fun(img,img_recon)
        ssim_loss = 1-SSIM_fun(img,img_recon)
        loss = mse_loss+lambd*ssim_loss
        loss.backward()
        optimizer.step()
        
        
        if index%loss_interval ==0:
            print('[%d,%d] -   Train    - MSE: %.10f, SSIM: %.10f'%
              (iteration,index,mse_loss.item(),ssim_loss.item()))
            mse_train.append(mse_loss.item())
            ssim_train.append(ssim_loss.item())
            loss_train.append(loss.item())
            
            with torch.no_grad():
                tmp1, tmp2 = .0, .0
                for _, img in enumerate(loader_val):
                    img = img.to(device)
                    img_recon = model(img)
                    tmp1 += (MSE_fun(img,img_recon)*img.shape[0]).item()
                    tmp2 += (SSIM_fun(img,img_recon)*img.shape[0]).item()
                tmp3 = tmp1+lambd*tmp2
                mse_val.append(tmp1/data_val.__len__())
                ssim_val.append(tmp1/data_val.__len__())
                loss_val.append(tmp1/data_val.__len__())
            print('[%d,%d] - Validation - MSE: %.10f, SSIM: %.10f'%
              (iteration,index,mse_val[-1],ssim_val[-1]))
            scio.savemat(os.path.join(train_path, 'TrainData.mat'), 
                         {'mse_train': np.array(mse_train),
                          'ssim_train': np.array(ssim_train),
                          'loss_train': np.array(loss_train)})
            scio.savemat(os.path.join(train_path, 'ValData.mat'), 
                         {'mse_val': np.array(mse_val),
                          'ssim_val': np.array(ssim_val),
                          'loss_val': np.array(loss_val)})
        
            plt.figure(figsize=[12,8])
            plt.subplot(2,3,1), plt.semilogy(mse_train), plt.title('mse train')
            plt.subplot(2,3,2), plt.semilogy(ssim_train), plt.title('ssim train')
            plt.subplot(2,3,3), plt.semilogy(loss_train), plt.title('loss train')
            plt.subplot(2,3,4), plt.semilogy(mse_val), plt.title('mse val')
            plt.subplot(2,3,5), plt.semilogy(ssim_val), plt.title('ssim val')
            plt.subplot(2,3,6), plt.semilogy(loss_val), plt.title('loss val')
            
            plt.savefig(os.path.join(train_path,'curve.png'),dpi=90)
        
        if index%model_interval ==0:
            torch.save( {'weight': model.state_dict(), 'epoch':iteration, 'batch_index': index},
                       os.path.join(train_path,'model_weight_new.pkl'))
            print('[%d,%d] - model is saved -'%(iteration,index))
            
