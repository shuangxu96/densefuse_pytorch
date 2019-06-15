# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:32:22 2019

@author: win10
"""
from PIL import Image
import os
import string
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

_tensor = transforms.ToTensor()
_pil_rgb    = transforms.ToPILImage('RGB')
_pil_gray = transforms.ToPILImage()
device = 'cuda'

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def load_img(img_path, img_type='gray'):
    img = Image.open(img_path)
    if img_type=='gray':
        img = img.convert('L')
    return _tensor(img).unsqueeze(0)

class Strategy(nn.Module):
    def __init__(self, mode='add', window_width=1):
        super().__init__()
        self.mode = mode
        if self.mode == 'l1':
            self.window_width = window_width
            
    def forward(self, y1, y2):
        if self.mode == 'add':
            return (y1+y2)/2
        
        if self.mode == 'l1':
            ActivityMap1 = y1.abs()
            ActivityMap2 = y2.abs()
            
            kernel = torch.ones(2*self.window_width+1,2*self.window_width+1)/(2*self.window_width+1)**2
            kernel = kernel.to(device).type(torch.float32)[None,None,:,:]
            kernel = kernel.expand(y1.shape[1],y1.shape[1],2*self.window_width+1,2*self.window_width+1)
            ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=self.window_width)
            ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=self.window_width)
            WeightMap1 = ActivityMap1/(ActivityMap1+ActivityMap2)
            WeightMap2 = ActivityMap2/(ActivityMap1+ActivityMap2)
            return WeightMap1*y1+WeightMap2*y2

def fusion(x1,x2,model,mode='l1', window_width=1):
    with torch.no_grad():
        fusion_layer  = Strategy(mode,window_width).to(device)
        feature1 = model.encoder(x1)
        feature2 = model.encoder(x2)
        feature_fusion = fusion_layer(feature1,feature2)
        return model.decoder(feature_fusion).squeeze(0).detach().cpu()

class Test:
    def __init__(self):
        pass
        
    def load_imgs(self, img1_path,img2_path, device):
        img1 = load_img(img1_path,img_type=self.img_type).to(device)
        img2 = load_img(img2_path,img_type=self.img_type).to(device)
        return img1, img2
    
    def save_imgs(self, save_path,save_name, img_fusion):
        mkdir(save_path)
        save_path = os.path.join(save_path,save_name)
        img_fusion.save(save_path)

class test_gray(Test):
    def __init__(self):
        self.img_type = 'rgray'
    
    def get_fusion(self,img1_path,img2_path,model,
                   save_path = './test_result/', save_name = 'none', mode='l1',window_width=1):
        img1, img2 = self.load_imgs(img1_path,img2_path,device)
        
        img_fusion = fusion(x1=img1,x2=img2,model=model,mode=mode,window_width=window_width)
        img_fusion = _pil_gray(img_fusion)
        
        self.save_imgs(save_path,save_name, img_fusion)
        return img_fusion

class test_rgb(Test):
    def __init__(self):
        self.img_type = 'rgb'
        
    def get_fusion(self,img1_path,img2_path,model,
                   save_path = './test_result/', save_name = 'none', mode='l1',window_width=1):
        img1, img2 = self.load_imgs(img1_path,img2_path,device)
        
        img_fusion = _pil_rgb(torch.cat(
                             [fusion(img1[:,i,:,:][:,None,:,:], 
                             img2[:,i,:,:][:,None,:,:], model,
                             mode=mode,window_width=window_width) 
                             for i in range(3)],
                            dim=0))
                             
        self.save_imgs(save_path,save_name, img_fusion)
        return img_fusion
    
    
def test(test_path, model, img_type='gray', save_path='./test_result/',mode='l1',window_width=1):
    img_list = glob(test_path+'*')
    img_num = len(img_list)/2
    suffix = img_list[0].split('.')[-1]
    img_name_list = list(set([img_list[i].split('\\')[-1].split('.')[0].strip(string.digits) for i in range(len(img_list))]))
    
    if img_type == 'gray':    
        fusion_phase = test_gray()
    elif img_type == 'rgb':
        fusion_phase = test_rgb()
    
    for i in range(int(img_num)):
        img1_path = test_path+img_name_list[0]+str(i+1)+'.'+suffix
        img2_path = test_path+img_name_list[1]+str(i+1)+'.'+suffix
        save_name = 'fusion'+str(i+1)+'_'+img_type+'_'+mode+'.'+suffix
        fusion_phase.get_fusion(img1_path,img2_path,model,
                   save_path = save_path, save_name = save_name, mode=mode,window_width=window_width)
