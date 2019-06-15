# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:03:42 2019

@author: win10
"""

import torch.utils.data as Data
import torchvision.transforms as transforms

from glob import glob
import os
from PIL import Image

class AEDataset(Data.Dataset):
    def __init__(self, root, resize= [256,256], transform = None, gray = True):
        self.files = glob(os.path.join(root, '*.*'))
        self.resize = resize
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img = Image.open(self.files[index]).resize(self.resize)
        
        if self.gray:
            img = img.convert('L')
        
        img = self._tensor(img)
        if self.transform is not None:
            img = self.transform(img)
        
        return img