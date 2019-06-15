# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 15:05:50 2019

@author: win10
"""
from glob import glob
import string
from utils import test_gray, test_rgb

test_path = './images/IV_images/'

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
        
test(test_path, model, mode='add')