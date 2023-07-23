import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
from ipywidgets import *
from PIL import Image
from matplotlib.pyplot import figure
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from glob import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import nibabel as nib
import random
from scipy import ndimage
import SimpleITK as sitk
import sys
sys.path.append("/root/repo/liver-tumor-segmentation/")
import argparser
sys.path.append("/root/repo/liver-tumor-segmentation/data")
from transform import Compose,RandomCrop,Center_Crop,MinMaxNormalize,ToTensor,AddChannel

class valDataset(Dataset):
    def __init__(self,file_path,mini_data):

        self.path_list=read_file(file_path)
        assert mini_data >=-1 , "invalid mini data"
        if mini_data !=-1:
            num_samples=min(len(self.path_list),mini_data)
            self.path_list=self.path_list[:num_samples]
    def __len__(self):
        return len(self.path_list)
    def __getitem__(self,idx):
        ct_path=self.path_list[idx][0]
        seg_path=self.path_list[idx][1]

        ct=sitk.ReadImage(ct_path,sitk.sitkInt16)
        seg=sitk.ReadImage(seg_path,sitk.sitkUInt8)
        ct_array = sitk.GetArrayFromImage(ct)

        seg_array = sitk.GetArrayFromImage(seg)
        ct_array = ct_array.astype(np.float32)
        transforms=Compose([
            ToTensor(),
            AddChannel(),
            MinMaxNormalize(),
            Center_Crop()

        ])
        ct_tensor,seg_tensor=transforms(ct_array,seg_array)
        return ct_tensor,seg_tensor
def read_file(path):
    file_name_list = []
    with open(path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list

if __name__=="__main__":
    
    args=argparser.args
    dataset=valDataset(file_path="/root/data/liver/fix_train/val_path_list.txt",mini_data=10)
    for i in range (2):
        print (dataset.__getitem__(i)[0].shape,dataset.__getitem__(i)[1].shape)


   
    
        
