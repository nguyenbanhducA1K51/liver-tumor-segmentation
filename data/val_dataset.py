import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
from ipywidgets import *
from PIL import Image
from matplotlib.pyplot import figure
from torch.utils.data import Dataset
import pandas as pd
import os
from glob import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import nibabel as nib
import random
from scipy import ndimage
import SimpleITK as sitk
from transform import Compose,RandomCrop,Center_Crop,MinMaxNormalize,ToTensor,AddChannel
class valDataset(Dataset):
    def __init__(self,file_path,root_ct_path,root_seg_path):
        self.root_ct_path=root_ct_path
        self.root_seg_path=root_seg_path
        self.names=self.read_file(path)
    def __len__(self):
        return self.names
    def __getitem__(self,idx):
        ct=self.names[idx][0]
        seg=self.names[idx][1]
        ct_path= os.path.join(self.root_ct_path,ct)
        seg_path=os.path.join(self.root_seg_path,seg)
        ct=sitk.ReadImage(ct_path,sitk.sitkInt16)
        seg=sitk.ReadImage(seg_path,sitk.sitkUInt8)
        ct_array = sitk.GetArrayFromImage(ct)

        seg_array = sitk.GetArrayFromImage(seg)
        transform=Compose([
            ToTensor(),
            AddChannel(),
            MinMaxNormalize(),
            Center_Crop()

        ])
        ct_tensor,seg_tensor=transforms(ct_array,seg_array)
        return ct_tensor,seg_tensor
def read_file(path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list
    
        
