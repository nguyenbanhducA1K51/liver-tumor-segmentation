import matplotlib.pyplot as plt
import torch
import math
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
import sys
sys.path.append("/root/repo/liver-tumor-segmentation/")
import argparser
sys.path.append("/root/repo/liver-tumor-segmentation/data")
from transform import Compose,RandomCrop,Center_Crop,MinMaxNormalize,ToTensor,AddChannel, Z_scoreNormalize
import numpy as np
pd.set_option('expand_frame_repr', False)

class trainDataset(Dataset):
    def __init__(self,file_path,mini_data,args):
        self.args=args
        self.path_list= read_file (file_path)
        self.file_path=file_path
        # self. mean,self.std=calculateZ_score(file_path=self.file_path)
        self.mean=19.657849270340645
        self.std=41.15456650839886
        assert mini_data >=-1 , "invalid mini data"
        if mini_data !=-1:
            num_samples=min(len(self.path_list),mini_data)
            self.path_list=self.path_list[:num_samples]
        # print (self.path_list)
     
    def __len__(self):
        return len(self.path_list)
    def __getitem__(self,idx):      
        ct = sitk.ReadImage(self.path_list[idx][0] ,sitk.sitkInt16)
        seg = sitk.ReadImage(self.path_list[idx][1], sitk.sitkUInt8)
       
        ct = sitk.Clamp(ct, 0, 400)

        ct_array = sitk.GetArrayFromImage(ct)
        
        seg_array = sitk.GetArrayFromImage(seg)
        # print ("seg",seg_array.shape)
        ct_array = ct_array.astype(np.float32)
       

        transforms=Compose([ 
        ToTensor(),
        AddChannel(),
        Z_scoreNormalize(mean=self.mean,std=self.std),
        RandomCrop (slices=self.args.crop_size)
    ])
        ct_tensor,seg_tensor = transforms(ct_array, seg_array)     
        return ct_tensor, seg_tensor.squeeze(0), os.path.basename(self.path_list[idx][0])
def read_file(path):
    file_name_list = []
    with open(path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list

def visualize(img,seg,slice=24):

    image_arr=np.array(img)
    seg_arr=np.array(seg)
    print("img",image_arr.shape,"seg",seg_arr.shape)

    slice_img=image_arr[:,slice].squeeze()
    slice_seg=seg_arr[slice].squeeze()
    print("img",slice_img.shape,"seg",slice_seg.shape)
    #why use cmap=bone
    path="/root/repo/liver-tumor-segmentation/model/output/train_img"
    fig,ax=plt.subplots(1,2)
    # ax[0].imshow( slice_img,cmap="bone",vmin=0, vmax=1)
    # ax[1].imshow( slice_seg,cmap="bone",vmin=0, vmax=1)
    ax[0].imshow( slice_img)
    ax[1].imshow( slice_seg)
    plt.axis('off') 
    plt.show()
    plt.savefig(path)
    plt.clf()

def readraw(path="/root/data/liver/fix_train/ct/volume-3.nii"):
    ct = sitk.ReadImage(path ,sitk.sitkInt16)
    ct = sitk.Clamp(ct, 0, 400)
    ct_array = sitk.GetArrayFromImage(ct)
    print (np.max(ct_array),np.min(ct_array))

def calculateZ_score(file_path="/root/data/liver/fix_train/train_path_list.txt"):
    image_list=read_file(file_path)
    mean_ls=[]
    for pair in image_list:
        img=pair[0]
        ct = sitk.ReadImage(img ,sitk.sitkInt16)
        ct = sitk.Clamp(ct, 0, 400)
        ct_array = sitk.GetArrayFromImage(ct)
        mean_ls.append(np.mean(ct_array))
    mean=np.mean(np.array(mean_ls))
    print ("mean",mean)
    std_ls=[]
    for pair in image_list:
        img=pair[0]
        ct = sitk.ReadImage(img ,sitk.sitkInt16)
        ct = sitk.Clamp(ct, 0, 400)
        ct_array = sitk.GetArrayFromImage(ct)
        variance=np.sum( (ct_array-mean)*(ct_array-mean) )
        std_ls.append(variance)

    sample=image_list[0][0]
    ct = sitk.ReadImage(sample,sitk.sitkInt16)
    # ct_array = ct.GetSpacing()
    total_voxel=len(image_list)*ct_array.shape[0]*ct_array.shape[1]*ct_array.shape[2]
    std=math.sqrt(np.sum(np.array(std_ls))/total_voxel)
    print("std",std)
    return mean, std




if __name__=="__main__":    
    args=argparser.args
    dataset=trainDataset(file_path="/root/data/liver/fix_train/train_path_list.txt",mini_data=10,args=args)
   
    visualize(dataset.__getitem__(8)[0],dataset.__getitem__(8)[1])
    readraw()
    # calculateZ_score()
        # print (dataset.__getitem__(i)[0].max(),dataset.__getitem__(i)[1].max(),dataset.__getitem__(i)[1].sum())


   
    
        
   









