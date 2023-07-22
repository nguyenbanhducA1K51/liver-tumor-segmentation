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
pd.set_option('expand_frame_repr', False)

class trainDataset(Dataset):
    def __init__(self,file_path,root_path):
    self.root_path=root_path
       self.names= read_file (file_path)
        
     
    def __len__(self):
        return len(self.names)
    def __getitem__(self,idx):      
        ct = sitk.ReadImage(self.names[idx][0] ,sitk.sitkInt16)
        seg = sitk.ReadImage(self.names[idx][1], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)

        seg_array = sitk.GetArrayFromImage(seg)
        ct_array = ct_array.astype(np.float32)

        transform=Compose([ 
        ToTensor(),
        AddChannel(),
        MinMaxNormalize(),
        RandomCrop (slices=args.crop_size)
    ])
            ct_tensor,seg_tensor = transforms(ct_array, seg_array)     
        return ct_tensor, seg_tensor
def read_file(path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list


# def train_val_loader(args):
#     train_val_rate=args.split
#     error_msg = "[!] val_rate should be in the range [0, 1]."
#     assert ((val_rate >= 0) and (val_rate <= 1)), error_msg
#     train_transform= Compose([ 
#         RandomCrop (slices=args.crop_size)
#     ])
#     val_transform=Compose([
#         Center_Crop(max_size=args.val_crop_max_size)
#     ])
#     train_dataset=liverDataset(transform=train_transform,data_path=args.ori_train_path)
#     val_dataset=liverDataset(transform=val_transform,data_path=args.ori_train_path)
#     num_train = len(dataset)
#     indices = list(range(num_train))
#     split = int(np.floor(valid_size * num_train))
#     train_idx, valid_idx = indices[split:], indices[:split]
#     train_sampler = SubsetRandomSampler(train_idx)
#     valid_sampler = SubsetRandomSampler(valid_idx)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, sampler=train_sampler,
#         num_workers=args.num_workers,
#     )
#     valid_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=1, sampler=valid_sampler,
#         num_workers=args.num_workers,
#     )
#     return train_loader,valid_loader
#     if __name__=="__main__":
#         import argparser
#         args=argparser.args
#         train_val_loader(args)

   






# print (os.path.exists("/root/data/liver/volume_pt6/volume-74.nii") )
# print (os.path.exists("/root/data/liver/segmentations/segmentation-74.nii"))
# data={
#     "image":"/root/repo/liver-tumor-segmentation/data/volume-0.nii.gz",
#     "label":"/root/repo/liver-tumor-segmentation/data/segmentation-0.nii.gz"
# }
# # seem like monai do not acceept .nii, but only nii gz
# loader = LoadImaged(keys=("image", "label"), image_only=False)
# data_dict = loader(data)
# print(f"image1 shape: {data_dict['image'].shape}")
# spacing = Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 5.0), mode=("bilinear", "nearest"))
# data_dict = spacing(data_dict)
# print(f"image2 shape: {data_dict['image'].shape}")
# print(f"label shape: {data_dict['label'].shape}")


# input_nii_file = '/root/repo/liver-tumor-segmentation/data/segmentation-0.nii'
# output_nii_gz_file = '/root/repo/liver-tumor-segmentation/data/segmentation-0.nii.gz'

# # Load the NIfTI file using nibabel
# nifti_image = nib.load(input_nii_file)

# # Save the NIfTI file in NIfTI compressed format (.nii.gz)
# nib.save(nifti_image, output_nii_gz_file)




# import shutil
# f_train="/root/data/liver/train"
# f_test="/root/data/liver/test"
# test_file=[f"volume-{i}.nii'" for i in range(27,46)]
# print ("test_file",test_file)
# data_path='/root/data/liver'
# file_list = []
# for dirname, _, filenames in os.walk(data_path):
#     for filename in filenames:
#         file_list.append((dirname, filename)) 

# df = pd.DataFrame(file_list, columns =['dirname', 'filename']) 
# df.sort_values(by=['filename'], ascending=True)  
# df["mask_dirname"]  = ""
# df["mask_filename"] = ""
# # print(df.head(15))
# print (df.info())



# for i in range (len(df)):

#         idx=int (os.path.splitext(df.loc[i,"filename"])[0] .split("-")[1])
#         if idx in range (27,48) :
#             print ( "in test file",df.loc[i,"filename"])
#             path=os.path.join(df.loc[i,"dirname"],df.loc[i,"filename"])

            # shutil.move(path,f_test)
# print (os.listdir(f_test))

# print (os.listdir("/root/data/liver/"))
# for i in range (len(df)):

#         idx=int (os.path.splitext(df.loc[i,"filename"])[0] .split("-")[1])
#         if idx not in range (27,48):
#             print ( "in test file",df.loc[i,"filename"])
#             path=os.path.join(df.loc[i,"dirname"],df.loc[i,"filename"])
#             shutil.move(path,f_train)
# print (os.listdir(f_train))


