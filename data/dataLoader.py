import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
from ipywidgets import *
from PIL import Image
from matplotlib.pyplot import figure
from torch.utils.data import Dataset
import pandas as pd
pd.set_option('expand_frame_repr', False)
file_list = []
class liverDataset(Dataset):
    def __init__(self,data_path='/root/data/liver_seg'):
        for dirname, _, filenames in os.walk(data_path):
            for filename in filenames:
                file_list.append((dirname, filename)) 
        
        df = pd.DataFrame(file_list, columns =['dirname', 'filename']) 
        df.sort_values(by=['filename'], ascending=True)  
        df["mask_dirname"]  = ""
        df["mask_filename"] = ""
        # print (df_files.head())
        # print (df.loc[:,"filename"].sample(n=10))
        # print (df.info())
        # print (df.head(100))
        df.to_csv('data.csv', index=True)
        count=0
        for i in range (len(df)):
            # if df[i:i+1]["filename"].startswith("volumn"):
            #     idx=os.path.basename(df[i:i+1]["filename"]).split("-")[1]
            #     df[i:i+1]["mask_filename"]=f"segmentation-{idx}.nii"
            #     df[i:i+1]["mask_dirname"]=os.path.join(data_path,"segmentation")
            if "volume" in df.loc[i,"filename"]:
                count+=1
                # idx=os.path.splitext(df.loc[i,"filename"])[0] .split("-")[1]
                # print (idx)
        #         df.loc[i,"mask_filename"]=f"segmentation-{idx}.nii"
        #         df.loc[i,"mask_dirname"]=os.path.join(data_path,"segmentation")
        # df= df[df.mask_filename != ''].sort_values(by=['filename']).reset_index(drop=True) 
        print (df.info())
        # print (df.sample(n=10))
        print ("count",count)
    
    def __len__(self):
        return 4
    def __getitem__(self,idx):
        return 1

dataset=liverDataset()

    # for i in range(int(len(df_files)/2) ):
    #     ct = f"volume-{i}.nii"
    #     mask = f"segmentation-{i}.nii"
    #     #
    #     df_files.loc[df_files['filename'] == ct, 'mask_filename'] = mask
    #     df_files.loc[df_files['filename'] == ct, 'mask_dirname'] = "/root/data/liver_seg/segmentations"
