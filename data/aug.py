import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import sys
sys.path.append("/root/repo/liver-tumor-segmentation/utils")
sys.path.append("/root/repo/liver-tumor-segmentation/data")
from val_dataset import valDataset
sys.path.append("/root/repo/liver-tumor-segmentation/")
import argparser
import random 
import os
matplotlib.rcParams['image.cmap'] = 'bone'

if __name__=="__main__":
    args=argparser.args 
    dataset=valDataset(file_path=os.path.join(args.fix_train_path,"val_path_list.txt"),mini_data=-1,args=args)
    idx=random.randint(0,len(dataset)-1)
    img,seg,_=dataset.__getitem__(idx)
    img=np.array(img.squeeze().detach().cpu())
    seg=np.array(seg)
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))
    ct1,ct2,ct3=img[img.shape[0]//2,:,:],img[:,img.shape[1]//2,:],img[:,:,img.shape[2]//2]
    seg1,seg2,seg3=seg[seg.shape[0]//2,:,:],seg[:,seg.shape[1]//2,:],seg[:,:,seg.shape[2]//2]
    ax[0,0].imshow(ct1)
    ax[0,1].imshow(ct2)
    ax[0,2].imshow(ct3)
    ax[1,0].imshow(seg1)
    ax[1,1].imshow(seg2)
    ax[1,2].imshow(seg3)
    plt.tight_layout()
    plt.show()
    plt.savefig('/root/repo/liver-tumor-segmentation/model/output/aug.png')
