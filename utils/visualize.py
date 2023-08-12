import matplotlib.pyplot as plt 
import sys 
sys.path.append("/root/repo/liver-tumor-segmentation/")
import argparser
import os
import random
from data.val_dataset import valDataset
from model.Unet import UNet,Unet3d
import torch 
import numpy as np
def visualize():
    args=argparser.args 
    dataset=valDataset(file_path=os.path.join(args.fix_train_path,"val_path_list.txt"),mini_data=-1,args=args)
    idx=random.randint(0,len(dataset)-1)
    img,seg,_=dataset.__getitem__(idx)
    
    checkpoint=torch.load("/root/repo/liver-tumor-segmentation/model/output/model.pth")
    model_state_dict = checkpoint['model_state_dict']
    model=UNet().to("cuda")
    model.load_state_dict(model_state_dict)
    img=img.unsqueeze(0).to("cuda").float()
    pred=model(img)
    max_indices = torch.argmax(pred, dim=1).to("cuda")
    pred = torch.zeros(pred.shape).to("cuda").scatter_(1,max_indices.unsqueeze(1),1)

    img=np.array(img.squeeze().detach().cpu())
    seg=np.array(seg)
    pred=np.array(pred[:,1].squeeze().detach().cpu())
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20, 10))
    
    ct1,ct2,ct3=img[img.shape[0]//2,:,:],img[:,img.shape[1]//2,:],img[:,:,img.shape[2]//2]
    seg1,seg2,seg3=seg[seg.shape[0]//2,:,:],seg[:,seg.shape[1]//2,:],seg[:,:,seg.shape[2]//2]
    pred1,pred2,pred3=pred[pred.shape[0]//2,:,:],pred[:,pred.shape[1]//2,:],pred[:,:,pred.shape[2]//2]
    ax[0,0].imshow(ct1,cmap="gray")
    ax[0,1].imshow(ct2,cmap="gray")
    ax[0,2].imshow(ct3,cmap="gray")
    ax[1,0].imshow(seg1,cmap="gray")
    ax[1,1].imshow(seg2,cmap="gray")
    ax[1,2].imshow(seg3,cmap="gray")
    ax[2,0].imshow(pred1,cmap="gray")
    ax[2,1].imshow(pred2,cmap="gray")
    ax[2,2].imshow(pred3,cmap="gray")
    plt.tight_layout()
    plt.show()
    plt.savefig('/root/repo/liver-tumor-segmentation/model/output/visualize.png')




    

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
    visualize()

