import argparser
from collections import OrderedDict
from data.train_dataset import trainDataset
from data.val_dataset import valDataset
from torch.utils.data import DataLoader
from utils.loss import HybridLoss
from model.Unet import UNet,Unet3d
from model.Unet3D import Unet3D 
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import metrics,common,weight
import time
from utils.modelUtils import save_model

def train (model,train_loader,epoch,optimizer,loss_func,labels,device):
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(labels)
    model.train()
    print (f"==========Epoch {epoch} Train==========")
    train_loss = metrics.LossAverage()
    with tqdm(train_loader, unit="batch") as tepoch:
        for idx, (data, target,_) in enumerate(tepoch):
            data, target = data.float(), target.long()
            # print(data.shape) #torch.Size([1, 1, 48, 256, 256])
            target = common.one_hot_encoder(target, labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_func(output, target)  
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
            train_dice.update(output,target)
            if idx %3==0:
                tepoch.set_postfix(loss=loss.item(),avg_liver_dice=train_dice.avg[1],round_liver_dice=train_dice.round_dice_avg[1]) 
    train_log = OrderedDict({'loss': train_loss.avg, 'liver_dice': train_dice.avg[1],'round_liver_dice':train_dice.round_dice_avg[1]})
    if labels==3:
        train_log.update({"tumor_dice":train_dice.avg[2] })
    return train_log

def val (model,val_loader,loss_func,labels,device ):
    model.eval()
    print ("==========Validation==========")
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(labels)
    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as tepoch:
            for idx, (data, target,name) in enumerate(tepoch):
                # print (target.shape)
                if target.shape[1]!=96:
                    print (f"{name} only have {target.shape[1]} slice")
                    continue
                data, target = data.float(), target.long()
                target = common.one_hot_encoder(target, labels)
                data, target = data.to(device), target.to(device)
                output1 = model(data[:,:,:48])
                output2 = model(data[:,:,48:])
                loss1=loss_func(output1, target[:,:,:48])  
                loss2=loss_func(output2, target[:,:,48:]) 
                loss=   (loss1+loss2)/2
                val_loss.update(loss.item(),data.size(0))
                val_dice.update(torch.cat([output1,output2],2), target)
                if idx% 3==0:
                    tepoch.set_postfix(loss=loss.item(),dice=val_dice.avg[1],round_dice=val_dice.round_dice_avg[1])

    val_log = OrderedDict({'loss': val_loss.avg, 'liver_dice': val_dice.avg[1],'round_dice':val_dice.round_dice_avg[1]})
    print ("val_log",val_log)
    if  labels==3: val_log.update({'tumor_dice': val_dice.avg[2]})
    return val_log
# def loadModel(args):
def load_model(args):
    if args.model=="unet":
        return Unet3D()
    elif args.model=="resunet":
        return Unet3D()

if __name__=='__main__':
    args=argparser.args
    device=torch.device('cuda')
    train_dataset=trainDataset(os.path.join(args.fix_train_path,"train_path_list.txt"),mini_data=args.mini_data,args=args)
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
    val_dataset=valDataset(os.path.join(args.fix_train_path,"val_path_list.txt") ,mini_data=args.mini_data,args=args)
    
    val_loader = DataLoader(val_dataset, batch_size=1,shuffle=False,num_workers=args.num_workers)
    model = load_model(args)
    # model=ResUNet()
    model.apply(weight.init_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss=HybridLoss()
    # log = logger.Train_Logger(save_path,"train_log")

    for epoch in range(1, args.epochs + 1):
        # common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model=model,train_loader= train_loader,epoch=epoch ,optimizer= optimizer,loss_func= loss, labels=args.n_labels,device=device)
        val_log = val(model, val_loader, loss, args.n_labels,device=device)
    save_model(model)

        # log.update(epoch,train_log,val_log)

