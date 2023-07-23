import argparser
from collections import OrderedDict
from data.train_dataset import trainDataset
from data.val_dataset import valDataset
from torch.utils.data import DataLoader
from utils.loss import HybridLoss
from model.Unet import UNet
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import metrics,common
def train (model,train_loader,epoch,optimizer,loss_func,labels):
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(labels)
    model.train()
    print (f"==========Epoch {epoch} Train==========")
    train_loss = metrics.LossAverage()
    with tqdm(train_loader, unit="batch") as tepoch:
        for idx, (data, target) in enumerate(tepoch):
            data, target = data.float(), target.long()
            target = common.one_hot_encoder(target, labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_func(output, target)  
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
            train_dice.update(output,target)
            if idx %3==0:
                tepoch.set_postfix(loss=loss.item(),avg_liver_dice=train_dice.avg[1]) 
    train_log = OrderedDict({'loss': train_loss.avg, 'liver_dice': train_dice.avg[1]})
    if labels==3:
        train_log.update({"tumor_dice":train_dice.avg[2] })
    return train_log

def val (model,val_loader,loss_func,labels ):
    model.eval()
    print ("==========Validation==========")
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(labels)
    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as tepoch:
            for idx, (data, target) in enumerate(tepoch):

                data, target = data.float(), target.long()
                target = common.one_hot_encoder(target, labels)
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss=loss_func(output, target)          
                val_loss.update(loss.item(),data.size(0))
                val_dice.update(output, target)
                if idx% 3==0:
                    tepoch.set_postfix(loss=loss.item(),dice=val_dice.avg[1])

    val_log = OrderedDict({'loss': val_loss.avg, 'liver_dice': val_dice.avg[1]})
    print ("val_log",val_log)
    if  labels==3: val_log.update({'tumor_dice': val_dice.avg[2]})
    return val_log

if __name__=='__main__':
    args=argparser.args
    device = torch.device('cuda' if args.cuda else 'cpu')
    train_dataset=trainDataset(os.path.join(args.fix_train_path,"train_path_list.txt"),mini_data=args.mini_data,args=args)
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
    val_dataset=valDataset(os.path.join(args.fix_train_path,"val_path_list.txt") ,mini_data=args.mini_data)
    
    val_loader = DataLoader(val_dataset, batch_size=1,shuffle=False,num_workers=args.num_workers)
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss=HybridLoss()
    # log = logger.Train_Logger(save_path,"train_log")

    for epoch in range(1, args.epochs + 1):
        # common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model=model,train_loader= train_loader,epoch=epoch ,optimizer= optimizer,loss_func= loss, labels=args.n_labels)
        val_log = val(model, val_loader, loss, args.n_labels)
        # log.update(epoch,train_log,val_log)

