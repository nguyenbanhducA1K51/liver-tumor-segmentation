import argparser
from collections import OrderedDict
from data.trainValLoader import train_val_loader
from models import UNet
    def train (self,model,train_loader):
        model.train()
        train_loss = metrics.LossAverage()
        for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_func(output, target)  
            loss.backward()
            optimizer.step()
            train_loss.update(loss3.item()
        train_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
        return train_log
    
    def val (self):
        model.eval()
        val_loss = metrics.LossAverage()
        val_dice = metrics.DiceAverage(n_labels)
        with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_func(output, target)          
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels==3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log

if __name__=='__main__':
    args=argparser.args
    device = torch.device('cuda' if args.cuda else 'cpu')
    train_loader,val_loader = train_val_loader(args=args)
    model = Unet()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss=loss.MixedLoss()
    log = logger.Train_Logger(save_path,"train_log")
    for epoch in range(1, args.epochs + 1):
        # common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch,train_log,val_log)

