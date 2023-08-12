import numpy as np
import torch
class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0
        self.round_dice_list=np.asarray([0]*self.class_num, dtype='float64')

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        self.round_dice_list+=DiceAverage.compute_dices(logits,targets)
        self.round_dice_avg= np.around(self.round_dice_list/ self.count, 4)

        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        print (f"class bg logits {torch.sum(logits[:, 0, :, :, :])} , target{torch.sum(targets[:, 0, :, :, :])} prod { torch.sum(logits[:, 0, :, :, :] * targets[:, 0, :, :, :])}" )
        print (f"class liver logits {torch.sum(logits[:, 1, :, :, :])} , target{torch.sum(targets[:, 1, :, :, :])} prod { torch.sum(logits[:, 1, :, :, :] * targets[:, 1, :, :, :])}" )
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)
    @staticmethod
    def compute_dices(logits,targets):
        dices=[]
        max_indices = torch.argmax(logits, dim=1).to("cuda")
        one_hot_tensor = torch.zeros(logits.shape).to("cuda").scatter_(1,max_indices.unsqueeze(1),1)
        for class_index in range(targets.size()[1]):
            inter = torch.sum(one_hot_tensor[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(one_hot_tensor[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)
