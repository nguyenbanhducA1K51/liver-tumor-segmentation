import torch
import torch.nn as nn
import torch.nn.functional as F

class ResUNet(nn.Module):
    def __init__(self,n_labels):
        self.n_labels=n_labels