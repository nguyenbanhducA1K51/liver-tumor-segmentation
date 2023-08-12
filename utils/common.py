import torch
def one_hot_encoder(tensor,n_labels): 
    n=tensor.shape[0]
    s=tensor.shape[1]
    h=tensor.shape[2]
    w=tensor.shape[3]
    one_hot = torch.zeros(n, n_labels, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot