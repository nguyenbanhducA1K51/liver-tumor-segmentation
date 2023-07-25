import torch
def one_hot_encoder(tensor,n_labels):
    #target has shape N, s,h,w. conver it to N,n_labels,s,h,w
    # target=target.squeeze(1)
    # assert len (target.shape)==4, f"target dim should be 4, but {target.shape}"
    # tmp_target=torch.zeros(target.shape[0],n_labels,target.shape[1],target.shape[2],target.shape[3])
    # target=target.unsqueeze(1)
    # tmp_target.scatter(1,target,1) #dim,index,src 
    # return tmp_target
    # def to_one_hot_3d(tensor, n_classes=3):  # shape = [batch, s, h, w]
    # n, s, h, w = tensor.size()
    n=tensor.shape[0]
    s=tensor.shape[1]
    h=tensor.shape[2]
    w=tensor.shape[3]
    one_hot = torch.zeros(n, n_labels, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot