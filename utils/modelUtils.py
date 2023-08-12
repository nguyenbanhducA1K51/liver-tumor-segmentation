import torch
import os
def save_model(model,dir='/root/repo/liver-tumor-segmentation/model/output'):
    print ("Saving model")
    torch.save({
                    'model_state_dict': model.state_dict(),

                  
                    },os.path.join(dir,"model.pth"))