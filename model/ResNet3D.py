import torch
import torch.nn as nn
from collections import OrderedDict
# out of memory :)
class Res_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,stride=1):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        
        self.skip= nn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride)
    def forward(self,x):
        skip=self.skip(x)
        y=self.conv(x)
        return skip+y
class Down_sample(nn.Module):
    def __init__(self,in_channels,kernel_size=3,padding=1,stride=2):
        super().__init__()
        self.down=nn.Conv3d(in_channels,in_channels,kernel_size=kernel_size,padding=padding,stride=2)
    def forward(self,x):
        return self.down(x)
class Decoder_block(nn.Module):
    def __init__(self,in_channels,out_channels,interpolate=True):
        super().__init__()
        self.interpolate=interpolate
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.up_sample_interpolate = nn.Sequential(nn.Upsample(scale_factor=2,
                                                               mode='trilinear'),

                                                  nn.Conv3d(in_channels=self.in_channels,
                                                            out_channels=self.in_channels//2,
                                                            kernel_size=3,
                                                            padding=1),
                                                 nn.ReLU()

                                                 )

        # Upsample via transposed convolution (know to produce artifacts)
        self.up_sample_transposed = nn.Sequential(nn.ConvTranspose3d(in_channels=self.in_channels,
                                                       out_channels=self.in_channels//2,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       output_padding=1
                                                       ),
                                                       nn.ReLU()  )
        self.conv_block=nn.Sequential(nn.Conv3d(in_channels=self.in_channels//2,out_channels=self.out_channels,kernel_size=3,padding=1))
    def forward(self,x,res):
        if self.interpolate:
           x=self.up_sample_interpolate(x)
        else:
            x=self.up_sample_transposed(x)
        x=x+res 
        x=self.conv_block(x)
        return x

    
class ResUnet(nn.Module):
    def __init__(self,in_channels=1,n_labels=2):
        super().__init__()
        self.enblock1=Res_block(in_channels=in_channels,out_channels=32)
        self.down1=Down_sample(32)
        self.enblock2=Res_block(in_channels=32,out_channels=64)
        self.down2=Down_sample(64)
        self.enblock3=Res_block(in_channels=64,out_channels=128)
        self.down3=Down_sample(128)
        self.enblock4=Res_block(in_channels=128,out_channels=256)
        self.deblock1=Decoder_block(in_channels=256,out_channels=128)
        self.deblock2=Decoder_block(in_channels=128,out_channels=64)
        self.deblock3=Decoder_block(in_channels=64,out_channels=32)
        self.out=nn.Conv3d(in_channels=32,out_channels=n_labels,kernel_size=3,padding=1)
        self.classify=nn.Softmax(dim=1)
        # for name, layer in self.named_children():
        #     layer.__name__ = name
        #     layer.register_forward_hook(
        #         lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
        #     )
    def forward(self,x):
        res1=self.enblock1(x)
        x=self.down1(res1)
        res2=self.enblock2(x)
        x=self.down2(res2)
        res3=self.enblock3(x)
        x=self.down3(res3)
        bridge=self.enblock4(x)
        x=self.deblock1(bridge,res3)
        x=self.deblock2(x,res2)
        x=self.deblock3(x,res1)
        x=self.out(x)
        x=self.classify(x)
        return x


class Layer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,stride=1):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        if stride==1:
            self.skip=nn.Identity()
        else:
            self.skip=nn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride)
    def forward(self,x):
        skip=self.skip(x)
        y=self.conv(x)
        return skip+y      
        
class ResBlock(nn.Module):
    def __init__(self,nums_layers,in_channels,out_channels,stride=2):
        super().__init__()
        layer_dict=OrderedDict()
        layer_dict.update({ "layer_1":Layer(in_channels=in_channels,out_channels=out_channels,stride=stride)})
        for i in range(nums_layers):
            layer_dict.update({ f"layer_{i+2}":Layer(in_channels=out_channels,out_channels=out_channels,stride=1)})
        
        self.layers=nn.Sequential(layer_dict)

    def forward(self,x):
        x=self.layers(x)
        return x
class ResNet3D(nn.Module):
    def __init__(self,input_channels=3,in_channels=64,n_labels=2,cfg=[3,4,6,3]):
        super().__init__()
        # self.blocks=nn.Sequential()
        self.initcon=nn.Sequential(
            nn.Conv3d(in_channels=input_channels,out_channels=in_channels,kernel_size=7,padding=3,stride=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU()

        )
        block_dict = OrderedDict()
        channels=in_channels
        block_dict.update({f"block_1":ResBlock(cfg[0],in_channels=channels,out_channels=channels,stride=1)})
        for i,val in enumerate(cfg[1:],start=1):
            
            block_dict.update({f"block_{i+1}":ResBlock(cfg[i],in_channels=channels,out_channels=channels*2,stride=2)})
            channels*=2
        self.blocks=nn.Sequential(block_dict)
        # self.outcov=nn.Conv3d()
        for name, layer in self.blocks.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self,x):
        x=self.initcon(x)
        x=self.blocks(x)
        return x
if __name__=="__main__":
    x=torch.rand (1,1,48,256,256).to("cuda")
    # net1=ResNet3D().to("cuda")
    # # print (net)  
    # y=net1(x)
    net2=ResUnet().to("cuda")
    # print (net2)
    y=net2(x)




