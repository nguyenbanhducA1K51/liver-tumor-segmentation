import torch 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
class EncodeBlock(nn.Module):
    def __init__(self,in_channels,out_channels,normalization="b",num_groups=8,kernel_size=3,padding=1,pool=True) -> None:
        super().__init__()
        self.pool_layer=nn.MaxPool3d(2)
        self.pool=pool
        if out_channels//2<in_channels:
            out_conv1_channels=in_channels
        else:
            out_conv1_channels=out_channels//2
        if normalization=="g":
            self.norm1=nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            self.norm2=nn.GroupNorm(num_groups=num_groups, num_channels=out_conv1_channels)
        elif  normalization=="b":
            self.norm1=nn.BatchNorm3d(num_features=in_channels)
            self.norm2=nn.BatchNorm3d(num_features=out_conv1_channels)
        else:
            print("invalid norm")
            
        self.conv1=nn.Sequential(
            self.norm1,
            nn.Conv3d(in_channels=in_channels,out_channels=out_conv1_channels,kernel_size=kernel_size,padding=padding),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            self.norm2,
            nn.Conv3d(in_channels=out_conv1_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),
            nn.ReLU()
        )
    def forward(self,x):
        if self.pool:
            x=self.pool_layer(x)
        x=self.conv1(x)
        x=self.conv2(x)
        return x
class DecodeBlock(nn.Module):
    def __init__(self,concat_channels,out_channels,mode="nearest",size=2) -> None:
        super().__init__()
        
        self.upsample=Upsample(mode=mode,size=size)
        self.conv=EncodeBlock(in_channels=concat_channels,out_channels=out_channels,pool=False)
    def forward(self,x,concat):
        # print (concat.shape,x.shape)
        x=self.upsample(x)
        x=torch.cat([x,concat],1)
        x=self.conv(x)
        return x
class Upsample(nn.Module):
    def __init__(self,mode="nearest",size=2) -> None:
        super().__init__()
        assert mode in ["nearest","linear","bilinear","trilinear"] , "invalid mode"
        self.mode=mode
        self.size=size
    def forward(self,x):
        x=F.interpolate(x,scale_factor=self.size,mode=self.mode)
        return x
class Unet3D(nn.Module):
    def __init__(self,in_channels=1,num_class=2,fmap=32,) -> None:
        super().__init__()
        self.enblock1=EncodeBlock(in_channels=in_channels,out_channels=fmap,pool=False)
        self.enblock2=EncodeBlock(in_channels=fmap,out_channels=fmap*2)
        self.enblock3=EncodeBlock(in_channels=fmap*2,out_channels=fmap*4)
        self.enblock4=EncodeBlock(in_channels=fmap*4,out_channels=fmap*8)
        self.deblock3=DecodeBlock(concat_channels=fmap*8+fmap*4,out_channels=fmap*4)
        self.deblock2=DecodeBlock(concat_channels=fmap*4+fmap*2,out_channels=fmap*2)
        self.deblock1=DecodeBlock(concat_channels=fmap*2+fmap,out_channels=fmap)
        self.outconv= EncodeBlock(in_channels=fmap,out_channels=num_class,pool=False)
        self.softmax=nn.Softmax(dim=1)
        # for name, layer in self.named_children():
        #     layer.__name__ = name
        #     layer.register_forward_hook(
        #         lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
        #     )
    def forward(self,x):
        path1=self.enblock1(x)
        path2=self.enblock2(path1)
        path3=self.enblock3(path2)
        path4=self.enblock4(path3)
        out3=self.deblock3(path4,path3)
        out2=self.deblock2(out3,path2)
        out1=self.deblock1(out2,path1)
        out=self.outconv(out1)

        return out

x=torch.rand(1,1,128,128,128).to("cuda")
net=Unet3D(1,3).to("cuda")
y=net(x)

