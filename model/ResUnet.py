import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
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
# PATH_PRETRAINED_WEIGHTS = "/home/jovyan/work/pretrained/resnet_10_23dataset.pth"
# net = resnet50(
#     pretrained=True,
#     spatial_dims=3,
# )
# print (net)
# net.load_state_dict(torch.load(PATH_PRETRAINED_WEIGHTS))
# class ResUNet(nn.Module):
#     def __init__(self,n_labels):
#         self.n_labels=n_labels
# class Conv_block(nn.Module):
#     def __init__(self,in_channels,out_channels,n_block=2,kernel=3,stride=1):
#         super().__init__()
#         self.blocks=nn.Sequential( nn.Conv3d(in_channels,out_channels,kernel,stride),
#         nn.BatchNorm3d(out_channels),
#         nn.PReLU(out_channels),
        
#         )
#         for i in range (n_block-1):
#             self.blocks.extend([
#             nn.Conv3d(out_channels,out_channels,kernel,stride),
#             nn.BatchNorm3d(out_channels),
#             nn.PReLU(out_channels)
#             ])
#         def forward(self,x):
#             x=self.blocks(x)
#             return x 


            

# class UResNet(nn.Module):
#     def __init__(self,in_channels,n_labels):
#         super().__init__()
#         self.init_block=nn.Sequential(nn.Conv3d(in))


# class ResUNet(nn.Module):
#     def __init__(self, in_channel=1, out_channel=2 ,training=True):
#         super().__init__()

#         self.training = training
#         self.drop_rate = 0.2

#         self.encoder_stage1 = nn.Sequential(
#             nn.Conv3d(in_channel, 16, 3, 1, padding=1),
#             nn.PReLU(16),

#             nn.Conv3d(16, 16, 3, 1, padding=1),
#             nn.PReLU(16),
#         )

#         self.encoder_stage2 = nn.Sequential(
#             nn.Conv3d(32, 32, 3, 1, padding=1),
#             nn.PReLU(32),

#             nn.Conv3d(32, 32, 3, 1, padding=1),
#             nn.PReLU(32),

#             nn.Conv3d(32, 32, 3, 1, padding=1),
#             nn.PReLU(32),
#         )

#         self.encoder_stage3 = nn.Sequential(
#             nn.Conv3d(64, 64, 3, 1, padding=1),
#             nn.PReLU(64),

#             nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
#             nn.PReLU(64),

#             nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
#             nn.PReLU(64),
#         )

#         self.encoder_stage4 = nn.Sequential(
#             nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
#             nn.PReLU(128),

#             nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
#             nn.PReLU(128),

#             nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
#             nn.PReLU(128),
#         )

#         self.decoder_stage1 = nn.Sequential(
#             nn.Conv3d(128, 256, 3, 1, padding=1),
#             nn.PReLU(256),

#             nn.Conv3d(256, 256, 3, 1, padding=1),
#             nn.PReLU(256),

#             nn.Conv3d(256, 256, 3, 1, padding=1),
#             nn.PReLU(256),
#         )

#         self.decoder_stage2 = nn.Sequential(
#             nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
#             nn.PReLU(128),

#             nn.Conv3d(128, 128, 3, 1, padding=1),
#             nn.PReLU(128),

#             nn.Conv3d(128, 128, 3, 1, padding=1),
#             nn.PReLU(128),
#         )

#         self.decoder_stage3 = nn.Sequential(
#             nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
#             nn.PReLU(64),

#             nn.Conv3d(64, 64, 3, 1, padding=1),
#             nn.PReLU(64),

#             nn.Conv3d(64, 64, 3, 1, padding=1),
#             nn.PReLU(64),
#         )

#         self.decoder_stage4 = nn.Sequential(
#             nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
#             nn.PReLU(32),

#             nn.Conv3d(32, 32, 3, 1, padding=1),
#             nn.PReLU(32),
#         )

#         self.down_conv1 = nn.Sequential(
#             nn.Conv3d(16, 32, 2, 2),
#             nn.PReLU(32)
#         )

#         self.down_conv2 = nn.Sequential(
#             nn.Conv3d(32, 64, 2, 2),
#             nn.PReLU(64)
#         )

#         self.down_conv3 = nn.Sequential(
#             nn.Conv3d(64, 128, 2, 2),
#             nn.PReLU(128)
#         )

#         self.down_conv4 = nn.Sequential(
#             nn.Conv3d(128, 256, 3, 1, padding=1),
#             nn.PReLU(256)
#         )

#         self.up_conv2 = nn.Sequential(
#             nn.ConvTranspose3d(256, 128, 2, 2),
#             nn.PReLU(128)
#         )

#         self.up_conv3 = nn.Sequential(
#             nn.ConvTranspose3d(128, 64, 2, 2),
#             nn.PReLU(64)
#         )

#         self.up_conv4 = nn.Sequential(
#             nn.ConvTranspose3d(64, 32, 2, 2),
#             nn.PReLU(32)
#         )

#         self.map4 = nn.Sequential(
#             nn.Conv3d(32, out_channel, 1, 1),
#             # nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
#             nn.Softmax(dim=1)
#         )
#         self.map3 = nn.Sequential(
#             nn.Conv3d(64, out_channel, 1, 1),
#             nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear', align_corners=False),
#             nn.Softmax(dim=1)
#         )
#         self.map2 = nn.Sequential(
#             nn.Conv3d(128, out_channel, 1, 1),
#             nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear', align_corners=False),

#             nn.Softmax(dim=1)
#         )
#         self.map1 = nn.Sequential(
#             nn.Conv3d(256, out_channel, 1, 1),
#             nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear', align_corners=False),
#             nn.Softmax(dim=1)
#         )
#         # for name, layer in self.named_children():
#         #     layer.__name__ = name
#         #     layer.register_forward_hook(
#         #         lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
#         #     )


#     def forward(self, inputs):
#         # print(self.encoder_stage1(inputs).shape,inputs.shape)
#         long_range1 = self.encoder_stage1(inputs) + inputs

#         short_range1 = self.down_conv1(long_range1)

#         long_range2 = self.encoder_stage2(short_range1) + short_range1
#         long_range2 = F.dropout(long_range2, self.drop_rate, self.training)

#         short_range2 = self.down_conv2(long_range2)

#         long_range3 = self.encoder_stage3(short_range2) + short_range2
#         long_range3 = F.dropout(long_range3, self.drop_rate, self.training)

#         short_range3 = self.down_conv3(long_range3)

#         long_range4 = self.encoder_stage4(short_range3) + short_range3
#         long_range4 = F.dropout(long_range4, self.drop_rate, self.training)

#         short_range4 = self.down_conv4(long_range4)

#         outputs = self.decoder_stage1(long_range4) + short_range4
#         outputs = F.dropout(outputs, self.drop_rate, self.training)

#         output1 = self.map1(outputs)

#         short_range6 = self.up_conv2(outputs)

#         outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
#         outputs = F.dropout(outputs, self.drop_rate, self.training)

#         output2 = self.map2(outputs)

#         short_range7 = self.up_conv3(outputs)

#         outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
#         outputs = F.dropout(outputs, self.drop_rate, self.training)

#         output3 = self.map3(outputs)

#         short_range8 = self.up_conv4(outputs)

#         outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

#         output4 = self.map4(outputs)

#         # if self.training is True:
#         #     return output1, output2, output3, output4
#         # else:
#         #     return output4
#         return output4
if __name__=="__main__":
    seq=nn.Sequential(
            nn.Conv3d(2, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )
    net=ResUNet()
    x=torch.rand(2,1,48,256,256)
    # y=seq(x)
    y=net(x)

    print (y.shape)

