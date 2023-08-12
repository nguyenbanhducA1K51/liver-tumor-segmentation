import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/ishaanb92/PyTorch-UNet/blob/master/src/unet/blocks.py
import sys
sys.path.append ("/root/repo/liver-tumor-segmentation/utils/")
import weight
class Encoder_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1):
        super(Encoder_block, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
             )
        self.conv2=nn.Sequential(
            nn.Conv3d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        return x
class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1):
        super(Bottleneck,self).__init__()
        self.block=nn.Sequential(
            nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
             )

class Decoder_block(nn.Module):
    def __init__(self,in_channels,concat_channels,out_channels,interpolate=False,kernel_size=3,padding=1):
        super(Decoder_block,self).__init__()
        self.interpolate=interpolate
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.concat_channels=concat_channels
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
        self.conv_block=nn.Sequential(nn.Conv3d(in_channels=self.in_channels//2+self.concat_channels,out_channels=self.out_channels,kernel_size=3,padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU()
        
        )
    def forward(self,x,skip_layer):
            if self.interpolate:
                x=self.up_sample_interpolate(x)
            else:
                x=self.up_sample_transposed(x)
            return  self.conv_block(torch.cat( [x,skip_layer],1))
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, training=True):
        super(UNet, self).__init__()

        self.training = training
        self.pool=nn.MaxPool3d(2) 
        self.enblock1=Encoder_block(in_channels=in_channels,out_channels=32)
        self.enblock2=Encoder_block(in_channels=32,out_channels=64)
        self.enblock3=Encoder_block(in_channels=64,out_channels=128)
        self.enblock4=Encoder_block(in_channels=128,out_channels=256)
        # self.bn=Encoder_block(in_channels=256,out_channels=256)
        self.deblock1=Decoder_block(in_channels=256,concat_channels=128,out_channels=128)
        self.deblock2=Decoder_block(in_channels=128,concat_channels=64,out_channels=64)
        self.deblock3=Decoder_block(in_channels=64,concat_channels=32,out_channels=32)
        self.out=nn.Conv3d(in_channels=32,out_channels=out_channels,kernel_size=3,padding=1)
        self.classify=nn.Softmax(dim=1)
        

      

    def forward(self, x):
        respath1=self.enblock1(x)
        x=self.pool(respath1)
        respath2=self.enblock2(x)
        x=self.pool(respath2)
        respath3=self.enblock3(x)
        x=self.pool(respath3)
        bn=self.enblock4(x)
        x=self.deblock1(bn,respath3)
        x=self.deblock2(x,respath2)
        x=self.deblock3(x,respath1)
        x=self.out(x)
        y=self.classify(x)
        return y


       
class EncoderBlock(nn.Module):
    def __init__(self, out_channels=64, in_channels=1, dropout=False, dropout_rate=0.3):

        super(EncoderBlock,self).__init__()
        self.out_channels = int(out_channels)
        self.in_channels = int(in_channels)
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv3d(in_channels=self.in_channels,
                               out_channels=self.out_channels,
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv3d(in_channels=self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=3,
                               padding=1)

        # self.bn_op_1 = nn.InstanceNorm2d(num_features=self.out_channels, affine=True)
        # self.bn_op_2 = nn.InstanceNorm2d(num_features=self.out_channels, affine=True)
        self.bn_op_1=nn.BatchNorm3d(self.out_channels)
        self.bn_op_2=nn.BatchNorm3d(self.out_channels)

        if dropout is True:
            self.dropout_1 = nn.Dropout(p=dropout_rate)
            self.dropout_2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn_op_1(x)
        x = F.leaky_relu(x)
        if self.dropout is True:       
                x = self.dropout_1(x)
        x = self.conv2(x)
        x = self.bn_op_2(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
                x = self.dropout_2(x)
        return x
class DecoderBlock(nn.Module):

    def __init__(self, in_channels, concat_channels, out_channels, interpolate=False, dropout=False):

        super(DecoderBlock, self).__init__()
        self.out_channels =out_channels
        self.in_channels = in_channels
        self.concat_channels = concat_channels
        self.interpolate = interpolate
        self.dropout = dropout

        # Upsample by interpolation followed by a 3x3x3 convolution to obtain desired depth
        self.up_sample_interpolate = nn.Sequential(nn.Upsample(scale_factor=2,
                                                               mode='nearest'),

                                                  nn.Conv3d(in_channels=self.in_channels,
                                                            out_channels=self.in_channels//2,
                                                            kernel_size=3,
                                                            padding=1)
                                                 )

        # Upsample via transposed convolution (know to produce artifacts)
        self.up_sample_transposed = nn.ConvTranspose3d(in_channels=self.in_channels,
                                                       out_channels=self.in_channels//2,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       output_padding=1)
        self.conv_block=nn.Sequential(nn.Conv3d(in_channels=self.in_channels//2+self.concat_channels,out_channels=self.out_channels,kernel_size=3,padding=1),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU()
        
        )


    def forward(self, x, skip_layer):

        if self.interpolate is True:
            up_sample_out = F.leaky_relu(self.up_sample_interpolate(x))
        else:
            up_sample_out = F.leaky_relu(self.up_sample_transposed(x))

        merged_out = torch.cat([up_sample_out, skip_layer], dim=1)
        out=self.conv_block(merged_out)
        return out
class Bridge(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Bridge, self).__init__()
        self.seq=nn.Sequential(
            nn.Conv3d(in_channels,out_channels,1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels,out_channels,1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.seq(x)
class Unet3d(nn.Module):
    def __init__(self,in_channels=1,out_channels=2):
        super(Unet3d, self).__init__()
        self.max_pool3d=nn.MaxPool3d(2,2)
        self.encoder_block1=EncoderBlock(out_channels=32,in_channels=in_channels)
        self.encoder_block2=EncoderBlock(out_channels=64,in_channels=32)
        self.encoder_block3=EncoderBlock(out_channels=128,in_channels=64)
        self.encoder_block4=EncoderBlock(out_channels=256,in_channels=128)
        self.bridge=Bridge(in_channels=256, out_channels=512)
        self.decoder_block1=DecoderBlock(in_channels=512,concat_channels=256,out_channels=256)
        self.decoder_block2=DecoderBlock(in_channels=256,concat_channels=128,out_channels=128)
        self.decoder_block3=DecoderBlock(in_channels=128,concat_channels=64,out_channels=64)
        self.decoder_block4=DecoderBlock(in_channels=64,concat_channels=32,out_channels=32)
        self.out_conv=nn.Conv3d(32,out_channels,kernel_size= 3,padding=1)
        self.softmax=nn.Softmax(dim=1)
        # for name, layer in self.named_children():
        #     layer.__name__ = name
        #     layer.register_forward_hook(
        #         lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
        #     )

    def forward(self,x):
        res1=self.encoder_block1(x)
        x=self.max_pool3d(res1)
        res2=self.encoder_block2(x)
        x=self.max_pool3d(res2)
        res3=self.encoder_block3(x)
        x=self.max_pool3d(res3)
        res4=self.encoder_block4(x)
        x=self.max_pool3d(res4)
        bn=self.bridge(x)
        x=self.decoder_block1(bn,res4)
        x=self.decoder_block2(x,res3)
        x=self.decoder_block3(x,res2)
        x=self.decoder_block4(x,res1)
        x=self.out_conv(x)
        x=self.softmax(x)
        return x
        


        



if __name__=="__main__":
    net=UNet().to("cuda")
    net.apply(weight.init_model)
    x=torch.rand(2,1,48,512,512).to("cuda")
    y=net(x)
    print (y.shape)