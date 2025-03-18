import torch
import torch.nn as nn
import numpy as np
import math
from .block import pixelshuffle_block
import torch.nn.functional as F

class S_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(S_Conv2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
    def forward(self, inputs):
        outputs = self.conv1(inputs)

        return outputs

class Double_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Double_Conv2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs) 
        return outputs


class Up_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up_Block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.conv = Double_Conv2d(in_channel, out_channel)


    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        diffY = inputs1.size()[2] - inputs2.size()[2]
        diffX = inputs1.size()[3] - inputs2.size()[3]
        inputs2 = nn.functional.pad(inputs2, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
        outputs = torch.cat([inputs1,inputs2], dim=1)
        return self.conv(outputs)


class UNet(nn.Module):
    def __init__(self, in_channel=3, num_class=2, filters=[16, 32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.conv1 = Double_Conv2d(in_channel, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = Double_Conv2d(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = Double_Conv2d(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = Double_Conv2d(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = Double_Conv2d(filters[3], filters[4])
        self.Up_Block4 = Up_Block(filters[4], filters[3])
        self.Up_Block3 = Up_Block(filters[3], filters[2])
        self.Up_Block2 = Up_Block(filters[2], filters[1])
        self.Up_Block1 = Up_Block(filters[1], filters[0])
        self.final = nn.Conv2d(filters[0], num_class, kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(self.maxpool1(conv1))
        conv3 = self.conv3(self.maxpool2(conv2))
        conv4 = self.conv4(self.maxpool3(conv3))
        center = self.center(self.maxpool4(conv4))
        up4 = self.Up_Block4(conv4, center)
        up3 = self.Up_Block3(conv3, up4)
        up2 = self.Up_Block2(conv2, up3)
        up1 = self.Up_Block1(conv1, up2)
        return self.final(up1)


from .common import default_conv, Upsampler
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0,fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x

class FoldConv_aspp(nn.Module):
    def __init__(self, in_channel, out_channel, 
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 win_size=3, win_dilation=1, win_padding=1):
        super(FoldConv_aspp, self).__init__()
        #down_C = in_channel // 8
        self.down_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3,padding=1),nn.BatchNorm2d(out_channel),
             nn.PReLU())
        self.win_size = win_size
        self.unfold = nn.Unfold(win_size, win_dilation, win_padding, win_size)
        fold_C = out_channel * win_size * win_size
        down_dim = fold_C // 2
        self.conv1 = nn.Sequential(nn.Conv2d(fold_C, down_dim,kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(fold_C, down_dim, kernel_size, stride, padding, dilation, groups),
            nn.BatchNorm2d(down_dim),
            nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d( down_dim), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(fold_C, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU()  #如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(nn.Conv2d(5 * down_dim, fold_C, kernel_size=1), nn.BatchNorm2d(fold_C), nn.PReLU())

        # self.fold = nn.Fold(out_size, win_size, win_dilation, win_padding, win_size)
        self.up_conv = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, in_feature):
        N, C, H, W = in_feature.size()
        
        in_feature = self.down_conv(in_feature) #降维减少通道数
        
        in_feature = self.unfold(in_feature) #滑窗 [B, C* kH * kW, L] 
        in_feature = in_feature.view(in_feature.size(0), in_feature.size(1),
                                     H // self.win_size+1, W // self.win_size+1)
        in_feature1 = self.conv1(in_feature)
        in_feature2 = self.conv2(in_feature)
        in_feature3 = self.conv3(in_feature)
        in_feature4 = self.conv4(in_feature)
        in_feature5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(in_feature, 2)), size=in_feature.size()[2:], mode='bilinear')
        in_feature = self.fuse(torch.cat((in_feature1, in_feature2, in_feature3,in_feature4,in_feature5), 1))
        in_feature = in_feature.reshape(in_feature.size(0), in_feature.size(1), -1)
        in_feature = F.fold(input=in_feature, output_size=H, kernel_size=3, dilation=1, padding=1, stride=3)
        in_feature = self.up_conv(in_feature)
        return in_feature


class FoldConv_asppUNet(nn.Module):
    def __init__(self, in_channel=1, num_class=1, filters=[16, 32, 64, 128, 256]):
        super(FoldConv_asppUNet, self).__init__()
        self.conv1 = FoldConv_aspp(in_channel, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = FoldConv_aspp(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = FoldConv_aspp(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = FoldConv_aspp(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        self.center = FoldConv_aspp(filters[3], filters[4])
        
        self.Up_Block4 = Up_Block(filters[4], filters[3])
        self.Up_Block3 = Up_Block(filters[3], filters[2])
        self.Up_Block2 = Up_Block(filters[2], filters[1])
        self.Up_Block1 = Up_Block(filters[1], filters[0])
        self.final = nn.Conv2d(filters[0], num_class, kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(self.maxpool1(conv1))
        conv3 = self.conv3(self.maxpool2(conv2))
        conv4 = self.conv4(self.maxpool3(conv3))
        center = self.center(self.maxpool4(conv4))
        up4 = self.Up_Block4(conv4, center)
        up3 = self.Up_Block3(conv3, up4)
        up2 = self.Up_Block2(conv2, up3)
        up1 = self.Up_Block1(conv1, up2)
        return self.final(up1)
    

if __name__ == '__main__':
    from summary import summary
    model = UNet(in_channel=1, num_class=1, adpsize=224).to('cpu')
    print(summary(model, input_size=(1, 2048, 2048), batch_size=1, device='cpu'))