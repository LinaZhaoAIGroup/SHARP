import torch
import torch.nn as nn


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


class Net(nn.Module):
    def __init__(self, in_channel=3, adpsize=224):
        super(Net, self).__init__()
        self.in_channel = in_channel
        self.adpsize = adpsize
        self.dowm = nn.AdaptiveAvgPool2d(adpsize)
        self.recon = UNet(in_channel=in_channel, num_class=1)
        pass

    def forward(self, inputs):
        recon = self.recon(self.dowm(inputs))
        return recon


from model.common import default_conv, Upsampler
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


class ESRT(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, upscale=4, conv=default_conv, n_feats = 32, kernel_size = 3):
        super(ESRT, self).__init__()
        act = nn.ReLU(True)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale = upscale
        self.kernel_size = kernel_size
        self.n_feats = n_feats
        
        self.up1 = nn.Sequential(
            Upsampler(conv, scale=self.scale, in_channel=self.in_channel, act=False),
            BasicConv(in_planes=self.in_channel, out_planes=self.out_channel, kernel_size=self.kernel_size))
        self.up2 = nn.Sequential(
            conv(self.in_channel, self.n_feats, self.kernel_size),
            conv(self.n_feats, self.n_feats, self.kernel_size),
            Upsampler(conv, self.scale, self.n_feats, act=False),
            conv(self.n_feats, self.out_channel, self.kernel_size))
    
    def forward(self, x1, x2 = None, test=False):
        return self.up1(x1) + self.up2(x1)


if __name__ == '__main__':
    from summary import summary
    model = UNet(in_channel=1, num_class=1, adpsize=224).to('cpu')
    print(summary(model, input_size=(1, 2048, 2048), batch_size=1, device='cpu'))