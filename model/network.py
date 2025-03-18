
from .archs.RFDN_arch import RFDN
from .Net import *

class Net(nn.Module):
    def __init__(self, in_channel=1, scale=4,nums_projects=1200,image_size=1024):
        super(Net, self).__init__()
        self.in_channel = in_channel
        self.scale = scale
        self.channels=nums_projects
        self.ml=nn.Linear(in_features=image_size, out_features=256*256, bias=False)
        self.conv1=nn.Conv2d(nums_projects, 1, kernel_size=1, stride=1)        
        self.recon = FoldConv_asppUNet(in_channel=self.in_channel, num_class=1)
        self.up = RFDN(in_nc=1, nf=50, num_modules=4, out_nc=1, upscale=self.scale)
        self.output=nn.ReLU()
        self.image_size =image_size
    def forward(self, x):
        x=x.view(-1,self.channels,self.image_size)
        x=self.ml(x)
        x=x.view(-1,self.channels,256,256)
        x=self.conv1(x)
        x = self.recon(x)
        x = self.up(x)
        return self.output(x)