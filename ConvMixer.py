

import torch
import torch.nn as nn
import torch.nn.functional as F

#torch.__version__ should be 1.9
class ConvMixer(nn.Module):
    def __init__(self,no_of_op_channels , depth , kernal , patch_size ,output ):
        super().__init__()
        self.o = no_of_op_channels
        self.d = depth
        self.k = kernal
        self.p = patch_size
        self.n = output
        self.bn = nn.BatchNorm2d(self.o)
        self.cnn1 = nn.Conv2d(3 , self.o , (self.p , self.p) , stride=self.p)
        self.bn1 = nn.BatchNorm2d(self.o)
        self.cnn2 = nn.Conv2d(self.o , self.o , (self.k , self.k) , groups=self.o , padding="same")
        self.bn2 = nn.BatchNorm2d(self.o)
        self.cnn3 = nn.Conv2d(self.o , self.o , (1,1))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.o , self.n)

    def forward(self , x):
        x = self.bn(F.gelu(self.cnn1(x)))
        for i in range(self.d):
          x = self.bn1(F.gelu(self.cnn2(x)))+ x #residual step and depthwise convolution
          x = self.bn2(F.gelu(self.cnn3(x))) #pointwise convolution
        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x

img = torch.randn(1,3 , 250 , 250)
m = ConvMixer(5 , 3,5,5,10)

print(m(img).shape)

