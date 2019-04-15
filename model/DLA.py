import torch
import torch.nn as nn 
from torch.nn import functional as F 
import torch.nn.init as init
from ResNet import BasicBlock, make_layer

#reference: LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving
class FeatureExtractor(nn.Module):
    def __init__(self, inplanes, planes, blocks, stride=1):
        super(FeatureExtractor, self).__init__()
        self.name = 'FeatureExtractor'
        self.block = make_layer(BasicBlock, inplanes, planes, blocks, stride=stride)

    def forward(self, x):
        x = self.block(x)
        return x 

class FeatureAggregator(nn.Module):
    def __init__(self, FinePlanes, CoarsePlanes, planes):
        super(FeatureAggregator, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(CoarsePlanes, planes, 3, stride=2, padding=1, output_padding=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.block2 = make_layer(BasicBlock, FinePlanes+planes, planes, 2, stride=1)

    def forward(self, xFine, xCoarse):
        x1 = self.block1(xCoarse)
        x = torch.cat((x1, xFine), 1)
        x = self.block2(x)
        return x

class DLA(nn.Module):
    def __init__(self, num_class):
        super(DLA, self).__init__()
        self.block1a = FeatureExtractor(3, 64, 6, stride=1)
        self.block2a = FeatureExtractor(64, 64, 6, stride=2)
        self.block3a = FeatureExtractor(64, 128, 6, stride=2)
        self.block1b = FeatureAggregator(64, 64, 64)
        self.block1c = FeatureAggregator(64, 128, 128)
        self.block2b = FeatureAggregator(64, 128, 128)

        self.cls = nn.Conv2d(128, num_class, 1)

    def forward(self, x):
        x1 = self.block1a(x)
        x2 = self.block2a(x1)
        y1 = self.block1b(x1, x2)
        x3 = self.block3a(x2)
        y2 = self.block2b(x2, x3)
        y = self.block1c(y1, y2)
        y = self.cls(y)
        return y


def test():
    net = DLA(5)
    x = torch.randn(1, 3, 100, 100)
    y = net(x)
    print(y[0,0,0,:])

# test()