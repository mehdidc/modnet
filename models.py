import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import math

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Block(nn.Module):

    def __init__(self, inplanes, outplanes, block_type=BasicBlock, stride=1):
        super().__init__()
        self.inplane = inplanes
        self.outplane = outplanes
        self.block_type = block_type
        self.stride = stride
        self.block = make_block(inplanes, outplanes, block_type, 1, stride=stride)

    def forward(self, x):
        return self.block(x)


def make_block(inplanes, outplanes, block, nb_blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            conv1x1(inplanes, outplanes, stride),
            nn.BatchNorm2d(outplanes),
        )
    layers = []
    layers.append(block(inplanes, outplanes, stride, downsample))
    for _ in range(1, nb_blocks):
        layers.append(block(outplanes, outplanes))
    return nn.Sequential(*layers)

class Classifier(nn.Module):

    def __init__(self, inplane, outplane=1000):
        super().__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.fc = nn.Linear(inplane, outplane)
    
    def forward(self, x):
        x = x.mean(3)
        x = x.mean(2)
        return self.fc(x)


class Model(nn.Module):

    def __init__(self, funcs):
        super().__init__()
        self.funcs = nn.ModuleList(funcs)
    
    def forward(self, x):
        for f in self.funcs:
            x = f(x)
        return x

class Controller(nn.Module):

    def __init__(self, modules):
        super().__init__()
        self.inplane = modules[0].inplane 
        self.outplane = modules[-1].outplane
        self.controller = nn.Sequential(
            nn.Conv2d(self.inplane, len(modules), kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.components = nn.ModuleList(modules)

    def forward(self, x):
        ctl = self.controller(x)
        #bs, nmods
        ctl = ctl.view(ctl.size(0), -1)
        
        #D = 0
        #ctl_max = ctl.max(dim=D, keepdim=True)[0].expand(ctl.size())
        #ctl = ctl * (ctl == ctl_max).float()

        ctl = nn.Softmax(dim=1)(ctl)

        
        #nmods, bs
        ctl = ctl.transpose(0, 1)
        #nmods, bs, 1, 1, 1
        ctl = ctl.view(ctl.size() + (1, 1, 1))
        #nmods, bs, c, h, w
        outs = torch.stack([mod(x) for mod in self.components], dim=0)
        out = (outs * ctl).sum(0)
        return out


def basic_model(num_classes=10):
    f1 = Block(3, 64)
    f2 = Block(3, 64)
    f3 = Block(64, 128, stride=2)
    f4 = Block(64, 128, stride=2)
    f5 = Block(128, 128, stride=2)
    f6 = Block(128, 128, stride=2)
    net = nn.Sequential(
        Controller([f1, f2]),
        Controller([f3, f4]),
        Controller([f5, f6]),
        Classifier(128, num_classes)
    )
    net.apply(weight_init)
    return net
