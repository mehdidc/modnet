import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

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
        self.inplanes = inplanes
        self.outplanes = outplanes
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

def basic_model(num_classes=10):
    functions = {
        'f0': Block(3, 64),
        'f1': Block(64, 64, stride=2),
        'f2': Classifier(64, num_classes),
    }
    fs = ['f0', 'f1', 'f1', 'f1', 'f2']
    fs = [functions[f] for f in fs]
    model = Model(fs)
    return model
