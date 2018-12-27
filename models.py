import torch.nn as nn
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
        self.block = make_block(inplanes, outplanes,
                                block_type, 1, stride=stride)

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


class FC(nn.Module):

    def __init__(self, inplane, outplane=1000):
        super().__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.fc = nn.Linear(inplane, outplane)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Controller(nn.Module):
    pass


class ModularNetController(Controller):

    def __init__(self, modules):
        super().__init__()
        self.inplane = modules[0].inplane
        self.outplane = modules[-1].outplane
        self.controller = nn.Sequential(
            nn.Conv2d(self.inplane, len(modules), kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.components = nn.ModuleList(modules)
        self.cur_assignments = None

    def forward(self, x):
        ctl = self.controller(x)
        ctl_logits = ctl.view(ctl.size(0), -1)
        _, ctl_decisions = ctl_logits.max(dim=1)
        outs = []
        for i, decision in enumerate(ctl_decisions):
            outs.append(self.components[decision](x[i:i+1]))
        out = torch.cat(outs, dim=0)
        return out

    def forward_E_step(self, x):
        ctl = self.controller(x)
        ctl_logits = ctl.view(ctl.size(0), -1)
        ctl_probs = nn.Softmax(dim=1)(ctl_logits)
        device = next(self.parameters()).device
        ctl_decisions = torch.multinomial(ctl_probs, 1)[:, 0].to(device)
        outs = []
        for i, decision in enumerate(ctl_decisions):
            outs.append(self.components[decision](x[i:i+1]))
        out = torch.cat(outs, dim=0)
        return out, ctl_logits, ctl_decisions

    def forward_M_step(self, x, indices):
        ctl = self.controller(x)
        ctl_logits = ctl.view(ctl.size(0), -1)
        ctl_decisions = self.cur_assignments[indices]
        outs = []
        for i, decision in enumerate(ctl_decisions):
            outs.append(self.components[decision](x[i:i+1]))
        out = torch.cat(outs, dim=0)
        return out, ctl_logits, ctl_decisions


class ModularNet(nn.Module):

    def __init__(self, layers, nb_trials=10):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.nb_trials = nb_trials

    @property
    def controllers(self):
        for layer in self.layers:
            if isinstance(layer, Controller):
                yield layer

    def initialize_assignments(self, nb_examples):
        for ctl in self.controllers:
            ctl.cur_assignments = torch.randint(
                len(ctl.components), (nb_examples,))

    def forward_E_step(self, input, indices):
        trial_outputs = []
        trial_decisions = []
            
        # cur assignment
        output, _, decisions = self.forward_M_step(input, indices)
        trial_outputs.append(output)
        trial_decisions.append(torch.stack(decisions, dim=0))

        # sampled assignments
        for trial in range(self.nb_trials):
            x = input
            controller_decisions = []
            for layer in self.layers:
                if isinstance(layer, Controller):
                    x, logits, decisions = layer.forward_E_step(x)
                    controller_decisions.append(decisions)
                else:
                    x = layer(x)
            trial_outputs.append(x)
            controller_decisions = torch.stack(controller_decisions, dim=0)
            trial_decisions.append(controller_decisions)
        #nb_trials, batch_size, nb_classes
        trial_outputs = torch.stack(trial_outputs, dim=0)
        #nb_trials, nb_controllers, batch_size
        trial_decisions = torch.stack(trial_decisions, dim=0)
        return trial_outputs, trial_decisions

    def update_assignments(self, indices, y_true, outputs, decisions):
        controllers = [
            layer for layer in self.layers if isinstance(layer, Controller)]
        #nb_trials, nb_examples
        p = self.log_likelihood(outputs, y_true)
        # nb_examples
        _, best_trial = p.max(dim=0)
        for i in range(len(indices)):
            example_index = indices[i]
            best_decisions = decisions[best_trial[i]]
            for j, controller in enumerate(controllers):
                controller.cur_assignments[example_index] = best_decisions[j][i]

    def log_likelihood(self, pred, true):
        #pred: (nb_trials, nb_examples, nb_classes)
        #true: (nb_examples,)
        nb_trials, nb_examples, nb_classes = pred.size()
        pred_ = pred.view(-1, nb_classes)
        true_ = true.view(1, -1)
        true_ = true_.expand(pred.size(0), true.size(0))
        true_ = true_.contiguous()
        true_ = true_.view(-1)
        true_ = true_.long()
        prob = -nn.functional.cross_entropy(pred_, true_, reduction='none')
        prob = prob.view(nb_trials, nb_examples)
        return prob

    def forward_M_step(self, x, indices):
        logits_list = []
        decisions_list = []
        for layer in self.layers:
            if isinstance(layer, Controller):
                x, logits, decisions = layer.forward_M_step(x, indices)
                logits_list.append(logits)
                decisions_list.append(decisions)
            else:
                x = layer(x)
        return x, logits_list, decisions_list

    def M_step_loss(self, logits_list, decisions_list):
        loss = 0
        for logits, decisions in zip(logits_list, decisions_list):
            loss += nn.functional.cross_entropy(logits, decisions)
        return loss

    def forward(self, x):
        for layer in self.layers:
            if isinstance(x, Controller):
                x, probs, decisions = layer.forward_train(x)
            else:
                x = layer(x)
        return x


def simple(nb_colors=3, nb_classes=10):
    net = nn.Sequential(
        Block(nb_colors, 64),
        Block(64, 64),
        Block(64, 128, stride=2),
        Block(128, 128),
        Block(128, 256, stride=2),
        Block(256, 256),
        Block(256, 512, stride=2),
        FC(512, nb_classes),
    )
    net.apply(weight_init)
    return net


def modular_simple(nb_colors=3, nb_classes=10):
    f1 = Block(nb_colors, 64)
    f2 = Block(nb_colors, 64)
    f3 = Block(64, 128, stride=2)
    f4 = Block(64, 128, stride=2)
    f5 = Block(128, 128, stride=2)
    f6 = Block(128, 128, stride=2)
    net = ModularNet([
        ModularNetController([f1, f2]),
        ModularNetController([f3, f4]),
        ModularNetController([f5, f6]),
        FC(128, nb_classes)
    ])
    net.apply(weight_init)
    return net


if __name__ == '__main__':
    net = modular_simple(nb_classes=2)
    x = torch.rand(5, 3, 32, 32)
    y = torch.Tensor([1, 0, 0, 1, 0])
    net.initialize_assignments(len(x))

    inds = torch.arange(0, len(x))
    o, d = net.forward_E_step(x)
    for cnt in net.controllers:
        print(cnt.cur_assignments)
    print('Update')
    net.update_assignments(inds, y, o, d)
    for cnt in net.controllers:
        print(cnt.cur_assignments)
