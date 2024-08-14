import torch.nn as nn
import torch
import random
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from .utils import *


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features,):
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        logits = F.linear(F.normalize(input), F.normalize(self.weight))   # [B, 10]
        return logits.clamp(-1, 1)


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_s, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


# ======================== modified ResNet ========================
class ResNet_modify(nn.Module):

    def __init__(self, block, num_blocks, nf=64, args=None):
        super(ResNet_modify, self).__init__()
        self.in_planes = nf
        self.args = args
        self.num_classes = args.num_classes

        self.layer0 = nn.Sequential(nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(self.in_planes),
                                    nn.ReLU(inplace=True),
                                    )
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)

        self.feature = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Flatten()
                                     )
        self.feat_dim = 4 * nf * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, ret=None):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        feat = self.feature(out)
        return feat


class ResNet(nn.Module):

    def __init__(self, block, num_block, args=None):
        super().__init__()
        self.in_channels = 64
        self.num_class = args.num_classes
        self.args = args

        layer_kwargs = {}
        if 'use_se' in args and args.use_se:
            layer_kwargs = {'use_se': args.use_se}

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        # we use a different input_size than the original paper so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64,  num_block[0], 1, **layer_kwargs)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, **layer_kwargs)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, **layer_kwargs)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, **layer_kwargs)

        self.feature = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Flatten()
                                     )
        self.feat_dim = 512 * block.expansion

    def _make_layer(self, block, out_channels, num_blocks, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, ret=None):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)

        feat = self.feature(output)
        return feat

# ======================== Define ResNet ========================

def resnet18(args=None):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], args=args)


def mresnet32(args=None):
    return ResNet_modify(BasicBlock_s, [5, 5, 5], args=args)


def resnet34(args=None):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], args=args)


def resnet50(args=None):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], args=args)


def resnet101(args=None):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], args=args)


# ======================== Define ResNet ========================

model_dict = {'resnet18': resnet18,
              'resnet50': resnet50,
              'mresnet32': mresnet32,
              }

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, args, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun = model_dict[args.arch]
        self.encoder = model_fun(args=args)
        dim_in = self.encoder.feat_dim
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, args):
        super(SupCEResNet, self).__init__()
        model_fun = model_dict[args.arch]
        self.encoder = model_fun(args=args)
        dim_in = self.encoder.feat_dim
        self.fc = nn.Linear(dim_in, args.num_class, bias=args.bias)

    def forward(self, x, ret='o'):
        feat = self.encoder(x)
        out = self.fc(feat)
        if ret == 'o':
            return out
        elif ret == 'of':
            return out, feat


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

