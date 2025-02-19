import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 5),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual
        return out3


class HourGlass(nn.Module):
    def __init__(self, depth, num_features):
        super(HourGlass, self).__init__()
        # self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class FAN(nn.Module):

    def __init__(self, num_modules=1, in_channels=3):
        super(FAN, self).__init__()
        self.num_modules = num_modules

        # Base part
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.conv1 = conv3x3(3, 64, bias=True)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect') # new add on
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            64, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(64,
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        # x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs


class FanCustom(nn.Module):
    def __init__(self, num_modules=2, in_channels=3):
        super(FanCustom, self).__init__()
        assert 0 <= num_modules
        self.num_modules = num_modules

        self.add_module('FAN', FAN(self.num_modules, in_channels))
        for m in range(self.num_modules):
            self.add_module('Conv1_' + str(m), nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        outs = self._modules['FAN'](x)

        output = []
        for m in range(self.num_modules):
            output.append(self._modules['Conv1_' + str(m)](outs[m]))
        return output

def model_size(model):
    params = [p.shape for n, p in model.named_parameters() if p.requires_grad]
    parasizes = [np.prod(np.asarray(p)) for p in params]
    return sum(parasizes)
