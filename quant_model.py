import torch
import torch.nn as nn
# needed to collect statistics for quantisation
from torch.quantization import QuantStub, DeQuantStub, fuse_modules


class QuantResNet(nn.Module):
    """Container module for the ResNet architecture."""

    def __init__(self, nclasses, nblock_layers=2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.res1 = QuantResNetBlock(16, 16, nblock_layers, first=True)
        self.res2 = QuantResNetBlock(16, 16, nblock_layers)
        self.res3 = QuantResNetBlock(16, 32, nblock_layers)
        self.res4 = QuantResNetBlock(32, 64, nblock_layers)
        self.end = nn.Sequential(
            # only works for 64x64 input images
            # feature maps are 8x8 here
            nn.MaxPool2d(8),
            nn.Flatten(),
            nn.Linear(64, nclasses)
        )
        self.quant1 = QuantStub()
        self.quant2 = QuantStub()
        self.dequant = DeQuantStub()
        self.init_weights()

    def init_weights(self):
        def initialiser(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
        self.apply(initialiser)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == QuantResidualBlock:
                fuse_modules(m, ["conv1", "bn1", "ReLU1"], inplace=True)
        fuse_modules(self.initial, ["0", "1", "2"], inplace=True)

    def forward(self, X):
        # added quantisation stubs
        X = self.quant1(X)
        X = self.initial(X)
        X = self.dequant(X)
        X = self.res1(X)
        X = self.res2(X)
        X = self.res3(X)
        X = self.res4(X)
        X = self.quant2(X)
        Y = self.end(X)
        Y = self.dequant(Y)
        return Y


class QuantResNetBlock(nn.Module):
    """Groups residual blocks together, reduces resolution by half"""

    def __init__(self, input_channels, output_channels, nres, first=False):
        super().__init__()
        blocks = []
        for i in range(nres):
            if i == 0 and not first:
                # reduce resolution at 1st residual block
                blocks += [QuantResidualBlock(input_channels, output_channels,
                                              stride=2, use1x1=True)]
            else:
                blocks += [QuantResidualBlock(output_channels,
                                              output_channels)]

        self.net = nn.Sequential(*blocks)
        
    def forward(self, X):
        return self.net(X)


class QuantResidualBlock(nn.Module):
    """Module for single residual block in ResNet."""

    def __init__(self, input_channels, output_channels,
                 stride=1, use1x1=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                               kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                               kernel_size=3, padding=1)
        if use1x1:
            # 1x1 convolution for the skip connection
            self.conv3 = nn.Conv2d(input_channels, output_channels,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = nn.Identity()
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

        # using ReLU
        self.ReLU1 = nn.ReLU()
        self.ReLU2 = nn.ReLU()

        # needed for quantisation, replaces normal float operation
        self.add1 = nn.quantized.FloatFunctional()
        # fuses when quanitsed
       # self.conv_bn_relu = torch.nn.intrinsic.ConvBnReLU2d(self.conv1,
        # self.bn1,
        # self.ReLU1)
        self.quant1 = QuantStub()
        self.quant2 = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, X):
        # Y = self.conv_bn_relu(X)
        X = self.quant1(X)
        Y = self.conv1(X)
        Y = self.bn1(Y)
        Y = self.ReLU1(Y)
        Y = self.conv2(Y)
        Y = self.bn2(Y)
        X = self.conv3(X)
        Y = self.dequant(Y)
        X = self.dequant(X)
        # Y = self.add1.add(self.conv3(X), Y)
        Y += X
        Y = self.quant2(Y)
        Y = self.ReLU2(Y)
        Y = self.dequant(Y)
        return Y


if __name__ == "__main__":
    test_model = QuantResNet(10)
    test_model.init_weights()
    test_model.fuse_model()
    print(test_model.initial)
    print(test_model.res1.net[0])
    print(test_model(torch.randn(2, 3, 64, 64)))
