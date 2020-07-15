import torch
import torch.nn as nn


class ResNet(nn.Module):
    """Container module for the ResNet architecture."""

    def __init__(self, nclasses, nblock_layers=2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res1 = ResNetBlock(128, 128, nblock_layers, first=True)
        self.res2 = ResNetBlock(128, 256, nblock_layers)
        self.res3 = ResNetBlock(256, 512, nblock_layers)
        self.end = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, nclasses)
        )

    def init_weights(self):
        def initialiser(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
        self.apply(initialiser)

    def forward(self, X):
        X = self.initial(X)
        X = self.res1(X)
        X = self.res2(X)
        X = self.res3(X)
        Y = self.end(X)
        return Y


class ResNetBlock(nn.Module):
    """Groups residual blocks together, reduces resolution by half"""

    def __init__(self, input_channels, output_channels, nres, first=False):
        super().__init__()
        blocks = []
        for i in range(nres):
            if i == 0 and not first:
                # reduce resolution at 1st residual block
                blocks += [ResidualBlock(input_channels, output_channels,
                                         stride=2, use1x1=True)]
            else:
                blocks += [ResidualBlock(output_channels, output_channels)]

        self.net = nn.Sequential(*blocks)

    def forward(self, X):
        return self.net(X)


class ResidualBlock(nn.Module):
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

        # using PReLU
        self.prelu1 = nn.PReLU(init=0.1)
        self.prelu2 = nn.PReLU(init=0.1)

    def forward(self, X):
        Y = self.conv1(X)
        Y = self.bn1(Y)
        Y = self.prelu1(Y)
        Y = self.conv2(Y)
        Y = self.bn2(Y)
        Y += self.conv3(X)
        Y = self.prelu2(Y)
        return Y


if __name__ == "__main__":
    test_model = ResNet(10)
    test_model.init_weights()
    # print(test_model)
    print(test_model(torch.randn(2, 1, 200, 200)))
