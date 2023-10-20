import torch.nn as nn
import torch


class GaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x):
        return torch.clamp(
            x + torch.randn(x.size()) * self.std + self.mean,
            min=0.,
            max=1.
        )


def DoubleConvolution(in_channels, out_channels, kernel_size=3, padding=1):
	'''Generic double convolution layer'''
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)


def FullConnection(in_channels, out_channels):
	'''Generic full connection layer'''
	return nn.Sequential(
		nn.Linear(in_channels, out_channels),
		nn.ReLU(inplace=True)
	)

class DigitOCR(nn.Module):
    def __init__(self):
        super().__init__()

        self.C1 = DoubleConvolution(1, 16, 5, 2)
        self.S1 = nn.MaxPool2d(2)

        self.C2 = DoubleConvolution(16, 32, 5, 2)
        self.S2 = nn.MaxPool2d(2)

        self.C3 = DoubleConvolution(32, 64, 5, 2)
        self.S3 = nn.MaxPool2d(2)

        self.F1 = FullConnection(576, 288)
        self.F2 = FullConnection(288, 144)

        self.last = nn.Linear(144, 11)
        self.soft = nn.Softmax(dim=1)

    def head(self, x):
        x = self.last(x)

        return x if self.training else self.soft(x)

    def forward(self, x):
        # Features extraction
        x = self.C1(x)
        x = self.S1(x)

        x = self.C2(x)
        x = self.S2(x)

        x = self.C3(x)
        x = self.S3(x)

        # Classification
        x = torch.flatten(x, 1)

        x = self.F1(x)
        x = self.F2(x)

        return self.head(x)

