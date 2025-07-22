import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class RandConv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, prob=0.5, bias=False, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.prob = prob
        self.bias = bias
        self.dilation = dilation
        self.weight_mean = 0.0
        self.weight_std = 0.01

        if bias:
            self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.prob:
            return x

        b, c, h, w = x.shape

        # Generate weight of shape [c, 1, k, k] for depthwise conv
        weight = torch.randn(c, 1, self.kernel_size, self.kernel_size,
                             device=x.device) * self.weight_std + self.weight_mean

        # Normalize each kernel
        weight = F.normalize(weight.view(c, -1), dim=1).view_as(weight)

        # Apply depthwise convolution
        out = F.conv2d(
            x, weight,
            bias=self.b if self.bias else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=c  # depthwise
        )

        return out
