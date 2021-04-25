import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv1D(nn.Module):
    """
    Convolution1d Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        '''
         x: (batch, length, in_channels)
        '''
        x = x.contiguous().transpose(1, 2)
        x = self.conv1d(x)
        x = x.contiguous().transpose(1, 2)

        return x
