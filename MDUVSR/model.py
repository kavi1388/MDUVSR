"""### Final Model"""
import torch
import torch.nn as nn
from DefConv import *
from ConvLSTM import *
from ddf import DDFUpPack
class mduvsr(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 activation, frame_size, num_layers, scale):
        super(mduvsr, self).__init__()

        self.convlstm1 = ConvLSTM(
            in_channels=num_channels, out_channels=num_kernels,
            kernel_size=kernel_size, padding=padding,
            activation=activation, frame_size=frame_size)


        self.deformable_convolution1 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        # Add rest of the layers

        self.deformable_convolution2 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        self.deformable_convolution3 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels+num_channels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

        self.ddfup1 = DDFUpPack(in_channels=num_channels, kernel_size=3,
                                scale_factor=1, head=1, kernel_combine="add").cuda()

        self.ddfup2 = DDFUpPack(in_channels=num_channels, kernel_size=3,
                                scale_factor=scale, head=1, kernel_combine="add").cuda()

        self.batchnorm1 = nn.BatchNorm2d(num_features=num_kernels)

    def forward(self, X):
        # Forward propagation through all the layers
        lr = X
        x = self.convlstm1(X)

        x = torch.cat((x,lr),1)
        x = self.deformable_convolution1(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution2(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.conv(x)
        output = self.ddfup2(x)

        output = torch.clamp(output, 0, 255)

        return output