"""### Final Model"""
import torch
import torch.nn as nn
from DefConv import *
from ConvLSTM import *
from ddf import DDFUpPack


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class mduvsr(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 scale, state = None):
        super(mduvsr, self).__init__()

        self.convlstm1 = ConvLSTMCell(
            input_size=num_channels,  hidden_size =num_kernels,
            kernel_size=kernel_size, padding=padding)


        self.deformable_convolution1 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        # Add rest of the layers

        self.deformable_convolution2 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        self.deformable_convolution3 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels+num_channels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

        # self.conv = nn.Conv2d(
        #     in_channels=num_kernels + num_channels, out_channels=num_channels * scale ** 2,
        #     kernel_size=kernel_size, padding=padding)

        self.ddfup1 = DDFUpPack(in_channels=num_channels, kernel_size=kernel_size[0],
                                scale_factor=1, head=1, kernel_combine="add").cuda()

        self.ddfup2 = DDFUpPack(in_channels=num_channels, kernel_size=kernel_size[0],
                                scale_factor=scale, head=1, kernel_combine="add").cuda()



        self.batchnorm1 = nn.BatchNorm2d(num_features=num_kernels)

    def forward(self, X, state=None):
        # Forward propagation through all the layers

        lr = X
        output_convlstm = self.convlstm1(X,state)
        x = output_convlstm
        x = torch.cat((x[0],lr),1)
        x = self.deformable_convolution1(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution2(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.conv(x)
        output = self.ddfup2(x)

        output = torch.clamp(output, 0, 255)

        return output, output_convlstm

class mduvsr_6defconv(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 scale, state = None):
        super(mduvsr_6defconv, self).__init__()

        self.convlstm1 = ConvLSTMCell(
            input_size=num_channels,  hidden_size =num_kernels,
            kernel_size=kernel_size, padding=padding)


        self.deformable_convolution1 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        # Add rest of the layers

        self.deformable_convolution2 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        self.deformable_convolution3 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels+num_channels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

        # self.conv = nn.Conv2d(
        #     in_channels=num_kernels + num_channels, out_channels=num_channels * scale ** 2,
        #     kernel_size=kernel_size, padding=padding)

        self.ddfup1 = DDFUpPack(in_channels=num_channels, kernel_size=kernel_size[0],
                                scale_factor=1, head=1, kernel_combine="add").cuda()

        self.ddfup2 = DDFUpPack(in_channels=num_channels, kernel_size=kernel_size[0],
                                scale_factor=scale, head=1, kernel_combine="add").cuda()



        self.batchnorm1 = nn.BatchNorm2d(num_features=num_kernels)

    def forward(self, X, state=None):
        # Forward propagation through all the layers

        lr = X
        output_convlstm = self.convlstm1(X,state)
        x = output_convlstm
        x = torch.cat((x[0],lr),1)
        x = self.deformable_convolution1(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution2(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.conv(x)
        output = self.ddfup2(x)

        output = torch.clamp(output, 0, 255)

        return output, output_convlstm

class mduvsr_6defconv_pixelshuff(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 scale, state = None):
        super(mduvsr_6defconv_pixelshuff, self).__init__()

        self.convlstm1 = ConvLSTMCell(
            input_size=num_channels,  hidden_size =num_kernels,
            kernel_size=kernel_size, padding=padding)


        self.deformable_convolution1 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        # Add rest of the layers

        self.deformable_convolution2 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        self.deformable_convolution3 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        self.conv = nn.Conv2d(
            in_channels=num_kernels + num_channels, out_channels=num_channels * scale ** 2,
            kernel_size=kernel_size, padding=padding)

        self.up_block = nn.PixelShuffle(scale)


    def forward(self, X, state=None):
        # Forward propagation through all the layers

        lr = X
        output_convlstm = self.convlstm1(X,state)
        x = output_convlstm
        x = torch.cat((x[0],lr),1)
        x = self.deformable_convolution1(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution2(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.conv(x)
        output = self.up_block(x)

        output = torch.clamp(output, 0, 255)

        return output, output_convlstm

class mdpvsr(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 scale, state = None):
        super(mdpvsr, self).__init__()

        self.convlstm1 = ConvLSTMCell(
            input_size=num_channels,  hidden_size =num_kernels,
            kernel_size=kernel_size, padding=padding)


        self.deformable_convolution1 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        # Add rest of the layers

        self.deformable_convolution2 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        self.deformable_convolution3 = DeformableConv2d(
            in_channels=num_kernels+num_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        # Add Convolutional Layer to predict output frame
        # self.conv = nn.Conv2d(
        #     in_channels=num_kernels+num_channels, out_channels=num_channels,
        #     kernel_size=kernel_size, padding=padding)

        self.conv = nn.Conv2d(
            in_channels=num_kernels + num_channels, out_channels=num_channels * scale ** 2,
            kernel_size=kernel_size, padding=padding)

        self.up_block = nn.PixelShuffle(scale)

    def forward(self, X, state=None):
        # Forward propagation through all the layers

        lr = X
        output_convlstm = self.convlstm1(X,state)
        x = output_convlstm
        x = torch.cat((x[0],lr),1)
        x = self.deformable_convolution1(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution2(x)
        x = torch.cat((x,lr),1)
        x = self.deformable_convolution3(x)
        x = torch.cat((x,lr),1)
        x = self.conv(x)
        output = self.up_block(x)

        output = torch.clamp(output, 0, 255)

        return output, output_convlstm