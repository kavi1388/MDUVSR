import torch
from ddfnet.ddf import DDFUpPack

DDFUpnet = DDFUpPack(in_channels=100, kernel_size=3,scale_factor=2, head=1, kernel_combine="add").cuda()

image = torch.ones(1, 100, 256, 256).cuda()
output = DDFUpnet(image)   # output is of size [1, 100, 512, 512]  (default upscaling of 2)
