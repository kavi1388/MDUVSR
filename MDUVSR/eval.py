import os
import time
import torch
from PIL import Image, ImageOps
print(torch.__version__)
import piq
from model import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchsummary import summary
from dataPrep import read_data, CustomDataset

"""### Preparing Data"""
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

# +
# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument


parser.add_argument("model", type=str, help="model to use")
parser.add_argument("hr_data", type=str, help="HR Path")
parser.add_argument("lr_data", type=str, help="LR Path")
parser.add_argument("batch_size", type=int, help="batch size")
parser.add_argument("workers", type=int, help="workers")
parser.add_argument("result", type=str, help="result Path (to save)")
parser.add_argument("scale", type=int, help="downsampling scale")
parser.add_argument("PATH", type=str, help="path")
# Read arguments from command line
args = parser.parse_args()
# -

model_to_use = args.model
res_path = args.result
scale = args.scale
PATH = args.PATH
hr_path = args.hr_data
lr_path = args.lr_data
batch_size = args.batch_size
workers = args.workers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(res_path):
    os.makedirs(res_path)

all_hr_data = read_data(hr_path)
all_lr_data = read_data(lr_path)

test_data = CustomDataset(np.asarray(all_lr_data),np.asarray(all_hr_data))
print(f'dataset created')
# Load Data as Numpy Array
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers)

print('loaded')
print(test_loader.dataset[0][0].shape)
# ### Defining Model

print('Computation device: ', device)

if model_to_use == 'mdpvsr':
    model = mdpvsr(num_channels=test_loader.dataset[0][0].shape[0], num_kernels=test_loader.dataset[0][0].shape[1]//2,
               kernel_size=(3,3), padding=(1, 1), scale=scale).to(device)

elif model_to_use == 'mduvsr_6defconv':
    model = mduvsr_6defconv(num_channels=test_loader.dataset[0][0].shape[0], num_kernels=test_loader.dataset[0][0].shape[1]//2,
               kernel_size=(3,3), padding=(1, 1), scale=scale).to(device)

elif model_to_use == 'mduvsr_6defconv_pixelshuff':
    model = mduvsr_6defconv_pixelshuff(num_channels=test_loader.dataset[0][0].shape[0], num_kernels=test_loader.dataset[0][0].shape[1] // 2,
    kernel_size=(3, 3), padding = (1, 1), scale = scale).to(device)

elif model_to_use == 'mduvsr_1defconv':
    model = mduvsr_1defconv(num_channels=test_loader.dataset[0][0].shape[0], num_kernels=test_loader.dataset[0][0].shape[1] // 2,
    kernel_size=(3, 3), padding = (1, 1), scale = scale).to(device)

elif model_to_use == 'mduvsr_2defconv':
    model = mduvsr_2defconv(num_channels=test_loader.dataset[0][0].shape[0], num_kernels=test_loader.dataset[0][0].shape[1] // 2,
    kernel_size=(3, 3), padding = (1, 1), scale = scale).to(device)

elif model_to_use == 'mdpvsr_1defconv':
    model = mdpvsr_1defconv(num_channels=test_loader.dataset[0][0].shape[0], num_kernels=test_loader.dataset[0][0].shape[1] // 2,
    kernel_size=(3, 3), padding = (1, 1), scale = scale).to(device)

elif model_to_use == 'mdpvsr_2defconv':
    model = mdpvsr_2defconv(num_channels=test_loader.dataset[0][0].shape[0], num_kernels=test_loader.dataset[0][0].shape[1] // 2,
    kernel_size=(3, 3), padding = (1, 1), scale = scale).to(device)

else:
    model = mduvsr(num_channels=test_loader.dataset[0][0].shape[0],
                   num_kernels=test_loader.dataset[0][0].shape[1] // 2,
                   kernel_size=(3, 3), padding=(1, 1), scale=scale).to(device)

print(model)
print(summary(model, (test_loader.dataset[0][0].shape)))

model.load_state_dict(torch.load(PATH))


model.eval()
ssim_val = []
psnr_val = []
lpips_val = []
running_psnr = 0

state=(None,None)
with torch.no_grad():
    for input, target in test_loader:
        state = model(input.cuda(), state[1])
        state[1][0].detach_()
        state[1][1].detach_()
        output = state[0]
        psnr_val.append(piq.psnr(output, target.cuda(), data_range=255., reduction='mean'))
        ssim_val.append(piq.ssim(output, target.cuda(), data_range=255.))
        lpips_val.append(piq.LPIPS(reduction='mean')(torch.clamp(output, 0, 1), torch.clamp(target.cuda(), 0, 255)))

        print(f'psnr value ={psnr_val[-1]}')
        print(f'ssim value ={ssim_val[-1]}')
        print(f'lpips value ={lpips_val[-1]}')

    with open(f'{res_path}/name_quality metrics', 'w') as fp:
        fp.write("\n PSNR")
        for item in psnr_val:
            # write each item on a new line
            fp.write("%s\n" % item)

        fp.write("\n SSIM")
        for item in ssim_val:
            # write each item on a new line
            fp.write("%s\n" % item)

        fp.write("\n LPIPS")
        for item in lpips_val:
            # write each item on a new line
            fp.write("%s\n" % item)


