import os

import torch
from PIL import Image, ImageOps
print(torch.__version__)
import piq
from model import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
from torchsummary import summary
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("hr_data", type=str, help="HR Path")
parser.add_argument("lr_data", type=str, help="LR Path")
parser.add_argument("result", type=str, help="result Path (to save)")
parser.add_argument("scale", type=int, help="downsampling scale")
parser.add_argument("epochs", type=int, help="epochs")
parser.add_argument("name", type=str, help="model name")
# Read arguments from command line
args = parser.parse_args()


hr_path = args.hr_data
lr_path = args.lr_data
res_path = args.result
scale = args.scale
epochs = args.epochs
name = args.name

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load Data as Numpy Array
def read_data(path):
    data=[]
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            f = os.path.join(dirname, filename)
            # print(f)
            if filename.split('.')[-1] == 'jpg':
                img = Image.open(f)
                img.load()
                img_array = np.asarray(img)
                img_array = np.swapaxes(img_array, np.where(np.asarray(img_array.shape)==min(img_array.shape))[0][0], 0)
                data.append(img_array)
    return data


all_hr_data = read_data(hr_path)
all_lr_data = read_data(lr_path)
#
# for dirname, _, filenames in os.walk(lr_path):
#     for filename in filenames:
#         f = os.path.join(dirname, filename)
#         # print(f)
#         if filename.split('.')[-1] == 'jpg':
#             img = Image.open(f)
#             img.load()
#             img_array = np.asarray(img)
#             img_array = np.swapaxis(img_array,img_array.shape.min(),0)
#             all_lr_data.append(img_array)

print(all_lr_data[0].shape)
print(all_hr_data[0].shape)
# Train, Test, Validation splits
train_data_hr = all_hr_data[:len(all_hr_data)//2]
train_data_lr = all_lr_data[:len(all_lr_data)//2]

val_data_hr = all_hr_data[len(all_hr_data)//2:(len(all_hr_data)//2)+(len(all_hr_data)//4)]
val_data_lr = all_lr_data[len(all_lr_data)//2:(len(all_lr_data)//2)+(len(all_lr_data)//4)]

test_data_hr = all_hr_data[(len(all_hr_data)//2)+(len(all_hr_data)//4):]
test_data_lr = all_lr_data[(len(all_lr_data)//2)+(len(all_lr_data)//4):]

print(f'hr shape {all_hr_data[0].shape}')
print(f'lr shape {all_lr_data[0].shape}')


class CustomDataset(Dataset):
    def __init__(self, image_data, labels):
        self.image_data = image_data
        self.labels = labels

    def __len__(self):
        return (len(self.image_data))

    def __getitem__(self, index):
        image = self.image_data[index]
        label = self.labels[index]
        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )


train_data = CustomDataset(np.asarray(train_data_lr),np.asarray(train_data_hr))
val_data = CustomDataset(np.asarray(val_data_lr),np.asarray(val_data_hr))
test_data = CustomDataset(np.asarray(test_data_lr),np.asarray(train_data_hr))

train_loader = DataLoader(train_data, batch_size=8, shuffle=False, num_workers=2)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=2)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=2)

"""### Defining Model"""

print('Computation device: ', device)
model = mduvsr(num_channels=all_lr_data[0].shape[0], num_kernels=all_lr_data[0].shape[1]//2,
               kernel_size=(3, 3), padding=(1, 1), activation="relu",
               frame_size=(all_lr_data[0].shape[1],all_lr_data[0].shape[2]), num_layers=3, scale=scale).to(device)

print(model)
print(summary(model, (all_lr_data[0].shape)))
"""### Loss Function"""


def get_outnorm(x: torch.Tensor, out_norm: str = '') -> torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1] * img_shape[-2]

    return norm


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, out_norm: str = 'bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps ** 2))
        return loss * norm


"""### Training"""


optim = Adam(model.parameters(), lr=1e-4)

# criterion = nn.L1Loss()
criterion = CharbonnierLoss()
num_epochs = epochs

# model.to(device)
for epoch in range(1, num_epochs + 1):

    train_loss = 0
    model.train()
    for batch_num, data in enumerate(train_loader, 0):
        input, target = data[0].to(device), data[1].to(device)
        # print(f'input {input.size()}')
        # plt.figure()
        # plt.title('Input')
        # plt.imshow(input[-1][0].cpu())
        # print(f'target {target.size()}')
        output = model(input.cuda())
        # plt.figure()
        # plt.imshow(output.cpu()[-1][0].detach().numpy())
        loss = criterion(output, target.cuda())
        loss.backward()
        optim.step()
        optim.zero_grad()
        train_loss += loss.item()
        torch.cuda.empty_cache()
    train_loss /= len(train_loader.dataset)

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for input, target in val_loader:
            output = model(input.cuda())
            loss = criterion(output, target.cuda())
            val_loss += loss.item()
    val_loss /= len(val_loader.dataset)

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))


"""### Save the Model"""

params = f'{epochs} epochs, charbonnier, 1 dfup,1 convlstm, 3 deformable num_channels={all_lr_data[0].shape[0]} num_kernels={all_lr_data[0].shape[1]//2},' \
         f'kernel_size={(3, 3)}, padding={(1, 1)}, activation={"relu"},' \
         f'frame_size={(all_lr_data[0].shape[1],all_lr_data[0].shape[2])}, ' \
         f'scale={scale}  {name}'
PATH = f'mdu-vsr-customdataser-{params}.pth'
torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))


model.eval()
ssim_val = []
psnr_val = []
lpips_val = []
running_psnr = 0

with torch.no_grad():
    for input, target in test_loader:
        output = model(input.cuda())
        psnr_val.append(piq.psnr(output, target.cuda(), data_range=255., reduction='mean'))
        ssim_val.append(piq.ssim(output, target.cuda(), data_range=255.))
        lpips_val.append(piq.LPIPS(reduction='mean')(torch.clamp(output, 0, 1), torch.clamp(target.cuda(), 0, 255)))

        print(f'psnr value ={psnr_val[-1]}')
        print(f'ssim value ={ssim_val[-1]}')
        print(f'lpips value ={lpips_val[-1]}')

    with open(r'name_quality metrics', 'w') as fp:
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



