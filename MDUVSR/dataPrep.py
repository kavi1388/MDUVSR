import os
import torch
from PIL import Image, ImageOps
print(torch.__version__)
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from patchify import patchify

# !pip install patchify

# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("hr_data", type=str, help="HR Path")
parser.add_argument("lr_data", type=str, help="LR Path")
# parser.add_argument("res_hr_patches", type=str, help="res_hr_patches path")
# parser.add_argument("res_lr_patches", type=str, help="res_lr_patches path")
parser.add_argument("batch_size", type=int, help="batch size")
parser.add_argument("workers", type=int, help="workers")


# Read arguments from command line
# args = parser.parse_args()
#
# # +
# hr_path = args.hr_data
# lr_path = args.lr_data
#
# res_hr_patches = args.res_hr_patches
# res_lr_patches = args.res_lr_patches
# batch_size = args.batch_size
# workers = args.workers

# hr_path = r'Custom Dataset/frames/hr'
# lr_path = r'Custom Dataset/frames/lr'
# batch_size = 32
# workers = 2
# -


# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load Data as Numpy Array
def read_data(path):
    data = []
    patch = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if len(data)<1200000:
                f = os.path.join(dirname, filename)
    #             print(f)
                if filename.split('.')[-1] == 'jpg':
                    img = Image.open(f)
                    img.load()
                    img_array = np.asarray(img)
                    img_array = np.swapaxes(img_array, np.where(np.asarray(img_array.shape) == min(img_array.shape))[0][0], 0)
                    # data.append(img_array)
                    patches = patchify(img_array, (3, img_array.shape[1] // 4, img_array.shape[2] // 4), step=(img_array.shape[1] // 8))
    #                 print(patches.shape)
                    for i in range(patches.shape[0]):
                        for j in range(patches.shape[1]):
                            for k in range(patches.shape[2]):
                                patch = patches[i, j, k]
                                data.append(patch)
    #                             patch = Image.fromarray(patch.T)
    #                             num = i * patches.shape[1] + j
    #                             patch.save(f"{respath}/{filename.split('.')[0]}_patch_{num}.jpg")
    return data

# +
# data = []
# patch = []
# c=0
# for dirname, _, filenames in os.walk(lr_path):
#         for filename in filenames:
#             c+=1
#             if c==1:
            
#                 f = os.path.join(dirname, filename)
#                 print(f)
#                 if filename.split('.')[-1] == 'jpg':
#                     img = Image.open(f)
#                     img.load()
#                     img_array = np.asarray(img)
#                     img_array = np.swapaxes(img_array, np.where(np.asarray(img_array.shape) == min(img_array.shape))[0][0], 0)
#                     # data.append(img_array)
#                     patches = patchify(img_array, (3, img_array.shape[1] // 4, img_array.shape[2] // 4), step=(img_array.shape[1] // 8))
#                     print(patches.shape)
#                     for i in range(patches.shape[0]):
#                         for j in range(patches.shape[1]):
#                             print k in range(patches.shape[2])
#                             data.append(patches[i, j, k])
                
                

# -

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


def data_load(all_lr_data,all_hr_data, batch_size, workers):

    # Train, Test, Validation splits
    train_data_hr = all_hr_data[:len(all_hr_data)//3]
    train_data_lr = all_lr_data[:len(all_lr_data)//3]
    print(f'len of train hr data ={len(train_data_hr)}')

    val_data_hr = all_hr_data[len(all_hr_data)//3:(len(all_hr_data)//3)+(len(all_hr_data)//4)]
    val_data_lr = all_lr_data[len(all_lr_data)//3:(len(all_lr_data)//3)+(len(all_lr_data)//4)]

    print(f'len of val hr data ={len(val_data_hr)}')
    # test_data_hr = all_hr_data[(len(all_hr_data)//3)+(len(all_hr_data)//4):(len(all_hr_data)//3)+(len(all_hr_data)//2)]
    # test_data_lr = all_lr_data[(len(all_lr_data)//3)+(len(all_lr_data)//4):(len(all_lr_data)//3)+(len(all_lr_data)//2)]

    print(f'hr {len(all_hr_data)}')
    print(f'lr {len(all_lr_data)}')

    train_data = CustomDataset(np.asarray(train_data_lr),np.asarray(train_data_hr))
    val_data = CustomDataset(np.asarray(val_data_lr),np.asarray(val_data_hr))
    # test_data = CustomDataset(np.asarray(test_data_lr),np.asarray(test_data_hr))
    print(f'dataset created')
    # Load Data as Numpy Array
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    print(f'train {len(train_data_hr)}')    
    print(f'val {len(val_data_hr)}')
    # print(f'test {len(test_data_hr)}')

#     torch.save(train_loader, 'train_loader.pt', pickle_protocol=4)
#     torch.save(val_loader, 'val_loader.pt', pickle_protocol=4)
#     torch.save(test_loader, 'test_loader.pt', pickle_protocol=4)

    return train_loader, val_loader
# -

# all_hr_data = read_data(hr_path)
# all_lr_data = read_data(lr_path)
# print('read')
#
# data_load(all_lr_data,all_hr_data)
