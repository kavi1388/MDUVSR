import os
import torch
from PIL import Image, ImageOps
print(torch.__version__)
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse

# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("hr_data", type=str, help="HR Path")
parser.add_argument("lr_data", type=str, help="LR Path")
parser.add_argument("batch_size", type=int, help="batch size")
parser.add_argument("workers", type=int, help="workers")


# Read arguments from command line
args = parser.parse_args()

hr_path = args.hr_data
lr_path = args.lr_data
batch_size = args.batch_size
workers = args.workers


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

def data_load(all_lr_data,all_hr_data):

    # Train, Test, Validation splits
    train_data_hr = all_hr_data[:len(all_hr_data)//2]
    train_data_lr = all_lr_data[:len(all_lr_data)//2]

    val_data_hr = all_hr_data[len(all_hr_data)//2:(len(all_hr_data)//2)+(len(all_hr_data)//4)]
    val_data_lr = all_lr_data[len(all_lr_data)//2:(len(all_lr_data)//2)+(len(all_lr_data)//4)]

    test_data_hr = all_hr_data[(len(all_hr_data)//2)+(len(all_hr_data)//4):]
    test_data_lr = all_lr_data[(len(all_lr_data)//2)+(len(all_lr_data)//4):]

    print(f'hr shape {all_hr_data[0].shape}')
    print(f'lr shape {all_lr_data[0].shape}')

    train_data = CustomDataset(np.asarray(train_data_lr),np.asarray(train_data_hr))
    val_data = CustomDataset(np.asarray(val_data_lr),np.asarray(val_data_hr))
    test_data = CustomDataset(np.asarray(test_data_lr),np.asarray(test_data_hr))

    # Load Data as Numpy Array
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers)

    torch.save(train_loader, 'train_loader.pt', pickle_protocol=4)
    torch.save(val_loader, 'val_loader.pt', pickle_protocol=4)
    torch.save(test_loader, 'test_loader.pt', pickle_protocol=4)


all_hr_data = read_data(hr_path)
all_lr_data = read_data(lr_path)
data_load(all_lr_data,all_hr_data)
