from patchify import patchify
import numpy as np
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("hr_data", type=str, help="HR Path")
parser.add_argument("lr_data", type=str, help="LR Path")
parser.add_argument("res_hr_patches", type=str, help="res_hr_patches path")
parser.add_argument("res_lr_patches", type=str, help="res_lr_patches path")

# Read arguments from command line
args = parser.parse_args()

hr_path = args.hr_data
lr_path = args.lr_data
res_hr_patches = args.res_hr_patches
res_lr_patches = args.res_lr_patches

def create_patches(path,respath):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            f = os.path.join(dirname, filename)
            # print(f)
            if filename.split('.')[-1] == 'jpg':
                image = Image.open(f)
                image = np.asarray(image)
                patches = patchify(image, (image.shape[0]//4, image.shape[1]//4, 3), step=(image.shape[1]//8))
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        patch = patches[i, j, k]
                        patch = Image.fromarray(patch)
                        num = i * patches.shape[1] + j
                        patch.save(f"{respath}/{filename.split('.')[0]}_patch_{num}.jpg")

create_patches(lr_path,res_lr_patches)
create_patches(hr_path,res_hr_patches)
