import torch
from os.path import isfile, join
from os import listdir
import os
from skimage.measure import label, regionprops
import scipy
import pickle
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
import random

IMG_PATH='/Users/piotrwojcik/Downloads/Fold 2/images/fold2/images.npy'
MAKS_PATH='/Users/piotrwojcik/Downloads/Fold 2/masks/fold2/masks.npy'
IMG_OUTPUT_DIR = '/Users/piotrwojcik/Downloads/pannuke/img_dir/val/'
ANN_OUTPUT_DIR = '/Users/piotrwojcik/Downloads/pannuke/ann_dir/val/'


palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34],
           [0, 11, 123], [0, 0, 0]]


if __name__ == '__main__':
    images = torch.tensor(np.load(IMG_PATH, allow_pickle=True))
    images = images.permute(0, 3, 1, 2)
    images = images.to(torch.uint8)
    masks = torch.tensor(np.load(MAKS_PATH, allow_pickle=True))
    masks = masks.permute(0, 3, 1, 2)
    print('Start')
    for i in range(masks.shape[0]):
        out = torch.zeros(3, 256, 256)

        for j in range(5):
            bck_mask = masks[i, j, :, :] != 0
            c_m = torch.zeros(3, 256, 256, dtype=torch.uint8)
            c_m += torch.tensor(palette[j])[:, None, None]
            c_m *= bck_mask.int()
            c_m *= (out[0] == 0).int()
            out += c_m
        #m = m.to(torch.uint8).unsqueeze(0).repeat(3, 1, 1)
        im = T.ToPILImage()(out)
        nim = T.ToPILImage()(images[i])
        im.save(os.path.join(ANN_OUTPUT_DIR, f"{i}.png"))
        nim.save(os.path.join(IMG_OUTPUT_DIR, f"{i}.png"))

